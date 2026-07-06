"""
MIR-ST500 dataset loader for CQT and wav2vec2 waveform frontends.

This version keeps the SSL baseline's waveform path and adds the current
mainline mixed-source training pipeline:
  - train_mix_sources for original + pitch-shift data
  - source-aware label caching so identical song ids do not mix annotations
  - onset labels widened to match the 50 ms onset tolerance
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


MIDI_MIN = 36
MIDI_MAX = 83
NUM_PITCHES = MIDI_MAX - MIDI_MIN + 1


class MIR_ST500_Dataset(Dataset):
    def __init__(self, config, split="train", max_songs=None):
        self.config = config
        self.split = split
        self.input_type = config["data"].get("input_type", "cqt")
        self.segment_frames = config["data"]["segment_frames"]
        self.hop_length = config["audio"]["hop_length"]
        self.sample_rate = config["data"]["sample_rate"]
        self.segment_samples = self.segment_frames * self.hop_length
        self._label_cache = {}
        self.audio_meta = {}

        self.sources = self._prepare_sources(config, split, max_songs)
        if not self.sources:
            raise RuntimeError(f"No valid data sources for split={split}")

        if split == "train":
            self._build_train_index()
        else:
            self.file_list = self.sources[0]["file_list"]

    def _prepare_sources(self, config, split, max_songs):
        data_cfg = config["data"]
        if split == "train" and data_cfg.get("train_mix_sources"):
            source_cfgs = data_cfg["train_mix_sources"]
        else:
            source_cfgs = [{
                "name": data_cfg.get("dataset_name", "original"),
                "audio_dir": data_cfg.get("audio_dir", ""),
                "label_path": data_cfg["label_path"],
                "splits_dir": data_cfg["splits_dir"],
                "cqt_cache_dir": data_cfg.get("cqt_cache_dir", ""),
            }]

        sources = []
        for source_idx, raw_source in enumerate(source_cfgs):
            source = {
                "name": raw_source.get("name", f"source{source_idx}"),
                "audio_dir": Path(raw_source.get("audio_dir", data_cfg.get("audio_dir", ""))),
                "label_path": Path(raw_source.get("label_path", data_cfg["label_path"])),
                "splits_dir": Path(raw_source.get("splits_dir", data_cfg["splits_dir"])),
                "cqt_cache_dir": Path(raw_source.get("cqt_cache_dir", data_cfg.get("cqt_cache_dir", ""))),
            }

            with open(source["label_path"], "r") as f:
                source["annotations"] = json.load(f)

            split_file = source["splits_dir"] / f"{split}.txt"
            with open(split_file, "r") as f:
                file_list = [line.strip() for line in f if line.strip()]
            if max_songs is not None:
                file_list = file_list[:max_songs]

            valid = []
            for sid in file_list:
                if self.input_type == "cqt":
                    if (source["cqt_cache_dir"] / f"{sid}.npy").exists():
                        valid.append(sid)
                elif self.input_type == "waveform":
                    meta = self._build_audio_meta(source, sid)
                    if meta is not None:
                        self.audio_meta[(source_idx, sid)] = meta
                        valid.append(sid)
                else:
                    raise ValueError(f"Unsupported input_type: {self.input_type}")

            if len(valid) < len(file_list):
                missing = len(file_list) - len(valid)
                print(f"Warning: {source['name']} missing {missing} songs in {split} split")

            source["file_list"] = valid
            sources.append(source)

        return sources

    def _resolve_audio_path(self, audio_dir: Path, song_id: str):
        candidates = [
            audio_dir / f"{song_id}_vocals.mp3",
            audio_dir / f"{song_id}_vocals.wav",
            audio_dir / f"{song_id}.mp3",
            audio_dir / f"{song_id}.wav",
            audio_dir / song_id / "Vocal.wav",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _build_audio_meta(self, source, song_id: str):
        path = self._resolve_audio_path(source["audio_dir"], song_id)
        if path is None:
            return None

        try:
            import soundfile as sf
            sf_info = sf.info(str(path))
            info_sr = sf_info.samplerate
            info_num_frames = sf_info.frames
        except Exception:
            try:
                info_waveform, info_sr = torchaudio.load(str(path), normalize=False)
                info_num_frames = info_waveform.shape[1]
            except Exception:
                return None

        target_num_samples = int(round(info_num_frames * self.sample_rate / info_sr))
        target_num_frames = max(1, math.ceil(target_num_samples / self.hop_length))
        return {
            "path": path,
            "orig_sr": info_sr,
            "orig_num_frames": info_num_frames,
            "target_num_samples": target_num_samples,
            "target_num_frames": target_num_frames,
        }

    def _get_num_frames(self, source_idx: int, song_id: str):
        source = self.sources[source_idx]
        if self.input_type == "cqt":
            cqt_path = source["cqt_cache_dir"] / f"{song_id}.npy"
            return int(np.load(str(cqt_path), mmap_mode="r").shape[1])
        return self.audio_meta[(source_idx, song_id)]["target_num_frames"]

    def _get_labels(self, source_idx: int, song_id: str, num_frames: int):
        cache_key = (source_idx, song_id, num_frames)
        cache = self._label_cache.get(cache_key)
        if cache is not None:
            return cache
        labels = self._create_labels(
            self.sources[source_idx]["annotations"],
            song_id,
            num_frames,
        )
        self._label_cache[cache_key] = labels
        return labels

    def _load_waveform_segment(self, source_idx: int, song_id: str,
                               start_frame: int, num_frames: int):
        meta = self.audio_meta[(source_idx, song_id)]
        target_start = start_frame * self.hop_length
        target_samples = num_frames * self.hop_length

        orig_start = int(round(target_start * meta["orig_sr"] / self.sample_rate))
        orig_num_frames = int(round(target_samples * meta["orig_sr"] / self.sample_rate))

        try:
            import soundfile as sf
            audio, sr = sf.read(
                str(meta["path"]),
                start=max(0, orig_start),
                frames=max(1, orig_num_frames),
                dtype="float32",
                always_2d=True,
            )
            waveform = torch.from_numpy(audio).mean(dim=1)
        except Exception:
            waveform, sr = torchaudio.load(
                str(meta["path"]),
                frame_offset=max(0, orig_start),
                num_frames=max(1, orig_num_frames),
            )
            waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if waveform.numel() < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
        else:
            waveform = waveform[:target_samples]
        return waveform

    def _load_full_waveform(self, source_idx: int, song_id: str):
        meta = self.audio_meta[(source_idx, song_id)]
        try:
            import soundfile as sf
            audio, sr = sf.read(
                str(meta["path"]),
                dtype="float32",
                always_2d=True,
            )
            waveform = torch.from_numpy(audio).mean(dim=1)
        except Exception:
            waveform, sr = torchaudio.load(str(meta["path"]))
            waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform

    def _build_train_index(self):
        """
        Build (source_idx, song_id, start_frame) entries.
        Each source is indexed independently, so augmented labels stay aligned
        with their corresponding audio.
        """
        self._train_index = []
        stride = self.segment_frames // 2
        oversample = self.config.get("data", {}).get("extreme_pitch_oversample", 0)
        low_thresh = 50
        high_thresh = 75

        for source_idx, source in enumerate(self.sources):
            annotations = source["annotations"]
            for sid in source["file_list"]:
                num_frames = self._get_num_frames(source_idx, sid)
                notes = annotations.get(sid, [])
                frame_time = self.hop_length / self.sample_rate
                note_frames = set()
                extreme_pitch_frames = set()
                for note in notes:
                    midi = int(note[2])
                    if not (MIDI_MIN <= midi <= MIDI_MAX):
                        continue
                    f_on = int(round(float(note[0]) / frame_time))
                    f_off = int(round(float(note[1]) / frame_time))
                    for frame_idx in range(f_on, min(f_off + 1, num_frames)):
                        note_frames.add(frame_idx)
                        if midi < low_thresh or midi > high_thresh:
                            extreme_pitch_frames.add(frame_idx)

                before = len(self._train_index)
                if num_frames <= self.segment_frames:
                    self._train_index.append((source_idx, sid, 0))
                    continue

                for start in range(0, num_frames - self.segment_frames + 1, stride):
                    end = start + self.segment_frames
                    segment_has_note = any(start <= f < end for f in note_frames)
                    if segment_has_note:
                        self._train_index.append((source_idx, sid, start))
                        if oversample > 0 and any(start <= f < end for f in extreme_pitch_frames):
                            for _ in range(oversample):
                                self._train_index.append((source_idx, sid, start))
                    elif random.random() < 0.15:
                        self._train_index.append((source_idx, sid, start))

                if len(self._train_index) == before:
                    self._train_index.append((source_idx, sid, 0))

    def __len__(self):
        if self.split == "train":
            return len(self._train_index)
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.split == "train":
            return self._get_train_item(idx)
        return self._get_full_song(idx)

    def _get_train_item(self, idx):
        source_idx, song_id, start = self._train_index[idx]
        jitter = random.randint(-self.segment_frames // 8, self.segment_frames // 8)
        num_frames = self._get_num_frames(source_idx, song_id)

        max_start = max(0, num_frames - self.segment_frames)
        start = max(0, min(start + jitter, max_start))
        end = start + self.segment_frames

        labels = self._get_labels(source_idx, song_id, num_frames)
        labels_seg = {k: v[start:end] for k, v in labels.items()}

        if self.input_type == "cqt":
            source = self.sources[source_idx]
            cqt = np.load(str(source["cqt_cache_dir"] / f"{song_id}.npy"), mmap_mode="r")
            input_tensor = torch.from_numpy(np.array(cqt[:, start:end], copy=True)).float()
        else:
            input_tensor = self._load_waveform_segment(
                source_idx, song_id, start, self.segment_frames
            ).float()

        label_tensors = {k: torch.from_numpy(v).float() for k, v in labels_seg.items()}
        return input_tensor, label_tensors

    def _get_full_song(self, idx):
        source_idx = 0
        song_id = self.sources[source_idx]["file_list"][idx]
        num_frames = self._get_num_frames(source_idx, song_id)
        labels = self._get_labels(source_idx, song_id, num_frames)

        if self.input_type == "cqt":
            source = self.sources[source_idx]
            cqt = np.load(str(source["cqt_cache_dir"] / f"{song_id}.npy"))
            input_tensor = torch.from_numpy(cqt).float()
        else:
            input_tensor = self._load_full_waveform(source_idx, song_id).float()

        label_tensors = {k: torch.from_numpy(v).float() for k, v in labels.items()}
        return input_tensor, label_tensors, song_id

    def _create_labels(self, annotations, song_id, num_frames):
        notes = annotations.get(song_id, [])
        frame_time = self.hop_length / self.sample_rate

        onset = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
        offset = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
        frame = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)

        onset_radius = max(1, round(0.05 / frame_time))

        for note in notes:
            t_on, t_off, midi = float(note[0]), float(note[1]), int(note[2])
            pitch_idx = midi - MIDI_MIN
            if not (0 <= pitch_idx < NUM_PITCHES):
                continue

            f_on = int(round(t_on / frame_time))
            f_off = int(round(t_off / frame_time))

            for df in range(-onset_radius, onset_radius + 1):
                f = f_on + df
                if 0 <= f < num_frames:
                    onset[f, pitch_idx] = 1.0
            if f_off < num_frames:
                offset[f_off, pitch_idx] = 1.0
            for frame_idx in range(f_on, min(f_off + 1, num_frames)):
                frame[frame_idx, pitch_idx] = 1.0

        return {"onset": onset, "offset": offset, "frame": frame}
