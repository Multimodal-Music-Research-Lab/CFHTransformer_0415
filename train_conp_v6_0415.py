"""
CFT v6_0415 训练脚本 — train_conp.py
==============================

基于 v6_0415/train_COn.py 改写。

修改说明：
  [修改1] 最佳模型选择标准改为 COnP F1 最大（onset + pitch，不看 offset）
  [修改2] 在 run/<timestamp>_COnP/ 下额外记录 test_monitor.txt
          仅用于 holdout test 监控，不参与阈值搜索或最佳模型选择
  [修改3] 配置里的本地路径按 config 文件所在目录解析，便于独立打包
"""

import argparse
import logging
import sys
import os
import random
import shutil
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json
import yaml

try:
    import mir_eval
    from mir_eval import transcription as mir_transcription, util as mir_util
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("WARNING: mir_eval not found, F1 metrics will be 0")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from model import CFT_v6 as CFT_v2, CFTLoss
from dataset import MIR_ST500_Dataset, MIDI_MIN, NUM_PITCHES


THRESHOLD_WORKER_DATA = None
THRESHOLD_WORKER_HOP = None
THRESHOLD_WORKER_SR = None

BEST_SLOT_RULES = {
    'con': {
        'slots': ['con1.pt', 'con2.pt'],
        'sort_keys': ('COn_f1', 'COnP_f1', 'COnPOff_f1'),
    },
    'conp': {
        'slots': ['conp1.pt', 'conp2.pt'],
        'sort_keys': ('COnP_f1', 'COnPOff_f1', 'COn_f1'),
    },
    'conpoff': {
        'slots': ['conpoff1.pt'],
        'sort_keys': ('COnPOff_f1', 'COnP_f1', 'COn_f1'),
    },
}
CHECKPOINT_SLOT_NAMES = [
    slot_name
    for rule in BEST_SLOT_RULES.values()
    for slot_name in rule['slots']
]


def pick_peak_frames(curve, thresh):
    """Collapse a contiguous above-threshold region to its strongest frame."""
    candidates = np.where(curve > thresh)[0]
    if len(candidates) == 0:
        return candidates

    picked = []
    start = prev = int(candidates[0])
    for frame in candidates[1:]:
        frame = int(frame)
        if frame == prev + 1:
            prev = frame
            continue
        local = curve[start:prev + 1]
        picked.append(start + int(np.argmax(local)))
        start = prev = frame

    local = curve[start:prev + 1]
    picked.append(start + int(np.argmax(local)))
    return np.array(picked, dtype=np.int64)


def pick_onset_frames(onset_curve, onset_thresh):
    return pick_peak_frames(onset_curve, onset_thresh)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(run_dir / 'train_stdout.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def resolve_config_paths(config, config_dir):
    config_dir = Path(config_dir).resolve()

    def resolve_value(value):
        if not value:
            return value
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((config_dir / path).resolve())

    path_fields = [
        ('data', 'audio_dir'),
        ('data', 'label_path'),
        ('data', 'splits_dir'),
        ('data', 'cqt_cache_dir'),
        ('model', 'wav2vec_path'),
        ('training', 'run_dir'),
    ]
    for section, key in path_fields:
        value = config.get(section, {}).get(key)
        if value:
            config[section][key] = resolve_value(value)

    for source in config.get('data', {}).get('train_mix_sources', []) or []:
        for key in ('audio_dir', 'label_path', 'splits_dir', 'cqt_cache_dir'):
            if source.get(key):
                source[key] = resolve_value(source[key])

    return config


def init_monitor_leaderboards():
    return {name: [] for name in BEST_SLOT_RULES}


def slot_label(slot_name):
    return slot_name[:-3] if slot_name.endswith('.pt') else slot_name


def build_checkpoint_filename(slot_name, record):
    label = slot_label(slot_name)
    epoch = int(record['epoch'])

    if label == 'last':
        return f"last_valconp{record['COnP_f1']:.6f}_epoch{epoch:04d}.pt"
    if label.startswith('conpoff'):
        score = record['COnPOff_f1']
    elif label.startswith('conp'):
        score = record['COnP_f1']
    elif label.startswith('con'):
        score = record['COn_f1']
    else:
        raise ValueError(f"Unsupported checkpoint slot: {slot_name}")

    return f"{label}_{score:.6f}_epoch{epoch:04d}.pt"


def _record_sort_key(record, sort_keys):
    return tuple(record[key] for key in sort_keys) + (record['epoch'],)


def update_monitor_leaderboards(leaderboards, monitor_record):
    updated = {}
    for name, rule in BEST_SLOT_RULES.items():
        records = [
            dict(item)
            for item in leaderboards.get(name, [])
            if item['epoch'] != monitor_record['epoch']
        ]
        records.append(dict(monitor_record))
        records.sort(
            key=lambda item: _record_sort_key(item, rule['sort_keys']),
            reverse=True,
        )
        updated[name] = records[:len(rule['slots'])]
    return updated


def build_checkpoint_slot_state(leaderboards):
    slot_state = {}
    for name, rule in BEST_SLOT_RULES.items():
        ranked_records = leaderboards.get(name, [])
        for idx, slot_name in enumerate(rule['slots']):
            slot_state[slot_name] = (
                dict(ranked_records[idx]) if idx < len(ranked_records) else None
            )
    return slot_state


def build_checkpoint_keep_names(slot_state, last_record):
    keep_names = set()
    for slot_name, record in slot_state.items():
        if record is not None:
            keep_names.add(build_checkpoint_filename(slot_name, record))
    if last_record is not None:
        keep_names.add(build_checkpoint_filename('last.pt', last_record))
    return keep_names


def save_ranked_checkpoint_slots(save_dir, prev_slot_state, next_slot_state,
                                 current_epoch, checkpoint_payload, logger):
    save_dir = Path(save_dir)
    stage_dir = save_dir / '.slot_stage'
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    previous_sources = {}
    for slot_name, record in prev_slot_state.items():
        if record is None:
            continue
        slot_path = save_dir / build_checkpoint_filename(slot_name, record)
        if slot_path.exists():
            previous_sources.setdefault(int(record['epoch']), slot_path)

    staged_payloads = {}
    staged_slot_names = set()
    for slot_name, record in next_slot_state.items():
        if record is None:
            continue

        epoch = int(record['epoch'])
        payload_path = staged_payloads.get(epoch)
        if payload_path is None:
            payload_path = stage_dir / f'epoch_{epoch:04d}.pt'
            if epoch == current_epoch:
                torch.save(checkpoint_payload, payload_path)
            else:
                source_path = previous_sources.get(epoch)
                if source_path is None or not source_path.exists():
                    logger.warning(
                        f"Skip updating {slot_name}: missing source checkpoint for epoch {epoch}"
                    )
                    continue
                shutil.copy2(source_path, payload_path)
            staged_payloads[epoch] = payload_path

        slot_filename = build_checkpoint_filename(slot_name, record)
        slot_stage_path = stage_dir / slot_filename
        if slot_stage_path.exists():
            slot_stage_path.unlink()
        os.link(payload_path, slot_stage_path)
        staged_slot_names.add(slot_filename)

    for slot_filename in staged_slot_names:
        final_path = save_dir / slot_filename
        slot_stage_path = stage_dir / slot_filename
        if final_path.exists():
            final_path.unlink()
        os.replace(slot_stage_path, final_path)

    shutil.rmtree(stage_dir, ignore_errors=True)


def prune_checkpoint_dir(save_dir, keep_names):
    save_dir = Path(save_dir)
    for ckpt_path in save_dir.glob('*.pt'):
        if ckpt_path.name not in keep_names:
            ckpt_path.unlink(missing_ok=True)


def get_total_frames(input_tensor, labels, input_type):
    if input_type == 'cqt':
        return input_tensor.shape[1]
    return labels['frame'].shape[0]


def build_input_chunk(input_tensor, input_type, start, end, hop_length, infer_chunk):
    if input_type == 'cqt':
        chunk = input_tensor[:, start:end].unsqueeze(0)
        chunk_frames = end - start
        if chunk_frames < infer_chunk:
            chunk = torch.nn.functional.pad(chunk, (0, infer_chunk - chunk_frames))
        return chunk, chunk_frames

    sample_start = start * hop_length
    sample_end = end * hop_length
    chunk = input_tensor[sample_start:sample_end].unsqueeze(0)
    chunk_frames = end - start
    target_samples = infer_chunk * hop_length
    if chunk.shape[-1] < target_samples:
        chunk = torch.nn.functional.pad(chunk, (0, target_samples - chunk.shape[-1]))
    return chunk, chunk_frames


# ---------------------------------------------------------------------------
# 音符级 F1 评估（v3：对齐原论文 evaluate_github.py）
# ---------------------------------------------------------------------------
# 修正内容（相比 v2）：
#   1. frames_to_notes 的 onset 分支允许 2 帧间隙，避免长音符被抖动截断
#   2. compute_note_f1_single 改用 transcription.evaluate() 一次调用：
#      - pitch 转 Hz（原来直接用 MIDI 编号导致 pitch 距离计算错误）
#      - COn = Onset_F-measure（只看 onset，不看 pitch）
#      - COnP = F-measure_no_offset（onset + pitch）
#      - COnPOff = F-measure（onset + pitch + offset）
#   3. ref 直接从 JSON 标注读取，不再从帧标签反推
# ---------------------------------------------------------------------------

def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=0.5, frame_thresh=0.5, min_note_len=2):
    """帧级预测 → 音符列表，返回 (intervals, pitches)，pitches 为 MIDI 编号"""
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    intervals = []
    pitches = []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = pick_onset_frames(onset_pred[:, p], onset_thresh)

        if len(onset_frames) == 0:
            # 纯帧模式
            active = frame_pred[:, p] > frame_thresh
            in_note, note_start = False, 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note, note_start = True, t
                elif not active[t] and in_note:
                    in_note = False
                    if t - note_start >= min_note_len:
                        intervals.append([note_start * frame_time, t * frame_time])
                        pitches.append(float(midi))
            if in_note and T - note_start >= min_note_len:
                intervals.append([note_start * frame_time, T * frame_time])
                pitches.append(float(midi))
        else:
            # onset 引导模式：允许最多 2 帧间隙，避免长音符被抖动截断
            for i, f_on in enumerate(onset_frames):
                next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else T
                f_off, gap = f_on, 0
                for t in range(f_on, min(next_onset, T)):
                    if frame_pred[t, p] > frame_thresh:
                        f_off = t
                        gap = 0
                    else:
                        gap += 1
                        if gap > 2 and t > f_on + 1:
                            break
                if f_off - f_on + 1 >= min_note_len:
                    intervals.append([f_on * frame_time, (f_off + 1) * frame_time])
                    pitches.append(float(midi))

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(intervals), np.array(pitches, dtype=float)


def estimate_frame_end(frame_curve, start, stop, frame_thresh, max_gap):
    """Fallback end from the frame branch: last active frame before a gap."""
    last_active = start
    gap = 0

    for t in range(start, stop):
        if frame_curve[t] > frame_thresh:
            last_active = t
            gap = 0
        else:
            gap += 1
            if gap > max_gap and t > start + 1:
                break

    return min(last_active + 1, stop)


def frames_to_notes_offset(frame_pred, onset_pred, offset_pred,
                           hop_length, sample_rate,
                           onset_thresh=0.5, frame_thresh=0.5,
                           offset_thresh=0.3, min_note_len=2,
                           max_gap=2):
    """Offset-aware post-processing used in threshold search and validation."""
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    intervals = []
    pitches = []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = pick_peak_frames(onset_pred[:, p], onset_thresh)
        offset_frames = pick_peak_frames(offset_pred[:, p], offset_thresh)

        if len(onset_frames) == 0:
            active = frame_pred[:, p] > frame_thresh
            in_note, note_start = False, 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note, note_start = True, t
                elif not active[t] and in_note:
                    in_note = False
                    if t - note_start >= min_note_len:
                        intervals.append([note_start * frame_time, t * frame_time])
                        pitches.append(float(midi))
            if in_note and T - note_start >= min_note_len:
                intervals.append([note_start * frame_time, T * frame_time])
                pitches.append(float(midi))
            continue

        for i, f_on in enumerate(onset_frames):
            next_onset = int(onset_frames[i + 1]) if i + 1 < len(onset_frames) else T
            search_start = int(f_on) + min_note_len
            search_stop = min(next_onset, T)

            valid_offsets = offset_frames[
                (offset_frames >= search_start) & (offset_frames < search_stop)
            ]
            if len(valid_offsets) > 0:
                end_frame = int(valid_offsets[0])
            else:
                end_frame = estimate_frame_end(
                    frame_pred[:, p], int(f_on), search_stop, frame_thresh, max_gap
                )

            if end_frame - int(f_on) >= min_note_len:
                intervals.append([int(f_on) * frame_time, end_frame * frame_time])
                pitches.append(float(midi))

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(intervals), np.array(pitches, dtype=float)


def compute_note_f1_single(pred_intervals, pred_pitches_midi,
                            ref_intervals, ref_pitches_midi,
                            onset_tolerance=0.05):
    """
    对齐原论文 evaluate_github.py 的评估逻辑。
    输入 pitches 为 MIDI 编号，内部转 Hz 后调用 transcription.evaluate()。
    返回 (COn_f1, COnP_f1, COnPOff_f1)
    """
    if not HAS_MIR_EVAL:
        return 0.0, 0.0, 0.0

    # 过滤 duration <= 0 的音符
    valid_pred = pred_intervals[:, 1] - pred_intervals[:, 0] > 0
    valid_ref  = ref_intervals[:, 1]  - ref_intervals[:, 0]  > 0
    pred_intervals  = pred_intervals[valid_pred]
    pred_pitches_midi = pred_pitches_midi[valid_pred]
    ref_intervals   = ref_intervals[valid_ref]
    ref_pitches_midi  = ref_pitches_midi[valid_ref]

    if len(ref_intervals) == 0:
        return None, None, None
    if len(pred_intervals) == 0:
        return 0.0, 0.0, 0.0

    # pitch 转 Hz（mir_eval 要求 Hz 输入）
    pred_pitches_hz = mir_util.midi_to_hz(pred_pitches_midi)
    ref_pitches_hz  = mir_util.midi_to_hz(ref_pitches_midi)

    try:
        raw = mir_transcription.evaluate(
            ref_intervals, ref_pitches_hz,
            pred_intervals, pred_pitches_hz,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=50,
        )
        con_f1     = raw['Onset_F-measure']
        conp_f1    = raw['F-measure_no_offset']
        conpoff_f1 = raw['F-measure']
    except Exception:
        con_f1 = conp_f1 = conpoff_f1 = 0.0

    return con_f1, conp_f1, conpoff_f1


def init_threshold_worker(preds, hop_length, sample_rate):
    """Share cached validation predictions with forked CPU workers."""
    global THRESHOLD_WORKER_DATA, THRESHOLD_WORKER_HOP, THRESHOLD_WORKER_SR
    THRESHOLD_WORKER_DATA = preds
    THRESHOLD_WORKER_HOP = hop_length
    THRESHOLD_WORKER_SR = sample_rate


def score_threshold_combo(combo):
    """Score one threshold combo on cached CPU numpy predictions."""
    if len(combo) == 2:
        onset_thresh, frame_thresh = combo
        offset_thresh = None
    else:
        onset_thresh, frame_thresh, offset_thresh = combo

    con_list = []
    conp_list = []
    conpoff_list = []

    for frame_sig, onset_sig, offset_sig, ref_intervals, ref_pitches in THRESHOLD_WORKER_DATA:
        if offset_thresh is None:
            pred_intervals, pred_pitches = frames_to_notes(
                frame_sig,
                onset_sig,
                THRESHOLD_WORKER_HOP,
                THRESHOLD_WORKER_SR,
                onset_thresh,
                frame_thresh,
            )
        else:
            pred_intervals, pred_pitches = frames_to_notes_offset(
                frame_sig,
                onset_sig,
                offset_sig,
                THRESHOLD_WORKER_HOP,
                THRESHOLD_WORKER_SR,
                onset_thresh,
                frame_thresh,
                offset_thresh,
            )

        con, conp, conpoff = compute_note_f1_single(
            pred_intervals, pred_pitches, ref_intervals, ref_pitches
        )
        if conp is not None:
            con_list.append(con)
            conp_list.append(conp)
            conpoff_list.append(conpoff)

    return {
        'onset': onset_thresh,
        'frame': frame_thresh,
        'offset': offset_thresh,
        'con': float(np.mean(con_list)) if con_list else 0.0,
        'conp': float(np.mean(conp_list)) if conp_list else 0.0,
        'conpoff': float(np.mean(conpoff_list)) if conpoff_list else 0.0,
    }


def score_threshold_combos(combos, preds, hop_length, sample_rate, threshold_workers):
    if not combos:
        return []

    n_workers = max(1, min(int(threshold_workers), len(combos)))
    init_args = (preds, hop_length, sample_rate)
    if n_workers == 1:
        init_threshold_worker(*init_args)
        return [score_threshold_combo(combo) for combo in combos]

    ctx = mp.get_context('fork')
    with ctx.Pool(
        processes=n_workers,
        initializer=init_threshold_worker,
        initargs=init_args,
    ) as pool:
        return pool.map(score_threshold_combo, combos)


# ---------------------------------------------------------------------------
# 训练 epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, epoch, logger,
                grad_clip=1.0, max_batches=None, scaler=None):
    model.train()
    total_loss = 0.0
    onset_loss_sum = 0.0
    frame_loss_sum = 0.0
    offset_loss_sum = 0.0
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)

    for batch_idx, (inputs, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        onset_label = labels['onset'].to(device)
        frame_label = labels['frame'].to(device)
        offset_label = labels['offset'].to(device)

        optimizer.zero_grad()
        with autocast():
            onset_pred, frame_pred, offset_pred = model(inputs)
            loss, onset_loss, frame_loss, offset_loss = criterion(
                onset_pred, frame_pred, offset_pred,
                onset_label, frame_label, offset_label
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        onset_loss_sum += onset_loss.item()
        frame_loss_sum += frame_loss.item()
        offset_loss_sum += offset_loss.item()

        if (batch_idx + 1) % max(1, n_batches // 3) == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                f"loss={loss.item():.4f} "
                f"onset={onset_loss.item():.4f} "
                f"frame={frame_loss.item():.4f} "
                f"offset={offset_loss.item():.4f}"
            )

    return {
        'total': total_loss / n_batches,
        'onset': onset_loss_sum / n_batches,
        'frame': frame_loss_sum / n_batches,
        'offset': offset_loss_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# 验证（全曲评估）
# ---------------------------------------------------------------------------

def validate_full_song(model, val_dataset, criterion, device, hop_length, sample_rate,
                       onset_thresh=0.5, frame_thresh=0.5, offset_thresh=0.3,
                       infer_chunk=256,
                       gt_annotations=None, input_type='cqt'):
    """
    全曲验证。
    gt_annotations: dict {song_id: [[onset, offset, midi], ...]}，用于 ref 音符。
    如果为 None，则从帧标签反推（不推荐）。
    """
    model.eval()
    total_loss = 0.0
    n_songs = 0
    con_f1_list = []
    conp_f1_list = []
    conpoff_f1_list = []
    onset_sig_list = []
    frame_sig_list = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            inputs, labels, song_id = val_dataset[idx]
            T_total = get_total_frames(inputs, labels, input_type)

            onset_lbl = labels['onset'].numpy()
            frame_lbl = labels['frame'].numpy()
            offset_lbl = labels['offset'].numpy()

            onset_sig_chunks = []
            frame_sig_chunks = []
            offset_sig_chunks = []
            chunk_losses = []

            for start in range(0, T_total, infer_chunk):
                end = min(start + infer_chunk, T_total)
                input_chunk, chunk_T = build_input_chunk(
                    inputs, input_type, start, end, hop_length, infer_chunk
                )
                input_chunk = input_chunk.to(device)

                onset_pred, frame_pred, offset_pred = model(input_chunk)

                onset_pred = onset_pred[:, :chunk_T, :]
                frame_pred = frame_pred[:, :chunk_T, :]
                offset_pred = offset_pred[:, :chunk_T, :]

                ol_chunk = torch.from_numpy(onset_lbl[start:end]).unsqueeze(0).to(device)
                fl_chunk = torch.from_numpy(frame_lbl[start:end]).unsqueeze(0).to(device)
                ofl_chunk = torch.from_numpy(offset_lbl[start:end]).unsqueeze(0).to(device)
                loss, _, _, _ = criterion(onset_pred, frame_pred, offset_pred,
                                          ol_chunk, fl_chunk, ofl_chunk)
                chunk_losses.append(loss.item())

                onset_sig_chunks.append(torch.sigmoid(onset_pred[0]).cpu().numpy())
                frame_sig_chunks.append(torch.sigmoid(frame_pred[0]).cpu().numpy())
                offset_sig_chunks.append(torch.sigmoid(offset_pred[0]).cpu().numpy())

            onset_sig = np.concatenate(onset_sig_chunks, axis=0)
            frame_sig = np.concatenate(frame_sig_chunks, axis=0)
            offset_sig = np.concatenate(offset_sig_chunks, axis=0)

            total_loss += float(np.mean(chunk_losses))
            n_songs += 1

            onset_sig_list.append(onset_sig.mean())
            frame_sig_list.append(frame_sig.mean())

            # 预测音符
            pred_intervals, pred_pitches = frames_to_notes_offset(
                frame_sig, onset_sig, offset_sig,
                hop_length, sample_rate, onset_thresh, frame_thresh, offset_thresh
            )

            # ref 音符：优先从 JSON 标注读取，否则从帧标签反推
            if gt_annotations is not None and song_id in gt_annotations:
                raw = gt_annotations[song_id]
                ref_notes = [[float(n[0]), float(n[1]), float(n[2])] for n in raw
                             if float(n[1]) - float(n[0]) > 0]
                if len(ref_notes) == 0:
                    continue
                ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
                ref_pitches   = np.array([n[2] for n in ref_notes])
            else:
                # 备用：从帧标签反推（不推荐，仅当没有 JSON 时使用）
                ref_intervals, ref_pitches = frames_to_notes(
                    frame_lbl.astype(np.float32), onset_lbl.astype(np.float32),
                    hop_length, sample_rate, onset_thresh=0.5, frame_thresh=0.5
                )

            if len(ref_intervals) == 0:
                continue

            con_f1, conp_f1, conpoff_f1 = compute_note_f1_single(
                pred_intervals, pred_pitches,
                ref_intervals, ref_pitches
            )
            if con_f1 is not None:
                con_f1_list.append(con_f1)
                conp_f1_list.append(conp_f1)
                conpoff_f1_list.append(conpoff_f1)

    avg_loss = total_loss / max(n_songs, 1)
    avg_con_f1 = float(np.mean(con_f1_list)) if con_f1_list else 0.0
    avg_conp_f1 = float(np.mean(conp_f1_list)) if conp_f1_list else 0.0
    avg_conpoff_f1 = float(np.mean(conpoff_f1_list)) if conpoff_f1_list else 0.0
    avg_onset_sig = float(np.mean(onset_sig_list)) if onset_sig_list else 0.0
    avg_frame_sig = float(np.mean(frame_sig_list)) if frame_sig_list else 0.0

    return avg_loss, avg_con_f1, avg_conp_f1, avg_conpoff_f1, avg_onset_sig, avg_frame_sig


def find_best_threshold(model, val_dataset, criterion, device, hop_length, sample_rate,
                        logger, gt_annotations=None, input_type='cqt',
                        infer_chunk=256, threshold_workers=1,
                        top_k_onset_frame=2):
    n_search = min(30, len(val_dataset))
    model.eval()
    preds = []  # (frame_sig, onset_sig, offset_sig, ref_intervals, ref_pitches)
    with torch.no_grad():
        for idx in range(n_search):
            inputs, labels, song_id = val_dataset[idx]
            T_total = get_total_frames(inputs, labels, input_type)
            onset_chunks, frame_chunks, offset_chunks = [], [], []
            for start in range(0, T_total, infer_chunk):
                end = min(start + infer_chunk, T_total)
                input_chunk, chunk_T = build_input_chunk(
                    inputs, input_type, start, end, hop_length, infer_chunk
                )
                input_chunk = input_chunk.to(device)
                op, fp, ofp = model(input_chunk)
                onset_chunks.append(torch.sigmoid(op[0, :chunk_T]).cpu().numpy())
                frame_chunks.append(torch.sigmoid(fp[0, :chunk_T]).cpu().numpy())
                offset_chunks.append(torch.sigmoid(ofp[0, :chunk_T]).cpu().numpy())
            onset_sig = np.concatenate(onset_chunks, axis=0)
            frame_sig = np.concatenate(frame_chunks, axis=0)
            offset_sig = np.concatenate(offset_chunks, axis=0)

            # ref 音符：优先从 JSON 标注读取
            if gt_annotations is not None and song_id in gt_annotations:
                raw = gt_annotations[song_id]
                ref_notes = [[float(n[0]), float(n[1]), float(n[2])] for n in raw
                             if float(n[1]) - float(n[0]) > 0]
                if len(ref_notes) == 0:
                    continue
                ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
                ref_pitches   = np.array([n[2] for n in ref_notes])
            else:
                onset_lbl = labels['onset'].numpy()
                frame_lbl = labels['frame'].numpy()
                ref_intervals, ref_pitches = frames_to_notes(
                    frame_lbl.astype(np.float32), onset_lbl.astype(np.float32),
                    hop_length, sample_rate, onset_thresh=0.5, frame_thresh=0.5
                )
            preds.append((frame_sig, onset_sig, offset_sig, ref_intervals, ref_pitches))

    if not preds:
        logger.info('  Threshold search skipped: no valid validation predictions')
        return 0.45, 0.50, 0.10

    onset_thresholds = [round(x * 0.05, 2) for x in range(2, 21)]
    frame_thresholds = [round(x * 0.05, 2) for x in range(2, 17)]
    offset_thresholds = [round(x * 0.05, 2) for x in range(0, 14)]

    onset_frame_combos = [
        (ot, ft)
        for ot in onset_thresholds
        for ft in frame_thresholds
    ]
    stage1_results = score_threshold_combos(
        onset_frame_combos,
        preds,
        hop_length,
        sample_rate,
        threshold_workers,
    )
    top_onset_frame = sorted(
        stage1_results,
        key=lambda item: (item['conp'], item['con'], item['conpoff']),
        reverse=True,
    )[:max(1, min(top_k_onset_frame, len(stage1_results)))]

    offset_combos = [
        (item['onset'], item['frame'], oft)
        for item in top_onset_frame
        for oft in offset_thresholds
    ]
    stage2_results = score_threshold_combos(
        offset_combos,
        preds,
        hop_length,
        sample_rate,
        threshold_workers,
    )
    best = max(stage2_results, key=lambda item: (item['conpoff'], item['conp'], item['con']))
    best_ot = best['onset']
    best_ft = best['frame']
    best_oft = best['offset']

    logger.info(
        f"  Threshold search: best on={best_ot:.2f}, fr={best_ft:.2f}, off={best_oft:.2f}, "
        f"COn_f1={best['con']:.4f}, COnP_f1={best['conp']:.4f}, "
        f"COnPOff_f1={best['conpoff']:.4f}, "
        f"stage1={len(onset_frame_combos)} combos, stage2={len(offset_combos)} combos, "
        f"workers={max(1, min(int(threshold_workers), max(len(onset_frame_combos), 1)))}"
    )
    return best_ot, best_ft, best_oft


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = resolve_config_paths(config, config_path.parent)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_type = config['data'].get('input_type', 'cqt')

    # 按启动时间生成独立 run 目录，格式：run/<时间戳>_COn/
    # 每次启动都会创建全新子目录，多个进程并发训练时互不干扰
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{run_timestamp}_COnP"
    run_base = Path(config['training'].get('run_dir', './run'))
    run_dir = run_base / run_name
    save_dir = run_dir / 'checkpoints'
    log_dir  = run_dir / 'logs'
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pid_file = Path(f'/tmp/cft_v6_COnP_{run_timestamp}.pid')
    pid_file.write_text(str(os.getpid()))

    logger = setup_logger(log_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    logger.info("=" * 60)
    logger.info("CFT v6_0415 — train_conp.py (best model: COnP F1)")
    logger.info("=" * 60)

    # 加载 JSON 标注（用于验证时 ref 音符）
    gt_json_path = config['data']['label_path']
    with open(gt_json_path) as f:
        gt_annotations = json.load(f)
    logger.info(f"Loaded GT annotations from {gt_json_path}")

    # 数据集
    train_dataset = MIR_ST500_Dataset(config, split='train')
    val_dataset = MIR_ST500_Dataset(config, split='val')
    test_dataset = MIR_ST500_Dataset(config, split='test')
    # Test 仅用于监控，不参与阈值搜索或 best model 选择。
    # 这里直接使用全量 test。
    test_monitor_dataset = test_dataset
    logger.info(
        f"Train samples: {len(train_dataset)}, Val songs: {len(val_dataset)}, "
        f"Test songs: {len(test_dataset)}, Test monitor songs: {len(test_monitor_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=config['training']['num_workers'] > 0,
        prefetch_factor=4 if config['training']['num_workers'] > 0 else None
    )

    # 模型
    model = CFT_v2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info("Tokenization: continuous kernels [3,5,7] (paper-aligned, no dilation)")

    # 混合精度
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        logger.info("Mixed precision (AMP) enabled")

    # 损失函数（论文公式1 + 正样本加权修复类别不平衡）
    criterion = CFTLoss(
        onset_weight=config['loss']['onset_weight'],
        frame_weight=config['loss']['frame_weight'],
        offset_weight=config['loss']['offset_weight'],
        onset_pos_weight=config['loss'].get('onset_pos_weight', 1.0),
        frame_pos_weight=config['loss'].get('frame_pos_weight', 1.0),
        offset_pos_weight=config['loss'].get('offset_pos_weight', 1.0),
    ).to(device)

    # 优化器（论文 Section 3.3：Adam, lr=3e-4）
    learning_rate = config['training']['learning_rate']
    backbone_lr = config['training'].get('backbone_learning_rate')
    if getattr(model, 'frontend_type', 'cqt') == 'wav2vec2' and backbone_lr is not None:
        backbone_params = [p for p in model.frontend.get_backbone_parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]
        optimizer = Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': backbone_params, 'lr': backbone_lr},
        ])
        logger.info(f"Optimizer: Adam(frontend={backbone_lr:.2e}, others={learning_rate:.2e})")
    else:
        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate
        )

    # 学习率调度：论文未提及，使用 CosineAnnealingLR（无 warmup）
    total_epochs = config['training']['epochs']
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6
    )
    logger.info(f"Scheduler: CosineAnnealingLR (no warmup, T_max={total_epochs})")

    writer = None
    if HAS_TB:
        writer = SummaryWriter(str(log_dir / 'tensorboard'))

    start_epoch = 1
    best_conp_f1 = 0.0
    best_onset_thresh = 0.45
    best_frame_thresh = 0.50
    best_offset_thresh = 0.10
    epochs_since_ranked_update = 0
    last_ranked_update_epoch = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_conp_f1 = ckpt.get('best_conp_f1', ckpt.get('best_con_f1', 0.0))
        best_onset_thresh = ckpt.get('best_onset_thresh', 0.45)
        best_frame_thresh = ckpt.get('best_frame_thresh', 0.50)
        best_offset_thresh = ckpt.get('best_offset_thresh', 0.10)
        epochs_since_ranked_update = ckpt.get('epochs_since_ranked_update', 0)
        last_ranked_update_epoch = ckpt.get('last_ranked_update_epoch', 0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_COnP_f1={best_conp_f1:.4f}")

    hop_length = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']
    infer_chunk = config['data'].get('infer_chunk_frames', config['data']['segment_frames'])

    max_samples = config['data'].get('max_samples_per_epoch', None)
    max_batches = None
    if max_samples:
        max_batches = max(1, max_samples // config['training']['batch_size'])
        logger.info(f"Max batches per epoch: {max_batches}")

    threshold_search_every = config['training'].get('threshold_search_every', 2)
    threshold_workers = config['training'].get('threshold_workers', 12)
    threshold_top_k_onset_frame = config['training'].get('threshold_top_k_onset_frame', 2)
    ranked_update_patience = config['training'].get('ranked_update_patience', 50)
    logger.info(
        f"Threshold search: every={threshold_search_every}, "
        f"workers={threshold_workers}, top_k_onset_frame={threshold_top_k_onset_frame}"
    )
    logger.info(
        f"Ranked checkpoint early stop patience: {ranked_update_patience} epochs"
    )

    # full test 监控：仅诊断，不参与阈值搜索或 best model 选择
    test_monitor_path = run_dir / 'test_monitor.txt'
    if not test_monitor_path.exists():
        with open(test_monitor_path, 'w') as tf:
            tf.write("epoch\ttrigger\ttest_loss\tCOn_f1\tCOnP_f1\tCOnPOff_f1\tonset_thresh\tframe_thresh\toffset_thresh\n")

    monitor_leaderboards = init_monitor_leaderboards()
    checkpoint_slot_state = build_checkpoint_slot_state(monitor_leaderboards)

    for epoch in range(start_epoch, total_epochs + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            grad_clip=config['training']['grad_clip'],
            max_batches=max_batches,
            scaler=scaler
        )
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        if epoch % threshold_search_every == 0 or epoch == 1:
            best_onset_thresh, best_frame_thresh, best_offset_thresh = find_best_threshold(
                model, val_dataset, criterion, device, hop_length, sample_rate,
                logger, gt_annotations=gt_annotations,
                input_type=input_type, infer_chunk=infer_chunk,
                threshold_workers=threshold_workers,
                top_k_onset_frame=threshold_top_k_onset_frame
            )

        val_loss, con_f1, conp_f1, conpoff_f1, onset_sig, frame_sig = validate_full_song(
            model, val_dataset, criterion, device, hop_length, sample_rate,
            onset_thresh=best_onset_thresh, frame_thresh=best_frame_thresh,
            offset_thresh=best_offset_thresh,
            gt_annotations=gt_annotations, input_type=input_type, infer_chunk=infer_chunk
        )

        logger.info(
            f"Epoch {epoch}/{total_epochs} | "
            f"train_loss={train_losses['total']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"COn_f1={con_f1:.4f} | "
            f"COnP_f1={conp_f1:.4f} | "
            f"COnPOff_f1={conpoff_f1:.4f} | "
            f"sig_onset={onset_sig:.4f} sig_frame={frame_sig:.4f} | "
            f"thresh(on={best_onset_thresh:.2f},fr={best_frame_thresh:.2f},off={best_offset_thresh:.2f}) | "
            f"lr={lr:.2e}"
        )
        if device.type == 'cuda':
            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            logger.info(
                f"  GPU peak memory: allocated={peak_alloc:.2f} GiB | "
                f"reserved={peak_reserved:.2f} GiB"
            )

        if writer:
            writer.add_scalar('Loss/train', train_losses['total'], epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/COn_f1', con_f1, epoch)
            writer.add_scalar('Metrics/COnP_f1', conp_f1, epoch)
            writer.add_scalar('Metrics/COnPOff_f1', conpoff_f1, epoch)
            writer.add_scalar('Thresholds/onset', best_onset_thresh, epoch)
            writer.add_scalar('Thresholds/frame', best_frame_thresh, epoch)
            writer.add_scalar('Thresholds/offset', best_offset_thresh, epoch)
            writer.add_scalar('LR', lr, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'val_loss': val_loss,
            'COn_f1': con_f1,
            'COnP_f1': conp_f1,
            'COnPOff_f1': conpoff_f1,
            'best_conp_f1': best_conp_f1,
            'best_onset_thresh': best_onset_thresh,
            'best_frame_thresh': best_frame_thresh,
            'best_offset_thresh': best_offset_thresh,
            'config': config
        }

        is_best_epoch = conp_f1 > best_conp_f1
        if is_best_epoch:
            best_conp_f1 = conp_f1
            ckpt['best_conp_f1'] = best_conp_f1
            logger.info(f"  -> New val-best epoch recorded. COnP_f1={best_conp_f1:.4f}")
        # full test 监控：每 40 个 epoch 固定评估一次；
        # 另外只要出现新的 best model，就从训练一开始额外评估一次。
        test_monitor_triggers = []
        if epoch % 40 == 0 or epoch == 1:
            test_monitor_triggers.append('periodic_40')
        if is_best_epoch:
            test_monitor_triggers.append('best')

        test_monitor_record = None
        ranked_slots_updated = False
        if test_monitor_triggers:
            test_mon_loss, test_con, test_conp, test_conpoff, _, _ = validate_full_song(
                model, test_monitor_dataset, criterion, device, hop_length, sample_rate,
                onset_thresh=best_onset_thresh, frame_thresh=best_frame_thresh,
                offset_thresh=best_offset_thresh,
                gt_annotations=gt_annotations, input_type=input_type, infer_chunk=infer_chunk
            )
            test_monitor_record = {
                'epoch': epoch,
                'trigger': '+'.join(test_monitor_triggers),
                'test_loss': test_mon_loss,
                'COn_f1': test_con,
                'COnP_f1': test_conp,
                'COnPOff_f1': test_conpoff,
                'onset_thresh': best_onset_thresh,
                'frame_thresh': best_frame_thresh,
                'offset_thresh': best_offset_thresh,
            }
            with open(test_monitor_path, 'a') as tf:
                tf.write(
                    f"{epoch}\t"
                    f"{'+'.join(test_monitor_triggers)}\t"
                    f"{test_mon_loss:.6f}\t"
                    f"{test_con:.6f}\t"
                    f"{test_conp:.6f}\t"
                    f"{test_conpoff:.6f}\t"
                    f"{best_onset_thresh:.2f}\t"
                    f"{best_frame_thresh:.2f}\t"
                    f"{best_offset_thresh:.2f}\n"
                )
            logger.info(
                f"  [TEST MONITOR FULL TEST:{'+'.join(test_monitor_triggers)}] "
                f"COn_f1={test_con:.4f} | "
                f"COnP_f1={test_conp:.4f} | "
                f"COnPOff_f1={test_conpoff:.4f}"
            )

            prev_slot_state = {
                slot_name: (dict(record) if record is not None else None)
                for slot_name, record in checkpoint_slot_state.items()
            }
            monitor_leaderboards = update_monitor_leaderboards(
                monitor_leaderboards, test_monitor_record
            )
            checkpoint_slot_state = build_checkpoint_slot_state(monitor_leaderboards)

            ranked_ckpt = dict(ckpt)
            ranked_ckpt['test_monitor_record'] = test_monitor_record
            ranked_ckpt['test_monitor_leaderboards'] = monitor_leaderboards
            ranked_ckpt['checkpoint_slot_state'] = checkpoint_slot_state

            if checkpoint_slot_state != prev_slot_state:
                save_ranked_checkpoint_slots(
                    save_dir=save_dir,
                    prev_slot_state=prev_slot_state,
                    next_slot_state=checkpoint_slot_state,
                    current_epoch=epoch,
                    checkpoint_payload=ranked_ckpt,
                    logger=logger,
                )
                ranked_slots_updated = True
                slot_summary = []
                for slot_name in CHECKPOINT_SLOT_NAMES:
                    record = checkpoint_slot_state.get(slot_name)
                    if record is not None and int(record['epoch']) == epoch:
                        slot_summary.append(build_checkpoint_filename(slot_name, record))
                if slot_summary:
                    logger.info(
                        "  -> Ranked checkpoints updated from test100 monitor: "
                        + ", ".join(slot_summary)
                    )

        if ranked_slots_updated:
            epochs_since_ranked_update = 0
            last_ranked_update_epoch = epoch
        else:
            epochs_since_ranked_update += 1

        last_ckpt = dict(ckpt)
        last_ckpt['test_monitor_record'] = test_monitor_record
        last_ckpt['test_monitor_leaderboards'] = monitor_leaderboards
        last_ckpt['checkpoint_slot_state'] = checkpoint_slot_state
        last_ckpt['epochs_since_ranked_update'] = epochs_since_ranked_update
        last_ckpt['last_ranked_update_epoch'] = last_ranked_update_epoch
        last_ckpt['ranked_update_patience'] = ranked_update_patience
        last_filename = build_checkpoint_filename('last.pt', last_ckpt)
        torch.save(last_ckpt, save_dir / last_filename)
        keep_names = build_checkpoint_keep_names(checkpoint_slot_state, last_ckpt)
        prune_checkpoint_dir(save_dir, keep_names)

        if (
            ranked_update_patience > 0
            and last_ranked_update_epoch > 0
            and epochs_since_ranked_update >= ranked_update_patience
        ):
            logger.info(
                "Early stop triggered: "
                f"no ranked checkpoint update for {epochs_since_ranked_update} epochs "
                f"since epoch {last_ranked_update_epoch}."
            )
            break

    if writer:
        writer.close()

    logger.info(f"Training complete! Best COnP_f1: {best_conp_f1:.4f}")
    pid_file.unlink(missing_ok=True)
    logger.info(f"Run directory: {run_dir}")


if __name__ == '__main__':
    main()
