import argparse
import json
from pathlib import Path

import torch
import yaml

from dataset import MIR_ST500_Dataset
from model import CFTLoss, CFT_v6
from train_conp_v6_0415 import resolve_config_paths, set_seed, validate_full_song


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_mert_base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--onset", type=float, required=True)
    parser.add_argument("--frame", type=float, required=True)
    parser.add_argument("--offset", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    config_path = Path(args.config).resolve()
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config = resolve_config_paths(config, config_path.parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_type = config["data"].get("input_type", "cqt")
    hop_length = config["audio"]["hop_length"]
    sample_rate = config["data"]["sample_rate"]
    infer_chunk = config["data"].get("infer_chunk_frames", config["data"]["segment_frames"])

    with open(config["data"]["label_path"]) as f:
        gt_annotations = json.load(f)

    dataset = MIR_ST500_Dataset(config, split=args.split)
    model = CFT_v6(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = CFTLoss(
        onset_weight=config["loss"]["onset_weight"],
        frame_weight=config["loss"]["frame_weight"],
        offset_weight=config["loss"]["offset_weight"],
        onset_pos_weight=config["loss"].get("onset_pos_weight", 1.0),
        frame_pos_weight=config["loss"].get("frame_pos_weight", 1.0),
        offset_pos_weight=config["loss"].get("offset_pos_weight", 1.0),
    ).to(device)

    loss, con, conp, conpoff, onset_sig, frame_sig = validate_full_song(
        model,
        dataset,
        criterion,
        device,
        hop_length,
        sample_rate,
        onset_thresh=args.onset,
        frame_thresh=args.frame,
        offset_thresh=args.offset,
        infer_chunk=infer_chunk,
        gt_annotations=gt_annotations,
        input_type=input_type,
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={ckpt.get('epoch')}")
    print(f"split={args.split}")
    print(f"thresholds onset={args.onset:.2f} frame={args.frame:.2f} offset={args.offset:.2f}")
    print(f"test_loss={loss:.6f}")
    print(f"COn_f1={con:.6f}")
    print(f"COnP_f1={conp:.6f}")
    print(f"COnPOff_f1={conpoff:.6f}")
    print(f"sig_onset={onset_sig:.6f}")
    print(f"sig_frame={frame_sig:.6f}")


if __name__ == "__main__":
    main()
