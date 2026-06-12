from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from scripts._common import DEFAULT_DATA_YAML, DEFAULT_MODEL_CFG, DEFAULT_RUNS_DIR, DEFAULT_TRAIN_NAME, dataset_hint, require_existing_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the YOLO11-MCU model.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_CFG), help="Model YAML or checkpoint path.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=192, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device to train on, e.g. cpu, 0, 0,1.")
    parser.add_argument("--project", type=str, default=str(DEFAULT_RUNS_DIR), help="Project directory for training runs.")
    parser.add_argument("--name", type=str, default=DEFAULT_TRAIN_NAME, help="Run name.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer name.")
    parser.add_argument("--box", type=float, default=None, help="Box loss gain.")
    parser.add_argument("--cls", type=float, default=None, help="Classification loss gain.")
    parser.add_argument("--cls-pw", dest="cls_pw", type=float, default=None, help="Class weight power.")
    parser.add_argument("--dfl", type=float, default=None, help="Distribution focal loss gain.")
    parser.add_argument(
        "--box-loss",
        type=str,
        default=None,
        choices=("ciou", "siou", "wiou"),
        help="Box regression loss type.",
    )
    parser.add_argument(
        "--wiou-version",
        type=str,
        default=None,
        choices=("origin", "v2", "v3"),
        help="Wise-IoU dynamic focusing mode.",
    )
    parser.add_argument("--wiou-momentum", type=float, default=None, help="Wise-IoU running mean momentum.")
    parser.add_argument("--wiou-alpha", type=float, default=None, help="Wise-IoU dynamic focusing alpha.")
    parser.add_argument("--wiou-delta", type=float, default=None, help="Wise-IoU dynamic focusing delta.")
    parser.add_argument("--siou-theta", type=float, default=None, help="SIoU shape exponent.")
    parser.add_argument("--cache", action=BooleanOptionalAction, default=False, help="Cache images in memory.")
    parser.add_argument("--resume", action=BooleanOptionalAction, default=False, help="Resume from last checkpoint.")
    parser.add_argument("--amp", action=BooleanOptionalAction, default=True, help="Use automatic mixed precision.")
    parser.add_argument("--pretrained", action=BooleanOptionalAction, default=True, help="Use pretrained initialization if available.")
    parser.add_argument("--cos-lr", dest="cos_lr", action=BooleanOptionalAction, default=True, help="Use cosine learning rate schedule.")
    parser.add_argument("--plots", action=BooleanOptionalAction, default=True, help="Save training plots.")
    parser.add_argument("--exist-ok", action=BooleanOptionalAction, default=True, help="Allow existing run directory.")
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=10, help="Disable mosaic augmentation in the last N epochs.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate fraction.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = require_existing_path(
        args.model,
        "Model YAML/checkpoint",
        hint=f"Expected {DEFAULT_MODEL_CFG.relative_to(ROOT)} or a trained checkpoint.",
    )
    data_path = require_existing_path(args.data, "Dataset YAML", hint=dataset_hint())

    overrides = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "box": args.box,
        "cls": args.cls,
        "cls_pw": args.cls_pw,
        "dfl": args.dfl,
        "box_loss": args.box_loss,
        "wiou_version": args.wiou_version,
        "wiou_momentum": args.wiou_momentum,
        "wiou_alpha": args.wiou_alpha,
        "wiou_delta": args.wiou_delta,
        "siou_theta": args.siou_theta,
        "cache": args.cache,
        "resume": args.resume,
        "amp": args.amp,
        "pretrained": args.pretrained,
        "cos_lr": args.cos_lr,
        "plots": args.plots,
        "exist_ok": args.exist_ok,
        "close_mosaic": args.close_mosaic,
        "lr0": args.lr0,
        "lrf": args.lrf,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    model = YOLO(str(model_path))
    results = model.train(**overrides)
    print(results)


if __name__ == "__main__":
    main()
