from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from scripts._common import DEFAULT_DATA_YAML, DEFAULT_RUNS_DIR, DEFAULT_TRAIN_WEIGHTS, DEFAULT_VAL_NAME, dataset_hint, require_existing_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the YOLO11-MCU model.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_TRAIN_WEIGHTS),
        help="Trained checkpoint or model YAML path.",
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=192, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device to validate on.")
    parser.add_argument("--project", type=str, default=str(DEFAULT_RUNS_DIR), help="Project directory for validation runs.")
    parser.add_argument("--name", type=str, default=DEFAULT_VAL_NAME, help="Run name.")
    parser.add_argument("--split", type=str, default="val", choices=("val", "test"), help="Dataset split.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold.")
    parser.add_argument("--max-det", dest="max_det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--save-json", action=BooleanOptionalAction, default=False, help="Save COCO JSON results.")
    parser.add_argument("--plots", action=BooleanOptionalAction, default=True, help="Save validation plots.")
    parser.add_argument("--exist-ok", action=BooleanOptionalAction, default=True, help="Allow existing run directory.")
    parser.add_argument("--half", action=BooleanOptionalAction, default=False, help="Use half precision when supported.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = require_existing_path(
        args.model,
        "Model checkpoint",
        hint=f"Run train.py first to create {DEFAULT_TRAIN_WEIGHTS.relative_to(ROOT)}.",
    )
    data_path = require_existing_path(args.data, "Dataset YAML", hint=dataset_hint())

    overrides = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "save_json": args.save_json,
        "plots": args.plots,
        "exist_ok": args.exist_ok,
        "half": args.half,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    model = YOLO(str(model_path))
    results = model.val(**overrides)
    print(results)


if __name__ == "__main__":
    main()
