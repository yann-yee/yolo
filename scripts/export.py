from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from scripts._common import (
    DEFAULT_DATA_YAML,
    DEFAULT_EXPORT_NAME,
    DEFAULT_RUNS_DIR,
    DEFAULT_TRAIN_WEIGHTS,
    dataset_hint,
    require_existing_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export YOLO11-MCU to quantized ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_TRAIN_WEIGHTS),
        help="Trained checkpoint or model YAML path.",
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Dataset YAML path used for INT8 calibration.")
    parser.add_argument("--imgsz", type=int, default=192, help="Export image size.")
    parser.add_argument("--batch", type=int, default=1, help="Calibration batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device to use during export.")
    parser.add_argument("--project", type=str, default=str(DEFAULT_RUNS_DIR), help="Project directory for export runs.")
    parser.add_argument("--name", type=str, default=DEFAULT_EXPORT_NAME, help="Run name.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--fraction", type=float, default=0.25, help="Calibration fraction for INT8 export.")
    parser.add_argument("--dynamic", action=BooleanOptionalAction, default=False, help="Export dynamic ONNX shapes.")
    parser.add_argument("--simplify", action=BooleanOptionalAction, default=False, help="Run ONNX simplification.")
    parser.add_argument("--nms", action=BooleanOptionalAction, default=False, help="Append NMS to the exported graph when supported.")
    parser.add_argument("--half", action=BooleanOptionalAction, default=False, help="Use FP16 weights where supported.")
    parser.add_argument("--exist-ok", action=BooleanOptionalAction, default=True, help="Allow existing run directory.")
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
        "format": "onnx",
        "int8": True,
        "data": str(data_path),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "opset": args.opset,
        "fraction": args.fraction,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "nms": args.nms,
        "half": args.half,
        "exist_ok": args.exist_ok,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    model = YOLO(str(model_path))
    exported = model.export(**overrides)
    print(exported)


if __name__ == "__main__":
    main()
