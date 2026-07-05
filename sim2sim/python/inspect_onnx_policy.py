from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _shape_of(value_info) -> list[int | str]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to policy.onnx")
    args = parser.parse_args()

    onnx_path = Path(args.onnx).expanduser()
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX file not found: {onnx_path}")

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("onnx is required. Install with: python -m pip install onnx") from exc

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    print(f"[INFO] ONNX: {onnx_path}")
    print("[INFO] Inputs:")
    for value in model.graph.input:
        print(f"  - name={value.name} shape={_shape_of(value)}")
    print("[INFO] Outputs:")
    for value in model.graph.output:
        print(f"  - name={value.name} shape={_shape_of(value)}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed. Skipping inference check.")
        return 0

    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required for ONNX Runtime inference. Install with: python -m pip install numpy") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise SystemExit(f"Expected exactly one ONNX input, got {len(inputs)}")
    if len(outputs) != 1:
        raise SystemExit(f"Expected exactly one ONNX output, got {len(outputs)}")

    obs = np.zeros((1, 62), dtype=np.float32)
    action = session.run([outputs[0].name], {inputs[0].name: obs})[0]
    print(f"[INFO] Inference output shape: {list(action.shape)}")
    if list(action.shape) != [1, 16]:
        raise SystemExit(f"Expected action shape [1, 16], got {list(action.shape)}")
    print("[INFO] ONNX inference check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
