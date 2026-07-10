from __future__ import annotations

import importlib
import importlib.metadata
import sys


def _check(package: str, import_name: str | None = None) -> bool:
    module_name = import_name or package
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        print(f"[FAIL] {package}: {exc}")
        return False

    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    print(f"[ OK ] {package}: {version}")
    return True


def main() -> int:
    print("python:", sys.executable)
    print("version:", sys.version)
    print()

    checks = [
        _check("numpy"),
        _check("mujoco"),
        _check("onnx"),
        _check("onnxruntime"),
    ]
    if not all(checks):
        print()
        print("Install missing dependencies with:")
        print("  python -m pip install mujoco onnx onnxruntime numpy")
        return 1
    print()
    print("All sim2sim Python dependencies are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
