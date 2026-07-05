#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in [
        "model_state_dict",
        "state_dict",
        "actor_critic_state_dict",
    ]:
        if key in ckpt:
            print("[INFO] using state dict key:", key)
            return ckpt[key]

    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    print("checkpoint:", ckpt_path)

    if isinstance(ckpt, dict):
        print("top-level keys:", list(ckpt.keys()))

    sd = extract_state_dict(ckpt)

    print("\n=== 2D tensors likely relevant to actor/critic MLP ===")
    for name, tensor in sd.items():
        if torch.is_tensor(tensor) and tensor.ndim == 2:
            print(f"{name:90s} shape={tuple(tensor.shape)}")

    print("\nInterpretation:")
    print("Actor first Linear weight usually has shape [hidden_dim, obs_dim].")
    print("If you see something like (..., 60), the checkpoint expects 60-dim obs.")
    print("If you see something like (..., 277), the checkpoint expects 277-dim obs.")


if __name__ == "__main__":
    main()
