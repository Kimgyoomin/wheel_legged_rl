#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim / Kit / omni extensions must be loaded before importing Isaac Lab env/task modules.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG_SRC = PROJECT_ROOT / "source" / "pongbot_w"

if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

print("[INFO] PROJECT_ROOT:", PROJECT_ROOT)
print("[INFO] PKG_SRC     :", PKG_SRC)

import gymnasium as gym

# Force external project task registration.
import pongbot_w.tasks.manager_based.pongbot_w  # noqa: F401,E402

print("\n=== Registered PongbotW tasks ===")
found = False

for env_id, spec in sorted(gym.envs.registry.items()):
    if "PongbotW" in env_id:
        found = True
        print(f"\nID: {env_id}")
        print(f"  entry_point: {spec.entry_point}")
        print(f"  kwargs     : {spec.kwargs}")

if not found:
    raise RuntimeError("No PongbotW tasks were found in Gymnasium registry.")

simulation_app.close()
