#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from pongbot_w_mujoco_contract import PHYSICS_DT


def ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def add_or_update_ground(worldbody: ET.Element) -> None:
    for geom in worldbody.findall("geom"):
        if geom.attrib.get("name") == "ground":
            geom.attrib.update(
                {
                    "type": "plane",
                    "pos": "0 0 0",
                    "size": "100 100 0.1",
                    "contype": "1",
                    "conaffinity": "1",
                    "friction": "1.0 0.005 0.0001",
                }
            )
            return

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "pos": "0 0 0",
            "size": "100 100 0.1",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.0 0.005 0.0001",
            "rgba": "0.5 0.5 0.5 1",
        },
    )


def add_light_if_missing(worldbody: ET.Element) -> None:
    for light in worldbody.findall("light"):
        if light.attrib.get("name") == "main_light":
            return

    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "main_light",
            "pos": "0 -3 4",
            "dir": "0 0 -1",
            "diffuse": "0.8 0.8 0.8",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Floating-base URDF path")
    parser.add_argument("--output", required=True, help="Output MuJoCo MJCF XML path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading floating URDF: {input_path}")
    model = mujoco.MjModel.from_xml_path(str(input_path))

    # Save MuJoCo's compiled representation as editable MJCF.
    tmp_path = output_path.with_suffix(".tmp.xml")
    mujoco.mj_saveLastXML(str(tmp_path), model)

    tree = ET.parse(tmp_path)
    root = tree.getroot()

    if root.tag != "mujoco":
        raise RuntimeError(f"Expected <mujoco> root after conversion, got <{root.tag}>")

    option = ensure_child(root, "option")
    option.set("timestep", f"{PHYSICS_DT:.8f}")
    option.set("gravity", "0 0 -9.81")

    worldbody = ensure_child(root, "worldbody")
    add_or_update_ground(worldbody)
    add_light_if_missing(worldbody)

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    tmp_path.unlink(missing_ok=True)

    print(f"[OK] MuJoCo XML written: {output_path}")

    # Reload test.
    test_model = mujoco.MjModel.from_xml_path(str(output_path))
    print("[OK] Reloaded generated XML")
    print("nq:", test_model.nq)
    print("nv:", test_model.nv)
    print("njnt:", test_model.njnt)
    print("ngeom:", test_model.ngeom)
    print("dt:", test_model.opt.timestep)


if __name__ == "__main__":
    main()
