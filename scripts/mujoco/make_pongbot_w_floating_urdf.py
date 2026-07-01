#!/usr/bin/env python3

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def collect_link_names(robot: ET.Element) -> set[str]:
    names = set()
    for link in robot.findall("link"):
        name = link.attrib.get("name")
        if name:
            names.add(name)
    return names


def collect_child_link_names(robot: ET.Element) -> set[str]:
    children = set()
    for joint in robot.findall("joint"):
        child = joint.find("child")
        if child is not None and "link" in child.attrib:
            children.add(child.attrib["link"])
    return children


def find_root_link(robot: ET.Element) -> str:
    links = collect_link_names(robot)
    children = collect_child_link_names(robot)

    root_links = sorted(links - children)

    # Ignore an already-inserted artificial world link if present.
    root_links_no_world = [name for name in root_links if name != "world"]

    if len(root_links_no_world) == 1:
        return root_links_no_world[0]

    if len(root_links_no_world) == 0 and "BASE" in links:
        return "BASE"

    raise RuntimeError(
        "Could not infer a unique robot root link.\n"
        f"links - children = {root_links}\n"
        "Pass --base-link explicitly."
    )


def remove_existing_floating_joint(robot: ET.Element, joint_name: str) -> None:
    for joint in list(robot.findall("joint")):
        if joint.attrib.get("name") == joint_name:
            robot.remove(joint)


def ensure_world_link(robot: ET.Element, world_link_name: str) -> None:
    links = collect_link_names(robot)
    if world_link_name in links:
        return

    world_link = ET.Element("link", {"name": world_link_name})
    robot.insert(0, world_link)


def insert_floating_joint(
    robot: ET.Element,
    world_link_name: str,
    base_link_name: str,
    joint_name: str,
) -> None:
    floating_joint = ET.Element(
        "joint",
        {
            "name": joint_name,
            "type": "floating",
        },
    )
    ET.SubElement(floating_joint, "parent", {"link": world_link_name})
    ET.SubElement(floating_joint, "child", {"link": base_link_name})

    # Keep zero origin here.
    # We will set base z=0.63 through qpos reset in MuJoCo.
    ET.SubElement(
        floating_joint,
        "origin",
        {
            "xyz": "0 0 0",
            "rpy": "0 0 0",
        },
    )

    # Put the floating joint near the top for readability.
    robot.insert(1, floating_joint)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    # xml.etree.ElementTree.indent exists in Python 3.9+.
    ET.indent(elem, space="  ", level=level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input fixed-base URDF")
    parser.add_argument("--output", required=True, help="Output floating-base URDF")
    parser.add_argument("--base-link", default=None, help="Robot base/root link name. Auto-detected if omitted.")
    parser.add_argument("--world-link", default="world")
    parser.add_argument("--joint-name", default="floating_base_joint")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    tree = ET.parse(input_path)
    robot = tree.getroot()

    if robot.tag != "robot":
        raise RuntimeError(f"Expected URDF root tag <robot>, got <{robot.tag}>")

    base_link_name = args.base_link or find_root_link(robot)

    if base_link_name not in collect_link_names(robot):
        raise RuntimeError(f"Base link '{base_link_name}' does not exist in URDF.")

    remove_existing_floating_joint(robot, args.joint_name)
    ensure_world_link(robot, args.world_link)
    insert_floating_joint(
        robot=robot,
        world_link_name=args.world_link,
        base_link_name=base_link_name,
        joint_name=args.joint_name,
    )

    indent_xml(robot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print("[OK] Floating-base URDF written")
    print(f"input     : {input_path}")
    print(f"output    : {output_path}")
    print(f"base link : {base_link_name}")
    print(f"world link: {args.world_link}")
    print(f"joint     : {args.joint_name}")


if __name__ == "__main__":
    main()
