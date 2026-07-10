from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim2sim.common.model_resolver import resolve_mesh_filename


def _vec(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(v) for v in text.split())


def _fmt(values) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)


def _rpy_to_rot(rpy: tuple[float, float, float]) -> np.ndarray:
    """Convert URDF fixed-axis roll, pitch, yaw to a rotation matrix."""
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rz @ ry @ rx


def _rotate_inertia_to_body_frame(
    inertia_values: tuple[float, float, float, float, float, float],
    rpy: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Rotate URDF inertia tensor from inertial frame to link/body frame."""
    ixx, iyy, izz, ixy, ixz, iyz = inertia_values
    inertia_inertial = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )
    rot = _rpy_to_rot(rpy)
    inertia_body = rot @ inertia_inertial @ rot.T
    inertia_body = 0.5 * (inertia_body + inertia_body.T)
    return (
        float(inertia_body[0, 0]),
        float(inertia_body[1, 1]),
        float(inertia_body[2, 2]),
        float(inertia_body[0, 1]),
        float(inertia_body[0, 2]),
        float(inertia_body[1, 2]),
    )


def _rpy_to_quat(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qw, qx, qy, qz


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return safe.strip("_") or "mesh"


@dataclass
class Inertial:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    mass: float
    fullinertia: tuple[float, float, float, float, float, float]


@dataclass
class GeomSpec:
    kind: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    geometry: ET.Element
    rgba: str | None = None


@dataclass
class LinkSpec:
    name: str
    inertial: Inertial | None = None
    visuals: list[GeomSpec] = field(default_factory=list)
    collisions: list[GeomSpec] = field(default_factory=list)


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    limit: dict[str, float]


class MeshStager:
    def __init__(self, urdf_path: Path, mesh_dir: Path, output_dir: Path):
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.stage_dir = output_dir / "meshes"
        self.assets: dict[Path, str] = {}
        self.count = 0

    def stage(self, filename: str) -> tuple[str, str]:
        src = resolve_mesh_filename(filename, self.urdf_path, self.mesh_dir)
        if src in self.assets:
            return self.assets[src], f"meshes/{src.name}"

        self.stage_dir.mkdir(parents=True, exist_ok=True)
        dst = self.stage_dir / src.name
        if dst.exists() or dst.is_symlink():
            try:
                if dst.resolve() != src.resolve():
                    dst.unlink()
            except OSError:
                dst.unlink()
        if not dst.exists():
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)

        mesh_name = _safe_name(src.stem)
        base_name = mesh_name
        index = 1
        while mesh_name in self.assets.values():
            index += 1
            mesh_name = f"{base_name}_{index}"
        self.assets[src] = mesh_name
        self.count += 1
        return mesh_name, f"meshes/{src.name}"


def _parse_inertial(elem: ET.Element | None) -> Inertial | None:
    if elem is None:
        return None
    origin = elem.find("origin")
    mass = elem.find("mass")
    inertia = elem.find("inertia")
    if mass is None or inertia is None:
        return None
    xyz = _vec(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _vec(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    return Inertial(
        xyz=xyz,  # type: ignore[arg-type]
        rpy=rpy,  # type: ignore[arg-type]
        mass=float(mass.get("value", "0.1")),
        fullinertia=(
            float(inertia.get("ixx", "0.001")),
            float(inertia.get("iyy", "0.001")),
            float(inertia.get("izz", "0.001")),
            float(inertia.get("ixy", "0")),
            float(inertia.get("ixz", "0")),
            float(inertia.get("iyz", "0")),
        ),
    )


def _parse_geom(elem: ET.Element, kind: str) -> GeomSpec | None:
    origin = elem.find("origin")
    geometry = elem.find("geometry")
    if geometry is None or len(list(geometry)) == 0:
        return None
    xyz = _vec(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _vec(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    rgba = None
    material = elem.find("material")
    color = material.find("color") if material is not None else None
    if color is not None and color.get("rgba"):
        rgba = color.get("rgba")
    return GeomSpec(kind=kind, xyz=xyz, rpy=rpy, geometry=list(geometry)[0], rgba=rgba)  # type: ignore[arg-type]


def _parse_urdf(urdf_path: Path) -> tuple[dict[str, LinkSpec], list[JointSpec]]:
    root = ET.parse(urdf_path).getroot()
    links: dict[str, LinkSpec] = {}
    joints: list[JointSpec] = []

    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        if not name:
            continue
        link = LinkSpec(name=name, inertial=_parse_inertial(link_elem.find("inertial")))
        for visual in link_elem.findall("visual"):
            parsed = _parse_geom(visual, "visual")
            if parsed is not None:
                link.visuals.append(parsed)
        for collision in link_elem.findall("collision"):
            parsed = _parse_geom(collision, "collision")
            if parsed is not None:
                link.collisions.append(parsed)
        links[name] = link

    for joint_elem in root.findall("joint"):
        parent = joint_elem.find("parent")
        child = joint_elem.find("child")
        if parent is None or child is None:
            continue
        origin = joint_elem.find("origin")
        axis = joint_elem.find("axis")
        limit = joint_elem.find("limit")
        joints.append(
            JointSpec(
                name=joint_elem.get("name", ""),
                joint_type=joint_elem.get("type", "fixed"),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                xyz=_vec(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
                rpy=_vec(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
                axis=_vec(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),  # type: ignore[arg-type]
                limit={key: float(value) for key, value in (limit.attrib.items() if limit is not None else [])},
            )
        )
    return links, joints


def _add_inertial(body: ET.Element, link: LinkSpec) -> None:
    if link.inertial is None:
        print(f"[WARN] Link '{link.name}' has no inertial. Using fallback mass/inertia.")
        ET.SubElement(body, "inertial", mass="0.1", diaginertia="0.001 0.001 0.001")
        return
    fullinertia_body = _rotate_inertia_to_body_frame(link.inertial.fullinertia, link.inertial.rpy)
    ET.SubElement(
        body,
        "inertial",
        pos=_fmt(link.inertial.xyz),
        mass=f"{link.inertial.mass:.10g}",
        fullinertia=_fmt(fullinertia_body),
    )


def _add_geom(parent: ET.Element, asset: ET.Element, geom: GeomSpec, name: str, stager: MeshStager) -> None:
    geometry = geom.geometry
    attrs = {
        "name": name,
        "pos": _fmt(geom.xyz),
        "quat": _fmt(_rpy_to_quat(geom.rpy)),
    }
    if geom.kind == "collision":
        attrs.update({"contype": "1", "conaffinity": "1", "friction": "1.0 0.005 0.0001"})
    else:
        attrs.update({"contype": "0", "conaffinity": "0", "group": "1"})
        if geom.rgba:
            attrs["rgba"] = geom.rgba

    if geometry.tag == "box":
        size = _vec(geometry.get("size"), (0.1, 0.1, 0.1))
        attrs.update({"type": "box", "size": _fmt([0.5 * value for value in size])})
    elif geometry.tag == "cylinder":
        radius = float(geometry.get("radius", "0.05"))
        length = float(geometry.get("length", "0.1"))
        attrs.update({"type": "cylinder", "size": _fmt((radius, 0.5 * length))})
    elif geometry.tag == "sphere":
        attrs.update({"type": "sphere", "size": geometry.get("radius", "0.05")})
    elif geometry.tag == "mesh":
        filename = geometry.get("filename") or geometry.get("file")
        if not filename:
            return
        mesh_name, mesh_file = stager.stage(filename)
        if asset.find(f"mesh[@name='{mesh_name}']") is None:
            ET.SubElement(asset, "mesh", name=mesh_name, file=mesh_file)
        attrs.update({"type": "mesh", "mesh": mesh_name})
    else:
        print(f"[WARN] Unsupported geometry tag '{geometry.tag}' in {name}; skipping.")
        return
    ET.SubElement(parent, "geom", **attrs)


def _add_link_contents(body: ET.Element, asset: ET.Element, link: LinkSpec, stager: MeshStager) -> None:
    _add_inertial(body, link)
    for index, geom in enumerate(link.collisions):
        _add_geom(body, asset, geom, f"{link.name}_collision_{index}", stager)
    for index, geom in enumerate(link.visuals):
        _add_geom(body, asset, geom, f"{link.name}_visual_{index}", stager)


def _add_joint(body: ET.Element, joint: JointSpec) -> None:
    if joint.joint_type == "fixed":
        return
    if joint.joint_type in {"revolute", "continuous"}:
        attrs = {"name": joint.name, "type": "hinge", "axis": _fmt(joint.axis)}
        if joint.joint_type == "revolute" and "lower" in joint.limit and "upper" in joint.limit:
            attrs["range"] = _fmt((joint.limit["lower"], joint.limit["upper"]))
        ET.SubElement(body, "joint", **attrs)
    elif joint.joint_type == "prismatic":
        attrs = {"name": joint.name, "type": "slide", "axis": _fmt(joint.axis)}
        if "lower" in joint.limit and "upper" in joint.limit:
            attrs["range"] = _fmt((joint.limit["lower"], joint.limit["upper"]))
        ET.SubElement(body, "joint", **attrs)
    else:
        print(f"[WARN] Unsupported joint type '{joint.joint_type}' for {joint.name}; treating as fixed.")


def _build_body_tree(
    parent_body: ET.Element,
    parent_link_name: str,
    links: dict[str, LinkSpec],
    children: dict[str, list[JointSpec]],
    asset: ET.Element,
    stager: MeshStager,
) -> None:
    for joint in children.get(parent_link_name, []):
        child_body = ET.SubElement(
            parent_body,
            "body",
            name=joint.child,
            pos=_fmt(joint.xyz),
            quat=_fmt(_rpy_to_quat(joint.rpy)),
        )
        _add_joint(child_body, joint)
        _add_link_contents(child_body, asset, links[joint.child], stager)
        _build_body_tree(child_body, joint.child, links, children, asset, stager)


def build_freebase_mjcf(urdf_path: Path, mesh_dir: Path, output_path: Path) -> None:
    links, joints = _parse_urdf(urdf_path)
    if "BASE" not in links:
        parent_links = {joint.parent for joint in joints}
        child_links = {joint.child for joint in joints}
        roots = sorted(parent_links - child_links)
        print(f"[WARN] BASE link not found. Candidate root links: {roots}")
        raise RuntimeError("PongbotW free-base MJCF generation requires a BASE link.")

    children: dict[str, list[JointSpec]] = defaultdict(list)
    for joint in joints:
        children[joint.parent].append(joint)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stager = MeshStager(urdf_path=urdf_path, mesh_dir=mesh_dir, output_dir=output_path.parent)

    mujoco = ET.Element("mujoco", model="PONGBOT_W")
    ET.SubElement(mujoco, "compiler", angle="radian", coordinate="local")
    ET.SubElement(mujoco, "option", timestep="0.002", gravity="0 0 -9.81")
    asset = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size="5 5 0.05", rgba="0.8 0.8 0.8 1")
    base_body = ET.SubElement(worldbody, "body", name="BASE", pos="0 0 0.63")
    ET.SubElement(base_body, "freejoint", name="floating_base")
    _add_link_contents(base_body, asset, links["BASE"], stager)
    _build_body_tree(base_body, "BASE", links, children, asset, stager)

    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"Generated MJCF: {output_path}")
    print(f"Links converted: {len(links)}")
    print(f"Joints converted: {len(joints)}")
    print(f"Meshes staged: {stager.count}")
    print("Root body: BASE")
    print("Expected freejoint: floating_base")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Input PongbotW URDF")
    parser.add_argument("--mesh-dir", required=True, help="Directory containing STL meshes")
    parser.add_argument("--output", required=True, help="Output free-base MJCF XML path")
    args = parser.parse_args()

    urdf_path = Path(args.urdf).expanduser().resolve()
    mesh_dir = Path(args.mesh_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh dir not found: {mesh_dir}")
    build_freebase_mjcf(urdf_path, mesh_dir, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
