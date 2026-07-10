from __future__ import annotations

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by walking upward from start."""
    cur = Path(start or __file__).resolve()
    if cur.is_file():
        cur = cur.parent

    for p in [cur, *cur.parents]:
        if (p / "sim2sim").exists() and (p / "source").exists():
            return p
        if (p / ".git").exists():
            return p

    return Path.cwd().resolve()


def guess_mesh_dir(model_path: Path, explicit_mesh_dir: Path | None = None) -> Path | None:
    """Guess where robot mesh files are located."""
    candidates: list[Path] = []

    if explicit_mesh_dir is not None:
        candidates.append(Path(explicit_mesh_dir).expanduser())

    model_path = Path(model_path).expanduser().resolve()
    project_root = find_project_root(model_path)

    candidates.extend(
        [
            model_path.parent / "meshes",
            model_path.parent.parent / "meshes",
            project_root / "assets" / "robots" / "pongbot_w" / "meshes",
        ]
    )

    for c in candidates:
        c = c.expanduser().resolve()
        if c.exists() and c.is_dir():
            return c

    return None


def _strip_uri_prefix(filename: str) -> str:
    """Handle common URI prefixes in URDF/MJCF mesh paths."""
    if filename.startswith("file://"):
        return filename[len("file://") :]

    if filename.startswith("package://"):
        # We do not know the ROS package root inside this conda sim2sim context.
        # Keep the suffix and resolve by basename / mesh_dir fallback.
        return filename[len("package://") :]

    return filename


def resolve_mesh_filename(filename: str, model_path: Path, mesh_dir: Path | None) -> Path:
    """Resolve a mesh filename from URDF/MJCF into an existing file path."""
    raw = filename
    filename = _strip_uri_prefix(filename)
    path = Path(filename)

    model_path = Path(model_path).expanduser().resolve()

    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)

    # Relative path as written.
    candidates.append(model_path.parent / path)

    # Bare basename relative to model file.
    candidates.append(model_path.parent / path.name)

    # Common layout candidates.
    candidates.append(model_path.parent / "meshes" / path.name)
    candidates.append(model_path.parent.parent / "meshes" / path.name)

    if mesh_dir is not None:
        candidates.append(Path(mesh_dir).expanduser().resolve() / path.name)

    for c in candidates:
        c = c.expanduser().resolve()
        if c.exists():
            return c

    msg = [f"Could not resolve mesh reference: {raw}"]
    msg.append("Tried:")
    for c in candidates:
        msg.append(f"  - {c}")
    raise FileNotFoundError("\n".join(msg))


def _stage_mesh(mesh_src: Path, stage_dir: Path) -> Path:
    """Place a symlink/copy of mesh_src next to the resolved XML.

    MuJoCo's URDF loader may resolve mesh basenames relative to the XML path.
    The robust solution is to stage all referenced meshes next to the resolved model.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    dst = stage_dir / mesh_src.name

    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == mesh_src.resolve():
                return dst
        except Exception:
            pass
        dst.unlink()

    try:
        os.symlink(mesh_src.resolve(), dst)
    except OSError:
        shutil.copy2(mesh_src, dst)

    return dst


def make_mujoco_resolved_xml(
    model_path: str | Path,
    mesh_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    keep: bool = True,
) -> Path:
    """Create a MuJoCo-loadable XML/URDF with staged mesh files.

    Important:
    - We stage meshes next to the resolved XML.
    - We rewrite mesh references to basename only.
    - This avoids MuJoCo trying to open stale relative paths such as:
        sim2sim/.cache/resolved_models/RL_CALF.STL
      without the mesh actually existing there.
    """
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model XML/URDF not found: {model_path}")

    mesh_dir_path = guess_mesh_dir(model_path, Path(mesh_dir).expanduser() if mesh_dir else None)

    project_root = find_project_root(model_path)
    if output_dir is None:
        output_dir = project_root / "sim2sim" / ".cache" / "resolved_models"
    else:
        output_dir = Path(output_dir).expanduser()

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = model_path.suffix
    out_path = output_dir / f"{model_path.stem}_mujoco_resolved{suffix}"

    tree = ET.parse(model_path)
    root = tree.getroot()

    rewrites: list[tuple[str, str]] = []

    # URDF: <mesh filename="...">
    # MJCF: <mesh file="...">
    mesh_attrs = ["filename", "file"]

    for elem in root.iter():
        for attr in mesh_attrs:
            if attr not in elem.attrib:
                continue

            old_ref = elem.attrib[attr]
            mesh_src = resolve_mesh_filename(old_ref, model_path, mesh_dir_path)
            staged = _stage_mesh(mesh_src, output_dir)

            # Use basename because staged mesh is next to resolved XML.
            elem.attrib[attr] = staged.name
            rewrites.append((old_ref, str(staged)))

    # Write XML.
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"[INFO] Mesh dir: {mesh_dir_path}")
    print(f"[INFO] Resolved XML output dir: {output_dir}")
    print(f"[INFO] Rewrote {len(rewrites)} mesh reference(s).")
    for old, new in rewrites:
        print(f"  {old} -> {new}")

    return out_path
