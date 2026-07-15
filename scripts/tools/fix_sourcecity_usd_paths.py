import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=Path, required=True)
parser.add_argument("--dst_dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Sdf


def export_usda(src: Path, dst: Path):
    layer = Sdf.Layer.FindOrOpen(str(src))
    if layer is None:
        raise RuntimeError(f"Could not open USD layer: {src}")
    text = layer.ExportToString()
    text = text.replace("@/props/", "@./props/")
    text = text.replace("@/textures/", "@./textures/")
    text = text.replace("@/materials/", "@./materials/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def main():
    src_dir = args_cli.src_dir.resolve()
    dst_dir = args_cli.dst_dir.resolve()
    for src in src_dir.rglob("*.usd"):
        rel = src.relative_to(src_dir)
        dst = (dst_dir / rel).with_suffix(".usda")
        export_usda(src, dst)
        print(f"{src} -> {dst}", flush=True)
    for src in src_dir.rglob("*"):
        if src.is_file() and src.suffix.lower() not in {".usd", ".usda"}:
            rel = src.relative_to(src_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
    simulation_app.close()
