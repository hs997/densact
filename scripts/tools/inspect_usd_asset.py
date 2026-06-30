import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd_path", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdGeom


def _print_refs(spec, indent=0):
    refs = [ref.assetPath for ref in spec.referenceList.prependedItems]
    payloads = [payload.assetPath for payload in spec.payloadList.prependedItems]
    if refs or payloads:
        print(" " * indent + f"{spec.path}: refs={refs} payloads={payloads}")
    for child in spec.nameChildren:
        _print_refs(child, indent + 2)


def main():
    usd_path = args_cli.usd_path.resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    print(f"path: {usd_path}", flush=True)
    print(f"defaultPrim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}", flush=True)
    print(f"upAxis: {UsdGeom.GetStageUpAxis(stage)}", flush=True)
    print(f"metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}", flush=True)
    print(f"rootPrims: {[prim.GetPath().pathString for prim in stage.GetPseudoRoot().GetChildren()]}", flush=True)

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    for prim in stage.GetPseudoRoot().GetChildren():
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        print(
            "bound:",
            prim.GetPath(),
            "min=",
            [round(v, 4) for v in box.GetMin()],
            "max=",
            [round(v, 4) for v in box.GetMax()],
            "size=",
            [round(v, 4) for v in box.GetSize()],
            flush=True,
        )

    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is not None:
        print(f"subLayers: {list(layer.subLayerPaths)}", flush=True)
        print("references/payloads:", flush=True)
        for spec in layer.rootPrims:
            _print_refs(spec)


if __name__ == "__main__":
    main()
    simulation_app.close()
