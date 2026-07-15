import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.client

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def list_dir(path: str):
    result, entries = omni.client.list(path)
    print(f"\nLIST {path}", flush=True)
    print(f"RESULT {result}", flush=True)
    if entries:
        for entry in entries:
            print(entry.relative_path, flush=True)
    sys.stdout.flush()


def main():
    print(f"ISAAC_NUCLEUS_DIR={ISAAC_NUCLEUS_DIR}", flush=True)
    for subdir in [
        "Environments",
        "Environments/Grid",
        "Environments/Simple_Room",
        "Environments/Simple_Warehouse",
        "Environments/Hospital",
        "Environments/Office",
        "Environments/Outdoor",
        "Environments/Outdoor/Rivermark",
        "Environments/Outdoor/Rivermark/Materials",
        "Environments/Outdoor/Rivermark/Props",
        "Environments/City",
        "Samples",
        "Samples/ROS2",
        "Samples/ROS2/Scenario",
        "Samples/Scene_Blox",
        "Samples/Scene_Blox/warehouse",
        "Samples/Scene_Blox/office",
        "Samples/Scene_Blox/navigation",
    ]:
        list_dir(f"{ISAAC_NUCLEUS_DIR}/{subdir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
