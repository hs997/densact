#!/usr/bin/env python3
"""Terminal S/space software stop, independent of policy; never sends chassis motion."""
import argparse
import select
import signal
import sys
import termios
import time
import tty

from core import load_config
from ros_io import ROSBus


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--ros', type=int, choices=[1, 2], default=1)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error('Use a local interactive terminal for keyboard stop')
    bus = ROSBus(args.ros, 'ams_car05_keyboard_stop')
    pub = bus.publisher(load_config(args.config)['topics']['estop'], bus.Bool)
    def send_stop():
        msg = bus.Bool(); msg.data = True
        pub.publish(msg)
    quitting = [False]
    def interrupt(*_):
        quitting[0] = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    old = termios.tcgetattr(sys.stdin)
    print('S / SPACE = latched software stop; Q = stop and quit. No arming from this terminal.', flush=True)
    print('This ROS stop requires a responsive guard/network; keep the physical stop accessible.', flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while bus.ok() and not quitting[0]:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1).lower()
                if key in ('s', ' ', 'q'):
                    send_stop()
                    print('STOP requested; check guard status and actual vehicle stop.', flush=True)
                if key == 'q': break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        try:
            for _ in range(3):
                send_stop(); time.sleep(0.05)
        finally:
            bus.close()


if __name__ == '__main__':
    main()
