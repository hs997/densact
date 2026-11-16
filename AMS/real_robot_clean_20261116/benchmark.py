#!/usr/bin/env python3
"""Read-only inference timing; no ROS imports or publishing. Run on the actual TX2."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from core import NumpyActor,leader_observation,leader_action


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2000); args=p.parse_args()
    if args.samples<100:p.error('Use at least 100 samples')
    actor=NumpyActor(args.model); profile=actor.meta['profile']; times=[]
    state=dict(x=-8.75,y=-.065,yaw=0.,vx_body=.1,vy_body=0.,yaw_rate=0.)
    for i in range(args.samples+100):
        start=time.perf_counter()
        raw=actor(leader_observation(state,profile)); leader_action(raw,state,profile)
        if i>=100: times.append(1000*(time.perf_counter()-start))
    print(json.dumps(dict(machine=platform.machine(),python=platform.python_version(),samples=len(times),
        median_ms=float(np.median(times)),p99_ms=float(np.percentile(times,99)),max_ms=max(times),
        period_budget_ms=20.,includes_ros_transport=False,model_sha256=actor.meta['weights_sha256']),indent=2))


if __name__=='__main__':main()
