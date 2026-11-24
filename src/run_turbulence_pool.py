"""Run the fixed 12-model grid; workers release JAX memory between models."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'turbulence_2d_2026_11_23'


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['passed'],'Numerical validation must pass first'
    tasks=list(itertools.product([0,1,2],[1101,1102],[1,4]))
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def worker(task):
        seed,dataset,n=task;name=f'data{dataset}_N{n}_seed{seed}';started=time.time()
        with (logs/f'{name}.log').open('a') as f:
            child=subprocess.Popen([sys.executable,'-u','src/train_turbulence.py',
                '--dataset',str(dataset),'--seed',str(seed),'--unroll',str(n)],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            print(json.dumps({'event':'launch','name':name,'pid':child.pid}),flush=True)
            rc=child.wait()
        return {'name':name,'returncode':rc,'seconds':time.time()-started}
    statuses=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for f in as_completed(futures):
            status=f.result();statuses.append(status);print(json.dumps(status),flush=True)
            tmp=OUT/'pool_status.json.tmp';tmp.write_text(json.dumps(statuses,indent=2));tmp.replace(OUT/'pool_status.json')
    raise SystemExit(any(x['returncode'] for x in statuses))


if __name__=='__main__':main()
