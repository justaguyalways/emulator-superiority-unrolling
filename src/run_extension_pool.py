"""Run independent cells in separate processes; retain failure/exit status.

The parent does not initialize JAX. Separate processes release compilation
memory between cells. No two workers are ever assigned the same configuration.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extension_2026_11_22"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--groups",nargs='+',default=['burgers','conv','fixedH'])
    args=p.parse_args()
    sizes={'burgers':12,'conv':36,'fixedH':18}
    tasks=[]
    for round_ in itertools.zip_longest(*(list(range(sizes[g])) for g in args.groups)):
        tasks.extend((g,i) for g,i in zip(args.groups,round_) if i is not None)
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    def launch(task):
        group,i=task
        log=OUT/'logs'/f'{group}_{i:02d}.log'
        started=time.time()
        with log.open('a') as f:
            child=subprocess.Popen([sys.executable,'-u','src/extension_study.py','run',
                '--group',group,'--no-reuse','--chunk','1000','--shard',str(i),str(sizes[group])],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            print(json.dumps({'event':'launch','group':group,'index':i,'pid':child.pid}),flush=True)
            rc=child.wait()
        return {'group':group,'index':i,'returncode':rc,'elapsed_s':time.time()-started,'log':str(log)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(launch,t) for t in tasks]
        statuses=[]
        for f in as_completed(futures):
            status=f.result();statuses.append(status)
            print(json.dumps({'event':'finished',**status}),flush=True)
            tmp=OUT/'pool_status.json.tmp'
            tmp.write_text(json.dumps(statuses,indent=2));tmp.replace(OUT/'pool_status.json')
    failures=[x for x in statuses if x['returncode']]
    print(json.dumps({'event':'pool_finished','cells':len(statuses),'failed':len(failures)}),flush=True)
    raise SystemExit(bool(failures))


if __name__=='__main__':main()
