"""Read-only progress of the prescribed extension; no ML imports required."""
import argparse
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'results'/'extension_2026_11_22'
LABELS={'extension_conv':('ConvNet',36),'extension_fixedH5':('FNO shared data',18),
        'extension_burgers':('UNet Burgers',12)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    complete={k:0 for k in LABELS};failed={k:0 for k in LABELS};running=[]
    for folder in sorted((OUT/'runs').glob('*')):
        summary=folder/'summary.json';failure=folder/'failure.json';checkpoint=folder/'checkpoint.json'
        path=summary if summary.exists() else failure if failure.exists() else checkpoint
        if not path.exists():continue
        d=json.loads(path.read_text());c=d['config'];tag=c['tag']
        if tag not in LABELS:continue
        if summary.exists():complete[tag]+=1
        elif failure.exists():complete[tag]+=1;failed[tag]+=1
        else:
            running.append({'group':LABELS[tag][0],'N':c['num_unrolled_steps'],
                'mode':c['train_mode'],'seed':c['seed_init'],
                'updates':d['completed_updates'],'budget':c['num_training_steps'],
                'stage':'evaluating' if d['completed_updates']==c['num_training_steps'] else 'training'})
    total=sum(complete.values());failed_count=sum(failed.values())
    parts=[f'{LABELS[k][0]} {n}/{LABELS[k][1]}' for k,n in complete.items()]
    result={'completed_main_cells':total,'planned_main_cells':66,'failed_training_cells':failed_count,
        'by_group':{LABELS[k][0]:n for k,n in complete.items()},'started_unfinished':running,
        'done':total==66,'message':f'Finished {total}/66 main experiments: '+ '; '.join(parts)+'.'}
    pool=OUT/'pool_status.json'
    if pool.exists():
        statuses=json.loads(pool.read_text())
        result['nonzero_process_exits']=[x for x in statuses if x['returncode']]
    print(json.dumps(result) if args.json else result['message'])
    if not args.json:
        for r in running:print(r)


if __name__=='__main__':main()
