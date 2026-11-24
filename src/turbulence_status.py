"""Read progress without importing the numerical runtime."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'results'/'turbulence_2d_2026_11_23'
done=[];running=[];failed=[]
for folder in sorted((OUT/'runs').glob('*')):
    if (folder/'training_summary.json').exists():done.append(folder.name)
    elif (folder/'failure.json').exists():failed.append(folder.name)
    elif (folder/'checkpoint.json').exists():
        p=json.loads((folder/'checkpoint.json').read_text())
        running.append({'name':folder.name,'updates':p['completed_updates'],'budget':p['config']['budget']})
result={'complete':len(done),'complete_names':done,'failed':failed,'planned':12,'running':running,
    'total_updates':len(done)*12000+sum(r['updates'] for r in running),
    'planned_updates':144000,'done':len(done)+len(failed)==12}
if (OUT/'pool_status.json').exists():
    result['process_failures']=[x for x in json.loads((OUT/'pool_status.json').read_text()) if x['returncode']]
print(json.dumps(result))
