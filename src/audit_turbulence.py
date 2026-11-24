"""Check the full fixed grid and independently recompute reported scores."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from turbulence_core import OUT,ROOT,atomic_json,resize


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    expected=set(itertools.product([1101,1102],[0,1,2],[1,4]));seen=set();versions=set()
    max_score_difference=0.;failures=[];nonfinite=[]
    reference=np.load(OUT/'data'/'reference.npz')
    full_error_difference=0.;spectrum_difference=0.
    for folder in sorted((OUT/'runs').glob('*')):
        training=folder/'training_summary.json';failure=folder/'failure.json'
        assert training.exists() or failure.exists(),f'Incomplete training: {folder.name}'
        s=json.loads((training if training.exists() else failure).read_text());c=s['config']
        ident=(c['training_dataset'],c['initialization_seed'],c['unroll'])
        assert ident not in seen;seen.add(ident)
        versions.add(json.dumps(s['provenance'],sort_keys=True))
        assert not c['pilot'] and c['budget']==12000 and c['batch']==8 and c['training_flows']==128
        assert [c['flow'][k] for k in ['hidden_channels','fno_modes','fno_blocks']]==[16,10,4]
        for name,digest in s['provenance']['source_sha256'].items():
            assert sha(OUT/'source_snapshot'/name)==digest,(folder.name,name)
            assert sha(ROOT/'src'/name)==digest,(folder.name,name,'live source changed')
        assert sha(OUT/'data'/f'train_{c["training_dataset"]}.npz')==c['data_sha256']
        assert sha(OUT/'data'/'validation.npz')==c['validation_sha256']
        if not training.exists():failures.append(folder.name);continue
        assert s['completed_updates']==12000 and s['n_params']==410753
        assert len(np.load(folder/'loss.npy'))==12000
        assert sha(folder/'model.eqx')==s['model_sha256']
        checkpoint=json.loads((folder/'checkpoint.json').read_text())
        assert checkpoint['completed_updates']==12000
        assert (folder/checkpoint['checkpoint_file']).exists()
        ev=json.loads((folder/'evaluation.json').read_text())
        assert ev['evaluation_provenance']['evaluator_source_sha256']==sha(ROOT/'src'/'evaluate_turbulence.py')
        assert ev['evaluation_provenance']['reference_sha256']==sha(OUT/'data'/'reference.npz')
        assert ev['evaluation_provenance']['test_data_sha256']==sha(OUT/'data'/'test.npz')
        if ev['status']!='completed':nonfinite.append(folder.name);continue
        d=np.load(folder/'diagnostics.npz')
        score=float(d['emulator_nRMSE'][:,:20].mean(axis=0).sum()/d['teacher_nRMSE'][:,:20].mean(axis=0).sum())
        diff=abs(score-ev['R20']);max_score_difference=max(max_score_difference,diff)
        assert diff<1e-6
        assert d['emulator_nRMSE'].shape==(64,100)
        spectrum_difference=max(spectrum_difference,abs(float(d['mean_energy_spectrum_1_20'].sum())-float(d['kinetic_energy'][:,:20].mean())))
        assert spectrum_difference<1e-5
        snapshots=d['snapshots'];fine=reference['fine_snapshots'][:,1:]
        lifted=np.asarray(resize(jnp.asarray(snapshots),96))
        direct=np.mean((lifted-fine)**2,axis=(-3,-2,-1))
        low=reference['resolved'][:3,[0,4,19,99]]
        reconstructed=np.mean((snapshots-low)**2,axis=(-3,-2,-1))+reference['unresolved_vorticity_variance'][:3,[0,4,19,99]]
        relative=float(np.max(abs(direct-reconstructed))/max(float(direct.max()),1e-8))
        full_error_difference=max(full_error_difference,relative)
        assert relative<1e-5,'Full-field error reconstruction failed'
    assert seen==expected,{'missing':sorted(expected-seen),'extra':sorted(seen-expected)}
    assert len(versions)==1,'Mixed main training provenance'
    validation=json.loads((OUT/'validation.json').read_text());assert validation['passed']
    ref=json.loads((OUT/'data'/'reference_validation.json').read_text())
    assert ref['test_flows']==64 and ref['refinement_fraction_of_teacher_gap']<.05
    result={'passed':True,'planned_models':12,'observed_outcomes':len(seen),
        'full_budget_completed':12-len(failures),'training_failures_retained':failures,
        'nonfinite_forecasts_retained':nonfinite,'training_datasets':2,'initializations_each':3,
        'test_flows':64,'primary_steps':20,'extended_steps':100,'parameters_per_model':410753,
        'training_provenance_versions':len(versions),'maximum_recomputed_primary_score_difference':max_score_difference,
        'maximum_full_field_error_reconstruction_relative_difference':full_error_difference,
        'maximum_spectrum_energy_Parseval_difference':spectrum_difference,
        'all_source_data_model_hashes_match':True,'numerical_validation_passed':True,
        'reference_refinement_fraction_of_teacher_gap':ref['refinement_fraction_of_teacher_gap']}
    atomic_json(OUT/'audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
