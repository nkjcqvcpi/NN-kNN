"""Bind a public-evaluation metric to its recorded train/tune selection.

The metric is either a legacy ``s<seed>/metric.pt`` written by ``tools/t3_need_matching.py``
without ``--study`` (bound through the protocol.json/summary.json files next to it), or a
complete run unit ``runs/t3/<study>/s<seed>/`` written with ``--study`` (bound through the
unit's TensorBoard records and its verified weight file ``weights/metric.pt``).
"""
import hashlib
import json
from pathlib import Path


def is_metric_unit(path):
    """True for a run unit directory (it holds event files),False for a legacy metric.pt."""
    path=Path(path)
    return path.is_dir() and any('tfevents' in p.name for p in path.iterdir() if p.is_file())


def _binding(protocol,run,digest,artifacts):
    if protocol.get('public_dev_annotations_used') is not False:
        raise ValueError('public evaluation requires recorded train-only metric fitting')
    selection=protocol['selection']
    if selection=='lowest held-out-article tune mean need loss,including initial epoch0':
        selection='tune_loss'
    if selection not in ('tune_loss','tune_recall'):
        raise ValueError('unknown metric selection rule')
    if type(run['selected_epoch']) is not int or not 0<=run['selected_epoch']<=protocol['epochs']:
        raise ValueError('metric selected epoch outside recorded budget')
    binding=protocol.get('query_binding')
    term_frequency=protocol.get('term_frequency','count')
    if term_frequency not in ('count','sublinear','binary'):raise ValueError('unknown metric term frequency')
    query_source=protocol.get('query_source','original public question')
    if query_source not in ('original public question','actual frozen host request') or bool(binding)!=(query_source=='actual frozen host request'):
        raise ValueError('metric query source/binding mismatch')
    return dict(checkpoint_sha256=digest,seed=run['seed'],selected_epoch=run['selected_epoch'],
        selection=selection,recorded_selection=protocol['selection'],need_objective=protocol.get('need_objective','softmax'),
        query_source=query_source,query_binding=binding,term_frequency=term_frequency,fit_ids=protocol['fit_ids'],tune_ids=protocol['tune_ids'],
        data_source=protocol['data_source'],training_source_fingerprint=protocol['source_fingerprint'],
        artifacts=artifacts,
        boundary='recorded train/tune selection; does not establish that the caller chose the seed before public evaluation')


def _unit_provenance(path):
    from model.common.tracking import TrackingError,load_run,verify_run
    try:
        problems=verify_run(path)
        if problems:raise ValueError('metric run unit is not complete and verified: '+'; '.join(problems))
        data=load_run(path,arrays=False)
    except TrackingError as exc:raise ValueError(f'metric run unit cannot be read: {exc}') from exc
    config=data.config or {}
    if data.identity.get('workstream')!='t3' or config.get('tool')!='t3_need_matching':
        raise ValueError('metric run unit was not written by t3_need_matching')
    final=data.final();protocol=data.record('protocol');state=data.states().get('metric')
    if state is None:raise ValueError('metric run unit holds no metric state')
    run=dict(seed=final['seed'],selected_epoch=final['selected_epoch'])
    if run['seed']!=config['seed'] or Path(path).name!=f"s{run['seed']}" or run['seed'] not in protocol['seeds']:
        raise ValueError('metric checkpoint seed directory mismatch')
    if protocol['epochs']!=config['epochs'] or protocol['data_source']['sha256']!=config['data_sha256']:
        raise ValueError('metric unit protocol differs from its config')
    artifacts=dict(run_unit=dict(path=str(Path(path).resolve()),run_id=data.info.run_id,result_digest=data.info.result_digest,
        config_sha256=data.info.config_sha256,state_sha256=state['state_sha256']))
    return _binding(protocol,run,state['file']['sha256'],artifacts)


def metric_provenance(checkpoint):
    checkpoint=Path(checkpoint)
    if is_metric_unit(checkpoint):return _unit_provenance(checkpoint)
    root=checkpoint.parent.parent
    protocol_path=root/'protocol.json'
    summary_path=root/'summary.json'
    seed_path=checkpoint.parent/'summary.json'
    protocol=json.loads(protocol_path.read_bytes())
    summary=json.loads(summary_path.read_bytes())
    seed_summary=json.loads(seed_path.read_bytes())
    if summary.get('protocol')!=protocol:
        raise ValueError('metric training protocol differs from recorded summary')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runs=[run for run in summary['runs'] if run.get('checkpoint_sha256')==digest]
    if len(runs)!=1:
        raise ValueError('metric checkpoint is not uniquely bound to a recorded training run')
    run=runs[0]
    if {key:value for key,value in run.items() if key!='checkpoint_sha256'}!=seed_summary:
        raise ValueError('metric seed summary differs from recorded training run')
    if checkpoint.parent.name!=f"s{run['seed']}":
        raise ValueError('metric checkpoint seed directory mismatch')
    return _binding(protocol,run,digest,{name:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                         for name,path in [('protocol',protocol_path),('summary',summary_path),('seed_summary',seed_path)]})


def metric_checkpoint_sha256(checkpoint):
    """SHA-256 of the metric weights: the bytes of a legacy metric.pt,or a unit's verified weights/metric.pt."""
    checkpoint=Path(checkpoint)
    if is_metric_unit(checkpoint):return _unit_provenance(checkpoint)['checkpoint_sha256']
    return hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def load_metric_checkpoint(checkpoint):
    """The checkpoint dict (selected_metric,selected_optimizer,final_metric,...) of either form,on the CPU."""
    import torch
    checkpoint=Path(checkpoint)
    if is_metric_unit(checkpoint):
        from model.common.tracking import load_state
        return load_state(checkpoint,'metric',map_location='cpu')
    return torch.load(checkpoint,weights_only=True,map_location='cpu')
