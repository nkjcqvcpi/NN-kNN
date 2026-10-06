"""Bind a public-evaluation metric to its recorded train/tune selection."""
import hashlib
import json
from pathlib import Path


def metric_provenance(checkpoint):
    checkpoint=Path(checkpoint)
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
        artifacts={name:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                   for name,path in [('protocol',protocol_path),('summary',summary_path),('seed_summary',seed_path)]},
        boundary='recorded train/tune selection; does not establish that the caller chose the seed before public evaluation')
