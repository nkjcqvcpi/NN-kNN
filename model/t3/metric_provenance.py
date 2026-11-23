"""Bind a public-evaluation metric to its recorded train/tune selection.

The metric is a complete run unit ``runs/t3/<study>/s<seed>/`` of ``tools/t3_need_matching.py``
(recording is mandatory), bound through the unit's TensorBoard records and its verified weight
file ``weights/metric.pt``. Legacy ``s<seed>/metric.pt`` files are no longer read.
"""
from pathlib import Path


def is_metric_unit(path):
    """True for a run unit directory (it holds event files)."""
    path=Path(path)
    return path.is_dir() and any('tfevents' in p.name for p in path.iterdir() if p.is_file())


def _require_unit(path):
    path=Path(path)
    if not is_metric_unit(path):
        raise ValueError(f'{path} is not a t3_need_matching run unit (runs/t3/<study>/s<seed>/); '
                         'legacy metric.pt files are no longer read')
    return path


def _binding(protocol,run,digest,artifacts):
    if protocol.get('public_dev_annotations_used') is not False:
        raise ValueError('public evaluation requires recorded train-only metric fitting')
    selection=protocol['selection']
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
    binding=protocol.get('query_binding')
    if (protocol['epochs']!=config['epochs'] or protocol['data_source']['sha256']!=config['data_sha256']
            or protocol['selection']!=config['selection'] or protocol.get('need_objective','softmax')!=config['objective']
            or protocol.get('term_frequency','count')!=config['term_frequency'] or protocol['dimensions']!=config['dimensions']
            or (binding['sha256'] if binding else None)!=config['need_queries_sha256']):
        raise ValueError('metric unit protocol differs from its config')
    artifacts=dict(run_unit=dict(path=str(Path(path).resolve()),run_id=data.info.run_id,result_digest=data.info.result_digest,
        config_sha256=data.info.config_sha256,state_sha256=state['state_sha256']))
    return _binding(protocol,run,state['file']['sha256'],artifacts)


def metric_provenance(checkpoint):
    """The recorded train/tune selection of a metric run unit (ValueError if it is not a complete,verified unit)."""
    return _unit_provenance(_require_unit(checkpoint))


def metric_checkpoint_sha256(checkpoint):
    """SHA-256 of the unit's verified weights/metric.pt."""
    return metric_provenance(checkpoint)['checkpoint_sha256']


def load_metric_checkpoint(checkpoint):
    """The checkpoint dict (selected_metric,selected_optimizer,final_metric,...) recorded in the unit,on the CPU."""
    from model.common.tracking import load_state
    return load_state(_require_unit(checkpoint),'metric',map_location='cpu')
