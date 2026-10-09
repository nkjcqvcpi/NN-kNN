"""Validated role/target-specific case quality journal, without a selector."""
import copy
import hashlib
import json
import math


class QualityLedger:
    def __init__(self, namespace):
        if not isinstance(namespace,str) or not namespace:
            raise ValueError('quality namespace must be explicit')
        self.namespace=namespace
        self.events={}
        self.rows={}

    def apply(self,event,*,event_id,active_ids):
        """Validate the whole transaction before changing any case statistic."""
        if not isinstance(event_id,str) or not event_id:
            raise ValueError('event_id must be explicit')
        role,stream=event['role'],event['stream']
        allowed={('critic','training_gae'),('critic','independent_mc'),('actor','actor_policy_surrogate')}
        if (role,stream) not in allowed:raise ValueError('unsupported role/target stream')
        snapshot=event.get('snapshot_sha256')
        if not isinstance(snapshot,str) or len(snapshot)!=64 or any(c not in '0123456789abcdef' for c in snapshot) or not event.get('phase'):
            raise ValueError('quality requires actual snapshot and phase provenance')
        if type(event.get('global_step')) is not int or event['global_step']<0:
            raise ValueError('global_step must be a nonnegative integer')
        if event.get('credit_rule')!='unweighted_positive_negative_removal_loss_delta':
            raise ValueError('unsupported credit semantics')
        ids=event['case_ids'];weights=event['weights']
        if len(ids)!=len(weights) or len(set(ids))!=len(ids):
            raise ValueError('retrieval identities must be unique and align to weights')
        if any(type(i) is not int or i<0 for i in ids):raise ValueError('invalid stable identity')
        if any(not math.isfinite(w) or w<0 for w in weights):raise ValueError('invalid retrieval weight')
        if not math.isclose(sum(weights),1.,abs_tol=2e-5,rel_tol=2e-5):raise ValueError('retrieval mass must sum to one')
        active=set(active_ids)
        if any(type(i) is not int or i<0 for i in active):raise ValueError('invalid active identities')
        if not set(ids)<=active:raise ValueError('event contains inactive identities')
        interventions=event['interventions']
        used={i for i,w in zip(ids,weights) if w>0}
        if len({i['case_id'] for i in interventions})!=len(interventions) or {i['case_id'] for i in interventions}!=used:
            raise ValueError('every actual positive-weight case needs exactly one intervention')
        s=event['smoothing'];full=event['full_loss']
        if not math.isfinite(s) or s<=0 or not math.isfinite(full):raise ValueError('invalid prior or loss')
        proposed=[]
        for intervention in interventions:
            delta=intervention['delta'];removed=intervention['removed_loss']
            if not math.isfinite(delta) or not math.isfinite(removed):raise ValueError('nonfinite intervention')
            if not math.isclose(delta,removed-full,rel_tol=2e-5,abs_tol=2e-5):raise ValueError('loss delta mismatch')
            c,h=max(delta,0.),max(-delta,0.)
            for actual,expected in [(intervention['C'],c),(intervention['H'],h),(intervention['Q'],(c+s)/(c+h+2*s))]:
                if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=2e-5,abs_tol=2e-5):
                    raise ValueError('credit mismatch')
            key=(role,stream,intervention['case_id'])
            old=self.rows.get(key,dict(C=0.,H=0.,exposure=0,smoothing=s))
            if old['smoothing']!=s:raise ValueError('cannot merge smoothing priors')
            row=dict(C=old['C']+c,H=old['H']+h,exposure=old['exposure']+1,smoothing=s,
                last_global_step=event.get('global_step'),last_phase=event['phase'],
                last_snapshot_sha256=event['snapshot_sha256'])
            if not math.isfinite(row['C']+row['H']+2*s):
                raise ValueError('accumulated support mass overflow')
            row['Q']=(row['C']+s)/(row['C']+row['H']+2*s)
            proposed.append((key,row))
        digest=hashlib.sha256(json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if event_id in self.events:
            if self.events[event_id]!=digest:raise ValueError('event identity reused with changed content')
            return False
        for key,row in proposed:self.rows[key]=row
        self.events[event_id]=digest
        return True

    def state_dict(self):
        return dict(namespace=self.namespace,events=copy.deepcopy(self.events),rows=[dict(role=key[0],stream=key[1],
            case_id=key[2],**copy.deepcopy(value)) for key,value in sorted(self.rows.items())],
            semantics='accumulated direct-loss candidate, separate streams; no calibration/return guarantee')

    @classmethod
    def from_state_dict(cls,state):
        ledger=cls(state['namespace'])
        ledger.events=copy.deepcopy(state['events'])
        for record in state['rows']:
            row=copy.deepcopy(record)
            key=(row.pop('role'),row.pop('stream'),row.pop('case_id'))
            if key in ledger.rows:raise ValueError('duplicate stored quality row')
            if (key[0],key[1]) not in {('critic','training_gae'),('critic','independent_mc'),('actor','actor_policy_surrogate')} or type(key[2]) is not int or key[2]<0:
                raise ValueError('invalid stored role identity')
            if any(not math.isfinite(row[k]) or row[k]<0 for k in ('C','H')) or not math.isfinite(row['smoothing']) or row['smoothing']<=0:
                raise ValueError('invalid stored contribution')
            if type(row['exposure']) is not int or row['exposure']<1:
                raise ValueError('invalid stored exposure')
            expected=(row['C']+row['smoothing'])/(row['C']+row['H']+2*row['smoothing'])
            if not math.isfinite(row['C']+row['H']+2*row['smoothing']):
                raise ValueError('stored support mass overflow')
            if not math.isfinite(row['Q']) or not math.isclose(row['Q'],expected,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('stored Q mismatch')
            ledger.rows[key]=row
        return ledger

    def active_rows(self,role,stream,active_ids):
        active=set(active_ids)
        return {key[2]:copy.deepcopy(row) for key,row in self.rows.items()
                if key[0]==role and key[1]==stream and key[2] in active}
