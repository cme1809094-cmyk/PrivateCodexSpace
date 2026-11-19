"""Select TinyBERT by request/avoid list quality before applying the policy."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from intent_contrast import packed_group_batches
from intent_optimization import add_training_device_argument, resolve_training_device, training_runtime, trim_padding
from permission_evidence import EVIDENCE_CHANNELS, evidence_loss, evidence_targets, evidence_weights, infer, load_profile
from permission_intent_model import balanced_weights, binary_metrics, encode, labels, save_checkpoint
from prepare_intent_v5 import DATA as V5
from prepare_intent_v6 import DATA
from train_intent_contrast import ROOT, read_json, read_rows, sha256, write_json
from train_intent_v4 import load_development
from train_intent_v5 import verify_data as verify_v5, verify_profile


CHANNELS=('requested','prohibited')
GRID=(0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95)


def verify_data():
    frozen=read_json(DATA/'freeze_manifest.json')
    for name,want in frozen['files_sha256'].items():
        if sha256(DATA/name)!=want:
            raise ValueError(f'Frozen v6 file changed: {name}')
    if sha256(ROOT/'ml/prepare_intent_v6.py')!=frozen['generator_sha256']:
        raise ValueError('Frozen generator changed')
    return frozen


def list_predictions(scores,thresholds):
    cutoffs=np.array([thresholds[name] for name in CHANNELS])
    return scores[:,:,:2]>=cutoffs


def list_metrics(pred, target):
    req,deny=pred[:,:,0],pred[:,:,1]
    true_req,true_deny=target[:,:,0],target[:,:,1]
    extra_req=(req & ~true_req).any(axis=1)
    extra_deny=(deny & ~true_deny).any(axis=1)
    empty=~target.any(axis=(1,2))
    exact=(pred==target).all(axis=(1,2))
    return {'rows':len(pred), 'requested':binary_metrics(req,true_req), 'prohibited':binary_metrics(deny,true_deny),
            'requested_exact':int((req==true_req).all(axis=1).sum()),
            'prohibited_exact':int((deny==true_deny).all(axis=1).sum()),
            'both_exact':int(exact.sum()),'both_exact_rate':float(exact.mean()),
            'extra_request_rows':int(extra_req.sum()), 'extra_prohibition_rows':int(extra_deny.sum()),
            'explicit_prohibition_in_request_rows':int((req & true_deny & ~true_req).any(axis=1).sum()),
            'missed_prohibition_rows':int((true_deny & ~deny).any(axis=1).sum()),
            'mention_only_rows':int(empty.sum()), 'mention_only_false_request_rows':int((req.any(axis=1)&empty).sum()),
            'mention_only_false_prohibition_rows':int((deny.any(axis=1)&empty).sum())}


def score_for(views):
    def score(v):
        return .3*v['requested']['f1']+.3*v['prohibited']['f1']+.4*v['both_exact_rate']
    return .8*score(views['v6'])+.2*score(views['v5'])


def evaluate_lists(scores,target,thresholds):
    views={name:list_metrics(list_predictions(value,thresholds),target[name]) for name,value in scores.items()}
    return {'score':score_for(views),'by_set':views}


def calibrate(scores,target,original):
    best=None
    for wanted in GRID:
        for avoided in GRID:
            thresholds={**original,'requested':wanted,'prohibited':avoided}
            result=evaluate_lists(scores,target,thresholds)
            rank=(result['score'],-abs(wanted-original['requested'])-abs(avoided-original['prohibited']))
            if best is None or rank>best[0]:
                best=rank,thresholds,result
    return best[1:]


def ranking_loss(logits,target,known):
    """Suppress a sibling capability if its score competes with a known active action."""
    scores=logits[:,:,:2]
    gold=target[:,:,:2].bool()
    valid=known[:,:,:2]
    positive=gold & valid
    negative=~gold & valid
    hardest=scores.masked_fill(~negative,-1e4).amax(dim=1)
    # Rank each active capability above the highest scoring wrong capability in its role.
    active=positive & negative.any(dim=1)[:,None,:]
    margin=F.relu(1.0+hardest[:,None,:]-scores)
    ranked=(margin*active).sum()/active.sum().clamp_min(1)
    # An empty list has no positive competitor. Train its highest false positive to stay low.
    empty=(~positive.any(dim=1)) & negative.any(dim=1)
    inactive=(F.softplus(hardest+0.5)*empty).sum()/empty.sum().clamp_min(1)
    return 0.7*ranked+0.3*inactive


def development_sets():
    old,caps,legacy=load_development()
    v5=verify_v5()
    frozen=verify_data()
    training=old['train']+read_rows(V5/'train.jsonl')+read_rows(DATA/'train.jsonl')
    validation={'v6':read_rows(DATA/'validation.jsonl'),'v5':read_rows(V5/'validation.jsonl')}
    return training,validation,caps,{'v6':frozen,'v5':v5,'legacy':legacy}


def scores_for(model, encoded):
    return {name:infer(model,x,True)[0] for name,x in encoded.items()}


def train(args):
    device=resolve_training_device(args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty v6 output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    start=time.monotonic()
    initial_lock=verify_profile(args.initial)
    training,validation,caps,hashes=development_sets()
    model,tokenizer,original=load_profile(args.initial)
    if original['schema_version']!=5 or original['states']!=EVIDENCE_CHANNELS or original['capability_ids']!=caps:
        raise ValueError('Start with the paired v5 dual-evidence profile')
    model.to(device)
    validation_x={name:encode(tokenizer,rows,64) for name,rows in validation.items()}
    validation_y={name:evidence_targets(rows,caps)[0][:,:,:2].numpy().astype(bool) for name,rows in validation.items()}
    base_scores=scores_for(model,validation_x)
    base_thresholds,base_calibrated=calibrate(base_scores,validation_y,original['thresholds'])
    base_original=evaluate_lists(base_scores,validation_y,original['thresholds'])
    write_json(args.output/'baseline_validation.json',{'original':base_original,'calibrated':base_calibrated,
               'thresholds':base_thresholds,'baseline_weights_sha256':initial_lock['weights_sha256'],
               'reserved_sha256':hashes['v6']['files_sha256']['reserved_test.jsonl']})
    x=encode(tokenizer,training,64)
    _,status=labels(training,caps)
    y,known=evidence_targets(training,caps)
    weights=tuple(w.to(device) for w in evidence_weights(y,known))
    intent_weights=balanced_weights(status,4,power=1.0)[0].to(device)
    status,y,known=status.to(device),y.to(device),known.to(device)
    params=[{'params':[p for n,p in model.named_parameters() if n.startswith('encoder.')==encoder],
             'lr':1.5e-5 if encoder else 1e-4} for encoder in (True,False)]
    opt=torch.optim.AdamW(params,weight_decay=.01)
    rng=torch.Generator().manual_seed(args.seed)
    counter=torch.Generator().manual_seed(args.seed)
    steps=sum(len(list(packed_group_batches(training,args.batch_size,counter))) for _ in range(args.epochs))
    sched=get_linear_schedule_with_warmup(opt,max(1,int(steps*.05)),steps)
    sources=['train_intent_v6.py','prepare_intent_v6.py','permission_evidence.py','permission_intent_model.py','intent_contrast.py','intent_optimization.py']
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'runtime':training_runtime(device),
            'rows':{'train':len(training),**{name+'_validation':len(rows) for name,rows in validation.items()}},
            'initial_weights_sha256':initial_lock['weights_sha256'],'frozen_data':hashes,
            'source_sha256':{name:sha256(ROOT/'ml'/name) for name in sources},'reserved_prompts_read':False}
    write_json(args.output/'training_config.json',config)
    snapshot=args.output/'code_snapshot'
    snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((ROOT/'ml'/name).read_bytes())
    print(json.dumps({'event':'start','runtime':config['runtime'],'rows':config['rows'],
                      'base_original_score':base_original['score'],'base_calibrated_score':base_calibrated['score']}),flush=True)
    best=None
    for epoch in range(1,args.epochs+1):
        tick=time.monotonic()
        model.train()
        losses=[]
        for indices in packed_group_batches(training,args.batch_size,rng):
            batch=trim_padding({k:v[indices] for k,v in x.items()})
            cap,intent=model(**{k:v.to(device) for k,v in batch.items()})
            loss=evidence_loss(cap,y[indices],known[indices],weights)+.30*ranking_loss(cap,y[indices],known[indices])
            loss=loss+.5*F.cross_entropy(intent,status[indices],weight=intent_weights,label_smoothing=.03)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        thresholds,result=calibrate(scores_for(model,validation_x),validation_y,original['thresholds'])
        if best is None or result['score']>best[0]:
            best=result['score'],epoch,{name:v.detach().cpu().clone() for name,v in model.state_dict().items()},thresholds,result
        entry={'epoch':epoch,'loss':float(np.mean(losses)),'thresholds':thresholds,
               'seconds':time.monotonic()-tick,**result}
        with (args.output/'history.jsonl').open('a') as f:
            f.write(json.dumps(entry)+'\n')
        print(json.dumps({'epoch':epoch,'score':result['score'],'request_f1':result['by_set']['v6']['requested']['f1'],
                          'avoid_f1':result['by_set']['v6']['prohibited']['f1'],'both_exact':result['by_set']['v6']['both_exact'],
                          'seconds':round(entry['seconds'],2)}),flush=True)
    model.cpu().load_state_dict(best[2])
    profile=copy.deepcopy(original)
    profile.update(schema_version=6,task='permission_intent_v6_role_accuracy',selected_epoch=best[1],thresholds=best[3],
                   primary_output='Independent requested/prohibited lists before policy',warm_start_weights_sha256=initial_lock['weights_sha256'])
    checkpoint=args.output/'checkpoint'
    save_checkpoint(model,tokenizer,checkpoint,profile)
    restored,_,_=load_profile(checkpoint)
    with torch.inference_mode():
        model.eval()
        ref={k:v[:2] for k,v in validation_x['v6'].items()}
        for a,b in zip(model(**ref),restored(**ref)):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    lock={'checkpoint_files_sha256':{p.name:sha256(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
          'weights_sha256':sha256(checkpoint/'model.safetensors'),'protocol_sha256':sha256(checkpoint/'protocol.json'),
          'reserved_sha256':hashes['v6']['files_sha256']['reserved_test.jsonl'],'reserved_prompts_read':False}
    write_json(args.output/'selection_lock.json',lock)
    chosen=max({'v5_original':base_original['score'],'v5_recalibrated':base_calibrated['score'],
                'v6_trained':best[0]},key=lambda name:{'v5_original':base_original['score'],
                    'v5_recalibrated':base_calibrated['score'],'v6_trained':best[0]}[name])
    report={'selected_on_validation':chosen,'trained_candidate_validation':best[4],'selected_epoch':best[1],
            'baseline_original_score':base_original['score'],'baseline_calibrated_score':base_calibrated['score'],
            'checkpoint_roundtrip_exact':True,'seconds':time.monotonic()-start}
    write_json(args.output/'training_report.json',report)
    print(json.dumps({'event':'complete','selected':chosen,'epoch':best[1],'development_score':best[0]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261119)
    add_training_device_argument(parser)
    args=parser.parse_args()
    if args.epochs<1:
        parser.error('--epochs must be positive')
    train(args)


if __name__=='__main__':
    main()
