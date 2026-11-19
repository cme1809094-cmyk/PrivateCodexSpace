"""Evaluate frozen requested/prohibited lists separately from authorization policy."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from permission_evidence import EVIDENCE_CHANNELS, decode_profile, evidence_targets, infer, load_profile
from permission_intent_model import encode, labels, metrics
from prepare_intent_v5 import DATA as V5
from train_intent_contrast import read_json, read_rows, write_json
from train_intent_v6 import CHANNELS, DATA, list_metrics, list_predictions, verify_data, verify_profile


def policy_report(cap, intent, protocol, rows, capabilities):
    view=decode_profile(cap,intent,protocol)
    target,status=(t.numpy() for t in labels(rows,capabilities))
    actual=view['state']
    conflict=np.array([r.get('kind')=='overlap' for r in rows])
    normal=(status==0)&np.array([r.get('kind')!='disjoint' for r in rows])
    accepted=view['accepted']
    prohibited=(target==2)
    leaked=((actual==1)&prohibited).any(axis=1)
    result=metrics(actual,view['status'],target,status)
    return {'normal_request_rows':int(normal.sum()),'normal_accepted_rows':int((normal&accepted).sum()),
            'normal_acceptance_rate':float((normal&accepted).sum()/max(1,normal.sum())),
            'required_recall':result['required']['recall'],'required_f1':result['required']['f1'],
            'overrequested_rows':result['overrequested_samples'],
            'contains_prohibition_rows':int(prohibited.any(axis=1).sum()),'prohibited_issued_rows':int(leaked.sum()),
            'conflict_rows':int(conflict.sum()),'conflicts_classified':int((conflict&(view['status']==1)).sum()),
            'conflicts_clarified':int((conflict&(view['decision']=='clarify')).sum()),
            'conflicts_with_prohibited_issued':int((conflict&leaked).sum())}, view


def assess(name, model, tokenizer, protocol, datasets, capabilities, out):
    results, records={},{}
    for key,rows in datasets.items():
        cap,intent=infer(model,encode(tokenizer,rows,64),True)
        gold,known=evidence_targets(rows,capabilities)
        if not known[:,:,:2].all():
            raise ValueError(f'Raw list comparison requires known labels for both roles: {key}')
        target=gold[:,:,:2].numpy().astype(bool)
        predicted=list_predictions(cap,protocol['thresholds'])
        policy,view=policy_report(cap,intent,protocol,rows,capabilities)
        categories=defaultdict(list)
        for i,r in enumerate(rows):
            categories[r['kind']].append(i)
        stats=list_metrics(predicted,target)
        results[key]={'lists':stats,'policy':policy,
                      'by_kind':{kind:list_metrics(predicted[idx],target[idx]) for kind,idx in categories.items()}}
        def names(mask):
            return [capability for capability,hit in zip(capabilities,mask) if hit]
        table=[]
        for i,row in enumerate(rows):
            record={'id':row['id'],'text':row['text'],'kind':row['kind'],'decision':str(view['decision'][i]),
                    'policy_required':names(view['state'][i]==1),
                    'both_lists_exact':bool((predicted[i]==target[i]).all())}
            for j,channel in enumerate(CHANNELS):
                record['expected_'+channel]=names(target[i,:,j])
                record[channel]=names(predicted[i,:,j])
                record['missing_'+channel]=names(target[i,:,j]&~predicted[i,:,j])
                record['extra_'+channel]=names(predicted[i,:,j]&~target[i,:,j])
            table.append(record)
        (out/f'{key}_{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in table))
        records[key]=table
    return results,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty v6 evaluation directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    frozen=verify_data()
    original=verify_profile(args.baseline)
    selected=verify_profile(args.run/'checkpoint')
    previous=read_json(args.run/'baseline_validation.json')
    training=read_json(args.run/'training_report.json')
    reserved=frozen['files_sha256']['reserved_test.jsonl']
    if selected['reserved_prompts_read'] or selected['reserved_sha256']!=reserved or previous['reserved_sha256']!=reserved:
        raise ValueError('Checkpoint or calibration was not locked against these reserved items')
    if previous['baseline_weights_sha256']!=original['weights_sha256']:
        raise ValueError('Original and recalibrated profiles must use the same weights')
    model,tokenizer,protocol=load_profile(args.baseline)
    final,final_tokenizer,final_protocol=load_profile(args.run/'checkpoint')
    caps=protocol['capability_ids']
    if (protocol['states']!=EVIDENCE_CHANNELS or final_protocol['states']!=EVIDENCE_CHANNELS
        or final_protocol['capability_ids']!=caps or tokenizer.get_vocab()!=final_tokenizer.get_vocab()
        or final_protocol['policy']!=protocol['policy']):
        raise ValueError('Role order, vocabulary and policy must match the v5 baseline')
    selection={'scores':{'v5_original':previous['original']['score'],'v5_recalibrated':previous['calibrated']['score'],
                         'v6_trained':training['trained_candidate_validation']['score']},
               'selected_on_validation':training['selected_on_validation'],'new_reserved_read_when_selected':False,
               'reserved_sha256':reserved,'checkpoint':selected,'v5_recalibrated_thresholds':previous['thresholds']}
    write_json(args.output/'final_selection.json',selection)
    datasets={'new80':read_rows(DATA/'reserved_test.jsonl'),'known_v5_64':read_rows(V5/'reserved_test.jsonl')}
    arms=[('v5_original',model,tokenizer,protocol),
          ('v5_recalibrated',model,tokenizer,{**protocol,'thresholds':previous['thresholds']}),
          ('v6_trained',final,final_tokenizer,final_protocol)]
    report={'selection':selection,'human_review':'pending','acceptance_scope':'Assistant-authored data; exact request and avoid lists before policy',
            'known_v5_64_scope':'Previously observed regression only','holdout_used_to_tune':False,'candidates':{}}
    predictions={}
    for name,network,words,profile in arms:
        report['candidates'][name],predictions[name]=assess(name,network,words,profile,datasets,caps,args.output)
    with (args.output/'new80_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['id','text','kind','true_requested','true_prohibited',
                         *(f'{name}_{head}' for name,*_ in arms for head in CHANNELS),'human_review','review_notes'])
        for idx,row in enumerate(datasets['new80']):
            values=[predictions[name]['new80'][idx][head] for name,*_ in arms for head in CHANNELS]
            writer.writerow([row['id'],row['text'],row['kind'],', '.join(row['requested_evidence']),
                             ', '.join(row['prohibited_evidence']),*(', '.join(v) for v in values),'pending',''])
    # Reproduce the published raw list scores and the conflict check before comparing improvements.
    old=report['candidates']['v5_original']['known_v5_64']['lists']
    if (old['requested']['tp'],old['prohibited']['tp'])!=(50,32):
        raise ValueError('The known v5 baseline must reproduce the published 50/62 requests and 32/60 prohibitions')
    report['original_v5_reproduced']=True
    write_json(args.output/'comparison_report.json',report)
    print(json.dumps({'selected':selection['selected_on_validation'],'scores':selection['scores'],
                      'reserved':{name:ans['new80']['lists'] for name,ans in report['candidates'].items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
