"""Build and freeze v6 request/prohibition contrasts without changing older inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import BertTokenizerFast

from prepare_intent_v5 import DATA as V5
from train_intent_contrast import ROOT, read_rows, sha256, write_json
from train_intent_v4 import DATA as V4, load_development
from train_intent_v5 import verify_data as verify_v5
from validate_permission_intent_v2 import normalized


DATA = ROOT / 'ml/data/permission_intent_v6'
PAIRS = [('pick_photo','read_photo'), ('add_photo','photo'), ('file_open','file_save'),
         ('contact_pick','contact_read'), ('contact_read','contact_write'), ('cal_read','cal_write'),
         ('notify','notify_settings'), ('rem_exact','rem_local'), ('bt_scan','bt_connect'),
         ('wlan_scan','wlan_connect'), ('nfc_read','nfc_write'), ('steps','activity'), ('bio','photo'),
         ('clip_paste','clip_read'), ('clip_write','clip_read'), ('video','mic'),
         ('coarse','precise'), ('precise','background')]


def lines(name):
    for line in (DATA / name).read_text().splitlines():
        if line and not line.startswith('#'):
            yield line.split('|', 3)


def make_row(split, group, idx, request, avoid, text, source):
    if set(request) & set(avoid):
        raise ValueError('A v6 role contrast must have disjoint gold list entries')
    identifier = f'v6_{split}_{group}'
    kind = 'mixed' if request and avoid else ('request_only' if request else ('prohibition_only' if avoid else 'neither'))
    return {'id':f'{identifier}_{idx}', 'family_id':identifier, 'scenario_group_id':identifier,
            'split':split, 'kind':kind, 'text':text, 'intent_status':'in_scope' if request else 'no_device_capability',
            'requested_evidence':request, 'prohibited_evidence':avoid,
            'required_capabilities':request, 'forbidden_capabilities':avoid,
            'overlapping_capabilities':[], 'disjoint_capabilities':[], 'source':source, 'human_review':'pending'}


def authored(split, name, options):
    count=Counter()
    for group, requested, prohibited, text in lines(name):
        count[group]+=1
        lists=[[options[alias][0] for alias in field.split(',')] if field!='-' else [] for field in (requested,prohibited)]
        yield make_row(split,group,count[group],*lists,text,'assistant_authored_roles')


def training(options):
    frames=(lambda req,ban:f'我只授权{req}，不要{ban}。',
            lambda req,ban:f'在这一步别{ban}。我想请你{req}。',
            lambda req,ban:f'实际操作要{req}；限制写清楚：不允许{ban}。',
            lambda req,ban:f'本次要完成的是{req}，绝不请求{ban}。')
    for number,(one,two) in enumerate(PAIRS):
        first,a,b=options[one]
        second,c,d=options[two]
        for j,frame in enumerate(frames):
            left,right=(a,c) if j%2==0 else (b,d)
            group=f'role_pair_{number:02d}_{j}'
            yield make_row('train',group,1,[first],[second],frame(left,right),'controlled_sibling_contrast')
            yield make_row('train',group,2,[second],[first],frame(right,left),'controlled_sibling_contrast')
        # Both named action phrases can appear without allocating the other role.
        for j,(req,other,cap,deny) in enumerate(((a,d,first,second),(c,b,second,first))):
            group=f'mention_{number:02d}_{j}'
            yield make_row('train',group,1,[cap],[],f'“{other}”是界面上的功能名称。我实际要{req}。','controlled_mention')
            yield make_row('train',group,2,[],[deny],f'在页面展示“{req}”作为标题，禁止{other}。','controlled_mention')
    for j,(cap,a,b) in enumerate(options.values()):
        examples=[([cap],[],f'现在帮我{a}。'),([cap],[],f'我想直接{b}。'),
                  ([],[cap],f'整个操作严禁{a}。'),([],[cap],f'本次不授权{b}。'),
                  ([],[],f'把“{a}”作为应用里的一行标题，居中排版。')]
        for i,(req,ban,text) in enumerate(examples,1):
            yield make_row('train',f'isolated_{j:02d}',i,req,ban,text,'controlled_isolated_role')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args()
    if (DATA/'freeze_manifest.json').exists():
        raise ValueError('Refusing to overwrite frozen v6 files')
    old,caps,_=load_development()
    verify_v5()
    options={alias:(cap,a,b) for alias,cap,a,b in lines('actions.txt')}
    if set(caps)!={cap for cap,_,_ in options.values()} or len(options)!=len(caps):
        raise ValueError('Training lexicon must cover the 32 capability IDs exactly')
    held=old['train']+old['validation']
    for path in [V4/'reserved_test.jsonl',V4/'product_examples.jsonl',
                 ROOT/'ml/data/permission_intent_v3/reserved_test.jsonl',
                 ROOT/'ml/data/permission_intent_v2/expansion/all_instructions.jsonl',
                 *(V5/f'{name}.jsonl' for name in ('train','validation','reserved_test'))]:
        held.extend(read_rows(path))
    seen={normalized(r['text']) for r in held}
    tokenizer=BertTokenizerFast.from_pretrained(args.tokenizer,local_files_only=True)
    summary={}
    splits={'validation':list(authored('validation','development.txt',options)),
            'reserved_test':list(authored('reserved_test','reserved.txt',options)),
            'train':list(training(options))}
    for split,rows in splits.items():
        lengths=tokenizer([r['text'] for r in rows],truncation=False,return_length=True)['length']
        discarded=set()
        for r,n in zip(rows,lengths):
            key=normalized(r['text'])
            if key in seen or n>64:
                if split!='train':
                    raise ValueError(f'{r["id"]}: duplicate={key in seen}, tokens={n}: {r["text"]}')
                discarded.add(r['scenario_group_id'])
            seen.add(key)
            r['tokens']=n
        rows=[r for r in rows if r['scenario_group_id'] not in discarded]
        scope={role:len(set(cap for r in rows for cap in r[role])) for role in ('requested_evidence','prohibited_evidence')}
        summary[split]={'rows':len(rows),'groups':len({r['scenario_group_id'] for r in rows}),
                        'by_kind':dict(Counter(r['kind'] for r in rows)), 'caps_covered':scope,
                        'max_tokens':max(r['tokens'] for r in rows), 'discarded_groups':len(discarded)}
        (DATA/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    names=('actions.txt','development.txt','reserved.txt','experiment_plan.json','train.jsonl','validation.jsonl','reserved_test.jsonl')
    write_json(DATA/'freeze_manifest.json',{'files_sha256':{n:sha256(DATA/n) for n in names},
               'generator_sha256':sha256(Path(__file__)),'summary':summary,'human_review':'pending'})
    print(summary)


if __name__=='__main__':
    main()
