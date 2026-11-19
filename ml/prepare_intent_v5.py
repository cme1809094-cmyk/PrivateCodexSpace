"""Freeze training evidence and separately authored development/test scenarios."""
import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import BertTokenizerFast

from train_intent_contrast import ROOT, read_json, read_rows, sha256, write_json
from train_intent_v4 import DATA as V4, load_development
from validate_permission_intent_v2 import normalized


DATA = ROOT / 'ml/data/permission_intent_v5'
ALIASES = {
    'cal_read': 'calendar.read', 'cal_write': 'calendar.write', 'nfc_read': 'nfc.read_tag',
    'nfc_write': 'nfc.write_tag', 'cam_photo': 'camera.capture_photo', 'video': 'camera.record_video',
    'mic': 'microphone.record_audio', 'bt_connect': 'bluetooth.connect', 'bt_scan': 'bluetooth.scan',
    'bluetooth_scan': 'bluetooth.scan', 'clip_read': 'clipboard.read', 'clip_write': 'clipboard.write',
    'notify': 'notifications.publish', 'notify_settings': 'notifications.open_settings',
    'contacts_write': 'contacts.write', 'contacts_read': 'contacts.read', 'contacts_pick': 'contacts.pick_one',
    'loc_precise': 'location.precise.foreground_once', 'loc_coarse': 'location.coarse.foreground_once',
    'loc_bg': 'location.background', 'wlan_scan': 'wlan.scan', 'wlan_connect': 'wlan.connect',
    'media_read': 'media.photo.read_library', 'media_pick': 'media.photo.pick_one', 'media_add': 'media.photo.add',
    'rem_local': 'reminders.schedule_local', 'rem_exact': 'reminders.schedule_exact',
    'files_save': 'files.save_user_selected', 'files_pick': 'files.pick_one', 'bio': 'biometric.authenticate',
}
# Both actions in a pair use the same capability, with independently named scopes.
SCOPES = [
    ('cal_write', '给团队日历增加一次复盘会', '向我的生日历增加一个预约'),
    ('cal_read', '读取办公日历六月的安排', '读取家庭日历六月的安排'),
    ('contacts_write', '更新谢工程师的通讯录备注', '修改冯医生的通讯录备注'),
    ('contacts_read', '查询通讯录里的业务分组号码', '查看通讯录里的家属分组号码'),
    ('mic', '录下发布会演讲的声音', '录下餐叙的声音'),
    ('video', '用相机录下颁奖过程', '用相机录下休息室的过程'),
    ('cam_photo', '拍摄展台的模型', '拍摄隔壁的游客'),
    ('wlan_connect', '连上已配置的实验室无线网', '连接访客无线网'),
    ('bt_connect', '接通已配对的会议室蓝牙音箱', '配对门口的蓝牙耳机'),
    ('nfc_read', '读测试台的绿色NFC标签', '读仓库的红色NFC标签'),
    ('nfc_write', '给授权的绿色测试NFC标签写入工号', '给仓库的红色NFC标签写入工号'),
    ('notify', '把出库消息发到系统通知栏', '把促销消息发到系统通知栏'),
    ('media_read', '直接查图库的培训相册', '直接查图库的聚会相册'),
    ('media_add', '把培训结果图添加到公共相册', '把未审稿的图片添加到公共相册'),
    ('loc_bg', '退到后台后仅在比赛赛段记录位置', '退到后台时在宿舍继续获取位置'),
    ('rem_local', '安排一个可延后的喝水提醒', '安排可延后的订阅购买提醒'),
]


def make_row(split, group, index, kind, desired, denied, text, source):
    common = sorted(set(desired) & set(denied))
    if bool(common) != (kind in ('overlap', 'disjoint')):
        raise ValueError(f'Inconsistent evidence for {split}/{group}: {text}')
    gid = f'v5_{split}_{group}'
    return {'id': f'{gid}_{index}', 'family_id': f'{gid}_{kind}', 'scenario_group_id': gid,
            'split': split, 'kind': kind, 'text': text,
            'requested_evidence': desired, 'prohibited_evidence': denied,
            'overlapping_capabilities': common if kind == 'overlap' else [],
            'disjoint_capabilities': common if kind == 'disjoint' else [],
            'required_capabilities': [] if common else desired,
            'forbidden_capabilities': denied,
            'intent_status': 'ambiguous' if kind == 'overlap' else ('in_scope' if desired else 'no_device_capability'),
            'source': source, 'human_review': 'pending'}


def authored(split, name):
    count = Counter()
    for line in (DATA / name).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        group, kind, request, ban, text = line.split('|', 4)
        count[group] += 1
        yield make_row(split, group, count[group], kind, [ALIASES[request]] if request != '-' else [],
                       [ALIASES[ban]] if ban != '-' else [], text, 'assistant_authored_contrast')


def training_rows(caps):
    lexicon = read_json(V4 / 'lexicon.json')['capabilities']
    frames = [
        lambda p, n: f'请按要求{p}；本次不得{n}，这个限制照旧。',
        lambda p, n: f'我坚持不让你{n}，处理事项时还是请你{p}。',
        lambda p, n: f'等我确认就{p}，而本轮禁令依然是不能{n}。',
        lambda p, n: f'任务是{p}。牢记我的限制：严禁{n}。',
    ]
    for i, cap in enumerate(caps):
        for j, frame in enumerate(frames):
            other = caps[(i + (1, 7, 13, 19)[j]) % len(caps)]
            deny = lexicon[cap]['negative']
            group = f'template_{i:02}_{j}'
            for k, (kind, req) in enumerate((('overlap', cap), ('distinct', other)), 1):
                wanted = lexicon[req]['positive'][j % 2]
                yield make_row('train', group, k, kind, [req], [cap], frame(wanted, deny), 'controlled_v4_lexicon')
    for i, (alias, permitted, restricted) in enumerate(SCOPES):
        cap = ALIASES[alias]
        general = lexicon[cap]['negative']
        scope_frames = [
            lambda p, b: f'只同意{p}，要求不要{b}。',
            lambda p, b: f'务必别{b}，仍允许{p}。',
            lambda p, b: f'安排是先{p}，限制是不可以{b}。',
        ]
        for j, frame in enumerate(scope_frames):
            group = f'scope_{i:02}_{j}'
            yield make_row('train', group, 1, 'disjoint', [cap], [cap], frame(permitted, restricted), 'controlled_scope_pair')
            yield make_row('train', group, 2, 'overlap', [cap], [cap], frame(permitted, general), 'controlled_scope_pair')
        yield make_row('train', f'revision_{i:02}', 1, 'revision', [cap], [],
                       f'说明改一下，之前说不能{restricted}那条撤回，现在请{permitted}。', 'controlled_revision')
        yield make_row('train', f'revision_{i:02}', 2, 'revision', [], [cap],
                       f'先前说要{restricted}的请求取消。现在明确禁止{general}。', 'controlled_revision')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    manifest = DATA / 'freeze_manifest.json'
    if manifest.exists():
        raise ValueError('Refusing to overwrite frozen v5 inputs')
    old, caps, _ = load_development()
    known = old['train'] + old['validation']
    for path in [V4 / 'reserved_test.jsonl', V4 / 'product_examples.jsonl',
                 ROOT / 'ml/data/permission_intent_v3/reserved_test.jsonl',
                 ROOT / 'ml/data/permission_intent_v2/expansion/all_instructions.jsonl']:
        known.extend(read_rows(path))
    seen = {normalized(r['text']) for r in known}
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
    built = {'train': list(training_rows(caps)), 'validation': list(authored('validation', 'development_cases.txt')),
             'reserved_test': list(authored('reserved_test', 'reserved_cases.txt'))}
    summary = {}
    for name, rows in built.items():
        retained, skipped_groups = [], set()
        for row in rows:
            key = normalized(row['text'])
            length = len(tokenizer(row['text'], truncation=False)['input_ids'])
            if key in seen or length > 64:
                if name != 'train':
                    raise ValueError(f'Invalid held-out authoring: {row["id"]}, length={length}, duplicate={key in seen}')
                skipped_groups.add(row['scenario_group_id'])
            seen.add(key)
            retained.append(dict(row, tokens=length))
        rows = [r for r in retained if r['scenario_group_id'] not in skipped_groups]
        (DATA / (name + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        summary[name] = {'rows': len(rows), 'kinds': dict(Counter(r['kind'] for r in rows)),
                         'groups': len({r['scenario_group_id'] for r in rows}),
                         'at_most_16_tokens': sum(r['tokens'] <= 16 for r in rows), 'skipped_train_groups': len(skipped_groups)}
    names = ['train.jsonl', 'validation.jsonl', 'reserved_test.jsonl', 'development_cases.txt', 'reserved_cases.txt', 'experiment_plan.json']
    write_json(manifest, {'files_sha256': {name: sha256(DATA / name) for name in names},
                          'generator_sha256': sha256(Path(__file__)), 'summary': summary,
                          'authoring': 'Assistant-authored validation and reserved cases; human review pending',
                          'test_rules': 'Reserved texts are authored and checked before training; only file hashes are used during training.'})
    print(summary)


if __name__ == '__main__':
    main()
