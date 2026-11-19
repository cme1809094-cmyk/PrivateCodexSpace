"""Run a paired dual-evidence checkpoint on one or more Chinese requests."""
import argparse
import json
from pathlib import Path

import torch

from permission_evidence import EVIDENCE_CHANNELS, decode_profile, infer, load_profile
from permission_intent_model import INTENTS, encode
from train_intent_v5 import verify_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('text', nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_profile(args.checkpoint)
    model, tokenizer, protocol = load_profile(args.checkpoint)
    if protocol['states'] != EVIDENCE_CHANNELS:
        raise ValueError('A v5 dual-evidence checkpoint is required')
    names = protocol['capability_ids']
    rows = [{'text': text, 'id': f'input_{i+1}'} for i, text in enumerate(args.text)]
    cap, intent = infer(model, encode(tokenizer, rows, 64), True)
    decoded = decode_profile(cap, intent, protocol)
    for i, text in enumerate(args.text):
        decision = str(decoded['decision'][i])
        mask = decoded['requested'][i] | decoded['prohibited'][i]
        if decoded['overlap'][i].any():
            reason = 'overlapping_request_and_prohibition'
        elif decision == 'review_scope':
            reason = 'disjoint_scopes_need_scoped_action'
        elif decoded['unresolved_scope'][i].any():
            reason = 'scope_overlap_uncertain'
        elif decoded['raw_status'][i] == 1:
            reason = 'ambiguous_intent'
        elif decision == 'clarify':
            reason = 'confidence_gate'
        else:
            reason = decision
        result = {'text': text, 'model_intent': INTENTS[decoded['raw_status'][i]],
                  'intent_status': INTENTS[decoded['status'][i]], 'decision': decision, 'reason': reason,
                  'active_evidence': {name: dict(zip(EVIDENCE_CHANNELS, cap[i, j].tolist()))
                                      for j, name in enumerate(names) if mask[j]},
                  'required_capabilities': [name for j, name in enumerate(names) if decoded['state'][i, j] == 1]}
        result.update({field: [name for j, name in enumerate(names) if decoded[field][i, j]]
                       for field in ('requested', 'prohibited', 'overlap', 'disjoint', 'unresolved_scope')})
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
