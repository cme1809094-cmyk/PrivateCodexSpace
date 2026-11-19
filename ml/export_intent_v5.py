"""Export one static dual-evidence ONNX and check independent scores and decisions on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from permission_evidence import EVIDENCE_CHANNELS, decode_profile, load_profile
from permission_intent_model import encode
from train_intent_contrast import read_rows, sha256, write_json
from train_intent_v5 import DATA, verify_data, verify_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, choices=[16, 32, 64], default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_data()
    verify_profile(args.checkpoint)
    model, tokenizer, protocol = load_profile(args.checkpoint)
    if protocol['states'] != EVIDENCE_CHANNELS:
        raise ValueError('Use a paired dual-evidence checkpoint')
    rows = [r for r in read_rows(DATA / 'reserved_test.jsonl') if r['tokens'] <= args.tokens]
    if not rows:
        raise ValueError('No frozen reference rows fit the token bucket')
    data = {key: val.to(torch.int32) for key, val in encode(tokenizer, rows, args.tokens).items()}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f'permission_evidence_{args.tokens}.onnx'
    if path.exists():
        raise ValueError(f'Refusing to overwrite {path}')
    torch.onnx.export(model.eval(), tuple(t[:1] for t in data.values()), str(path),
                      input_names=list(data), output_names=['capability_logits', 'intent_logits'],
                      opset_version=13, dynamo=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    max_error = [0.0, 0.0]
    refs = []
    for i, row in enumerate(rows):
        args_pt = {key: v[i:i+1] for key, v in data.items()}
        with torch.inference_mode():
            ref = tuple(v.numpy() for v in model(**args_pt))
        actual = session.run(None, {key: v.numpy() for key, v in args_pt.items()})
        for j, (want, got) in enumerate(zip(ref, actual)):
            np.testing.assert_allclose(want, got, rtol=1e-4, atol=1e-4)
            max_error[j] = max(max_error[j], float(np.abs(want - got).max()))
        decoded = [decode_profile(torch.from_numpy(a).sigmoid().numpy(), torch.from_numpy(b).softmax(-1).numpy(), protocol)
                   for a, b in (ref, actual)]
        for name in decoded[0]:
            np.testing.assert_array_equal(decoded[0][name], decoded[1][name])
        refs.append({'id': row['id'], **{key: v[0].tolist() for key, v in args_pt.items()},
                     'capability_logits': ref[0][0].tolist(), 'intent_logits': ref[1][0].tolist(),
                     'decision': str(decoded[0]['decision'][0])})
    (args.output / f'reference_{args.tokens}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in refs))
    report = {'tokens': args.tokens, 'onnx': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path),
              'heldout_reference_rows': len(rows), 'logit_max_absolute_errors': max_error, 'decision_mismatches': 0,
              'capability_logit_channels': EVIDENCE_CHANNELS, 'scope': 'PyTorch to ONNX Runtime CPU'}
    write_json(args.output / f'verification_{args.tokens}.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
