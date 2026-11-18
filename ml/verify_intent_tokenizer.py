"""Compare saved HF tokenizer with the application's unchanged WordPiece code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from transformers import BertTokenizerFast

from permission_intent_model import encode


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "ml/data/permission_intent_v2/expansion/all_instructions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint, local_files_only=True)
    encoded = encode(tokenizer, rows, 64)
    references = [{"id": r["id"], "text": r["text"], "ids": encoded["input_ids"][i].tolist(),
                   "mask": encoded["attention_mask"][i].tolist()} for i, r in enumerate(rows)]
    source = ROOT / "entry/src/main/ets/model/WordPiece.ets"
    vocab = args.checkpoint / "vocab.txt"
    node = shutil.which("node")
    if not node:
        raise ValueError("Node 22+ is required for actual application tokenizer verification")
    with tempfile.TemporaryDirectory(prefix="intent-tokenizer-") as temporary:
        path = Path(temporary)
        shutil.copyfile(source, path / "WordPiece.mts")
        (path / "references.json").write_text(json.dumps(references, ensure_ascii=False))
        (path / "check.mjs").write_text('''
import fs from 'node:fs';
import {WordPiece} from './WordPiece.mts';
const refs=JSON.parse(fs.readFileSync(new URL('./references.json',import.meta.url),'utf8'));
const tokenizer=new WordPiece(fs.readFileSync(process.argv[2],'utf8'));
const mismatches=[];
for(const row of refs){
 const actual=tokenizer.encode(row.text);
 if(actual.tooLong || JSON.stringify(Array.from(actual.ids))!==JSON.stringify(row.ids) ||
    JSON.stringify(Array.from(actual.mask))!==JSON.stringify(row.mask)) mismatches.push(row.id);
}
process.stdout.write(JSON.stringify({rows:refs.length,mismatch_ids:mismatches}));
''')
        result = subprocess.run([node, "--experimental-strip-types", str(path / "check.mjs"), str(vocab.resolve())],
                                capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    report.update(passed=not report["mismatch_ids"], scope="Actual app WordPiece versus saved HF tokenizer; no NPU execution",
                  node_version=subprocess.check_output([node, "--version"], text=True).strip(),
                  input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  app_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  vocab_sha256=hashlib.sha256(vocab.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError("Application and training tokenizers differ: " + str(report["mismatch_ids"]))
    print(f"PASS: saved tokenizer and actual app WordPiece agree on all {len(rows)} raw texts")


if __name__ == "__main__":
    main()
