"""Task-semantic and artifact guards for the new TinyBERT trainer."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from transformers import BertConfig, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from permission_intent_model import (INTENTS, STATES, PermissionIntentModel, balanced_weights,
                                     decode, encode, labels, load_checkpoint, metrics,
                                     objective, save_checkpoint)
from permission_intent_model import selection_score
from train_permission_intent_v2 import prepare_data


class IntentTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def row(self, **changes):
        return {"id": "case", "text": "录音", "required_capabilities": ["audio"],
                "forbidden_capabilities": ["camera"], "intent_status": "in_scope", **changes}

    def tokenizer(self, path):
        vocab = path / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n录\n音\n不\n要\n")
        return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)

    def test_tristate_keeps_unmentioned_distinct_from_forbidden(self):
        target, intent = labels([self.row()], ["audio", "camera", "location"])
        self.assertEqual(target.tolist(), [[1, 2, 0]])
        self.assertEqual(intent.tolist(), [0])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            labels([self.row(forbidden_capabilities=["audio"])], ["audio"])

    def test_nonexecuting_intent_gates_required_but_keeps_prohibitions(self):
        caps = np.array([[[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]])
        intent = np.array([[0.01, 0.97, 0.01, 0.01]])
        state, status = decode(caps, intent, {"required": 0.5, "forbidden": 0.5})
        self.assertEqual(state.tolist(), [[0, 2]])
        self.assertEqual(status.tolist(), [1])

    def test_explicit_denial_violation_is_measured(self):
        result = metrics(np.array([[1, 0]]), np.array([0]), np.array([[2, 0]]), np.array([0]))
        self.assertEqual(result["explicit_prohibition_violation_samples"], 1)
        self.assertEqual(result["overrequested_samples"], 1)
        self.assertEqual(result["joint_states_and_intent_exact"], 0)

    def test_capability_balancing_gives_rare_states_equal_loss_mass(self):
        targets = torch.tensor([[0]] * 8 + [[1], [2]])
        weights, counts = balanced_weights(targets, 3, power=1.0)
        mass = weights * counts
        torch.testing.assert_close(mass, mass.mean().expand_as(mass))

    def test_calibration_does_not_reward_always_empty_required_predictions(self):
        empty = {"required": {"f1": 0.0}, "forbidden": {"f1": 0.12}, "intent_macro_f1": 0.6}
        learning = {"required": {"f1": 0.08}, "forbidden": {"f1": 0.10}, "intent_macro_f1": 0.6}
        self.assertGreater(selection_score(learning), selection_score(empty))

    def test_long_input_is_rejected_without_cutting_negation(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = self.tokenizer(Path(temporary))
            with self.assertRaisesRegex(ValueError, "silent truncation"):
                encode(tokenizer, [self.row(text="录" * 63 + "不要录音")], 64)
            self.assertEqual(encode(tokenizer, [self.row()], 64)["input_ids"].shape, (1, 64))

    def test_both_heads_and_encoder_receive_finite_gradients(self):
        config = BertConfig(vocab_size=9, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                            intermediate_size=24, max_position_embeddings=64)
        model = PermissionIntentModel(config, 3)
        targets = torch.tensor([[0, 1, 2], [1, 2, 0]])
        intents = torch.tensor([0, 1])
        cap_weights, _ = balanced_weights(targets, 3)
        intent_weights, _ = balanced_weights(intents, 4)
        ids = torch.tensor([[2, 5, 3, 0], [2, 6, 3, 0]])
        cap, status = model(ids, (ids != 0).long())
        loss = objective(cap, status, targets, intents, cap_weights, intent_weights)
        loss.backward()
        for parameter in (model.capability_classifier.weight, model.intent_classifier.weight,
                          model.encoder.embeddings.word_embeddings.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_checkpoint_roundtrip_preserves_heads_and_label_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            tokenizer = self.tokenizer(path)
            config = BertConfig(vocab_size=9, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                                intermediate_size=24, max_position_embeddings=64)
            model = PermissionIntentModel(config, 3).eval()
            protocol = {"capability_ids": ["audio", "camera", "location"], "states": STATES, "intent_statuses": INTENTS}
            save_checkpoint(model, tokenizer, path / "checkpoint", protocol)
            restored, _, restored_protocol = load_checkpoint(path / "checkpoint")
            ids = torch.tensor([[2, 5, 3]])
            with torch.inference_mode():
                for before, after in zip(model(ids, torch.ones_like(ids)), restored(ids, torch.ones_like(ids))):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(restored_protocol["capability_ids"], protocol["capability_ids"])

    def test_run_split_covers_statuses_and_never_moves_test(self):
        import json
        data = ROOT / "ml/data/permission_intent_v2/expansion"
        catalog = json.loads((data.parent / "capability_catalog.json").read_text())
        rows, meta = prepare_data(data, 20261118, [c["id"] for c in catalog["capabilities"]])
        self.assertEqual({k: len(v) for k, v in rows.items()}, {"train": 705, "validation": 180, "test": 295})
        self.assertEqual({r["intent_status"] for r in rows["validation"]}, set(INTENTS))
        self.assertEqual(len(meta["validation_augmentation"]), 2)
        self.assertTrue(all("context_required" not in r["tags"] for selected in rows.values() for r in selected))
        families = {name: {r["family_id"] for r in selected} for name, selected in rows.items()}
        self.assertFalse(families["train"] & families["validation"])
        self.assertFalse(families["train"] & families["test"])


if __name__ == "__main__":
    unittest.main()
