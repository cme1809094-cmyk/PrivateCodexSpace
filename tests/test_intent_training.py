"""Task-semantic and artifact guards for the new TinyBERT trainer."""
from pathlib import Path
import copy
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
from intent_contrast import packed_group_batches, pair_margin_loss, calibrate_policy
from evaluate_intent_contrast import grouped_result
from intent_optimization import focal_objective, trim_padding
from distill_permission_intent import correct_teacher_kl, load_training_cache, clarification_specialist_kl
from refit_permission_heads import cap_linear, fold


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

    def test_label_attention_is_masked_trainable_and_reloadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            tokenizer = self.tokenizer(path)
            config = BertConfig(vocab_size=9, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                                intermediate_size=24, max_position_embeddings=64,
                                permission_intent_head="label_attention", cls_token_id=2, sep_token_id=3)
            model = PermissionIntentModel(config, 3)
            model.initialize_label_queries(tokenizer, ["录音", "不要录音", "录"])
            ids = torch.tensor([[2, 5, 6, 3, 0]])
            cap, intent = model(ids, (ids != 0).long())
            self.assertEqual(cap.shape, (1, 3, 3))
            (cap.square().mean() + intent.square().mean()).backward()
            self.assertTrue(torch.isfinite(model.label_queries.grad).all())
            self.assertGreater(float(model.label_queries.grad.abs().sum()), 0)
            model.eval()
            protocol = {"capability_ids": ["audio", "camera", "location"], "states": STATES, "intent_statuses": INTENTS}
            save_checkpoint(model, tokenizer, path / "checkpoint", protocol)
            restored, _, _ = load_checkpoint(path / "checkpoint")
            self.assertEqual(restored.head_type, "label_attention")
            with torch.inference_mode():
                for before, after in zip(model(ids, (ids != 0).long()), restored(ids, (ids != 0).long())):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_opposite_roles_stay_together_and_receive_opposite_margin_gradients(self):
        rows = [{"id": str(i), "family_id": "g_" + role, "scenario_group_id": "g", "contrast_role": role}
                for i, role in enumerate(["a", "a", "b", "b"])]
        batches = list(packed_group_batches(rows, 4, torch.Generator().manual_seed(1)))
        self.assertEqual(batches[0].tolist(), [0, 1, 2, 3])
        caps = torch.zeros((4, 1, 3), requires_grad=True)
        statuses = torch.zeros((4, 4), requires_grad=True)
        target = torch.tensor([[1], [1], [2], [2]])
        loss = pair_margin_loss(caps, statuses, target, torch.zeros(4, dtype=torch.long), rows)
        loss.backward()
        self.assertLess(float(caps.grad[0, 0, 1]), 0)
        self.assertGreater(float(caps.grad[0, 0, 2]), 0)
        self.assertLess(float(caps.grad[2, 0, 2]), 0)

    def test_policy_calibration_cannot_win_by_refusing_everything(self):
        caps = np.array([[[0.05, 0.9, 0.05]], [[0.05, 0.9, 0.05]]])
        probs = np.array([[0.8, 0.1, 0.05, 0.05], [0.8, 0.1, 0.05, 0.05]])
        rows = [{"family_id": "a"}, {"family_id": "b"}]
        policy, result = calibrate_policy(caps, probs, np.ones((2, 1), dtype=int), np.zeros(2, dtype=int),
                                          rows, {"required": 0.6, "forbidden": 0.6})
        self.assertEqual(result["accepted_samples"], 2)
        self.assertEqual(result["required"]["recall"], 1)

    def test_weak_policy_fallback_keeps_coverage_visible(self):
        caps = np.array([[[0.05, 0.9, 0.05], [0.9, 0.05, 0.05]]])
        probs = np.array([[0.8, 0.1, 0.05, 0.05]])
        policy, result = calibrate_policy(caps, probs, np.array([[0, 1]]), np.array([0]),
                                          [{"family_id": "a"}], {"required": 0.6, "forbidden": 0.6})
        self.assertEqual(result["required"]["recall"], 0)
        self.assertEqual(result["accepted_samples"], 1)
        self.assertEqual(result["overrequested_samples"], 1)

    def test_opposite_roles_must_both_pass_for_a_scenario_to_pass(self):
        rows = [{"family_id": "g_" + role, "scenario_group_id": "g"}
                for role in ("a", "a", "b", "b")]
        target = np.array([[1], [1], [2], [2]])
        predicted = np.array([[1], [1], [2], [1]])
        status = np.zeros(4, dtype=int)
        result = grouped_result(predicted, status, target, status, rows)
        self.assertEqual(result["joint_states_and_intent_exact"], 3)
        self.assertEqual(result["families_all_paraphrases_correct"], 1)
        self.assertEqual(result["scenario_groups_all_roles_and_paraphrases_correct"], 0)

    def test_evidence_gate_preserves_old_head_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            tokenizer = self.tokenizer(path)
            config = BertConfig(vocab_size=9, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                                intermediate_size=24, max_position_embeddings=64, cls_token_id=2, sep_token_id=3)
            old = PermissionIntentModel(config, 3).eval()
            gated_config = copy.deepcopy(config)
            gated_config.permission_intent_head = "pooled_label_attention"
            gated = PermissionIntentModel(gated_config, 3).eval()
            missing, unexpected = gated.load_state_dict(old.state_dict(), strict=False)
            self.assertEqual(set(missing), {"label_queries", "label_gates"})
            self.assertFalse(unexpected)
            ids = torch.tensor([[2, 5, 6, 3]])
            with torch.no_grad():
                gated.label_gates.fill_(-100)
                for left, right in zip(old(ids, torch.ones_like(ids)), gated(ids, torch.ones_like(ids))):
                    torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
            protocol = {"capability_ids": ["audio", "camera", "location"], "states": STATES, "intent_statuses": INTENTS}
            save_checkpoint(gated, tokenizer, path / "checkpoint", protocol)
            restored, _, _ = load_checkpoint(path / "checkpoint")
            with torch.inference_mode():
                for left, right in zip(gated(ids, torch.ones_like(ids)), restored(ids, torch.ones_like(ids))):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_evidence_ignores_padding_and_trains_queries(self):
        config = BertConfig(vocab_size=9, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                            intermediate_size=24, max_position_embeddings=64, cls_token_id=2, sep_token_id=3,
                            permission_intent_head="pooled_label_attention")
        model = PermissionIntentModel(config, 3).eval()
        ids = torch.tensor([[2, 5, 6, 7, 3, 0]])
        mask = (ids != 0).long()
        changed = ids.clone()
        changed[0, -1] = 8
        with torch.inference_mode():
            for a, b in zip(model(ids, mask), model(changed, mask)):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
            for a, b in zip(model(ids, mask), model(ids[:, :-1], mask[:, :-1])):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        model.train()
        cap, status = model(ids, mask)
        (cap.square().mean() + status.square().mean()).backward()
        for parameter in (model.label_queries, model.label_gates, model.encoder.embeddings.word_embeddings.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_focal_disabled_matches_existing_objective(self):
        cap = torch.randn(3, 2, 3, requires_grad=True)
        status = torch.randn(3, 4, requires_grad=True)
        targets = torch.tensor([[0, 1], [1, 2], [2, 0]])
        statuses = torch.tensor([0, 1, 3])
        cw, _ = balanced_weights(targets, 3)
        iw, _ = balanced_weights(statuses, 4)
        old = objective(cap, status, targets, statuses, cw, iw)
        new = focal_objective(cap, status, targets, statuses, cw, iw, gamma=0, smoothing=0)
        torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_padding_trim_keeps_final_content_and_separator(self):
        batch = {"input_ids": torch.tensor([[2, 5, 3, 0, 0, 0], [2, 5, 6, 7, 3, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])}
        trimmed = trim_padding(batch)
        self.assertEqual(trimmed["input_ids"].shape, (2, 5))
        self.assertEqual(trimmed["input_ids"][1].tolist(), [2, 5, 6, 7, 3])
        self.assertEqual(trimmed["attention_mask"].sum().item(), batch["attention_mask"].sum().item())

    def test_distillation_ignores_incorrect_teacher_targets(self):
        student = torch.zeros((2, 3), requires_grad=True)
        teacher = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        loss = correct_teacher_kl(student, teacher, torch.tensor([1, 1]), torch.ones(2))
        loss.backward()
        torch.testing.assert_close(student.grad[0], torch.zeros(3), rtol=0, atol=0)
        self.assertLess(float(student.grad[1, 1]), 0)

    def test_distillation_cache_must_match_training_ids_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.npz"
            np.savez_compressed(path, ids=np.asarray(["train_a", "train_b"]),
                                capability_logits=np.zeros((2, 3, 3)), intent_logits=np.zeros((2, 4)),
                                pooled_features=np.ones((2, 8), dtype=np.float16))
            loaded = load_training_cache(path, ["train_a", "train_b"])
            self.assertEqual(loaded["pooled_features"].dtype, torch.float32)
            for ids in (["validation_a", "validation_b"], ["train_b", "train_a"]):
                with self.assertRaisesRegex(ValueError, "IDs/order"):
                    load_training_cache(path, ids)

    def test_head_normalization_fold_preserves_class_probabilities(self):
        for conditional in (False, True):
            features = torch.randn(7, 3, 5) if conditional else torch.randn(7, 5)
            mean = torch.randn(3, 5) if conditional else torch.randn(5)
            scale = (torch.rand(3, 5) if conditional else torch.rand(5)) + 0.2
            weight, bias = torch.randn(3, 3, 5), torch.randn(3, 3)
            before = cap_linear((features - mean) / scale, weight, bias).softmax(-1)
            after_weight, after_bias = fold(weight, bias, mean, scale)
            after = cap_linear(features, after_weight, after_bias).softmax(-1)
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)

    def test_clarification_specialist_does_not_copy_other_intent_classes(self):
        student = torch.zeros((3, 4), requires_grad=True)
        teacher = torch.tensor([[0., 10., 0., 0.], [0., 10., 0., 0.], [0., 0., 10., 0.]])
        loss = clarification_specialist_kl(student, teacher, torch.tensor([0, 1, 3]), torch.ones(3))
        loss.backward()
        torch.testing.assert_close(student.grad[0], torch.zeros(4), rtol=0, atol=0)
        self.assertLess(float(student.grad[1, 1]), 0)
        torch.testing.assert_close(student.grad[2, 0], student.grad[2, 2], rtol=0, atol=0)
        torch.testing.assert_close(student.grad[2, 2], student.grad[2, 3], rtol=0, atol=0)

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
