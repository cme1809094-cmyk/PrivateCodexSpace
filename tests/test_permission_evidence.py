import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))

from permission_evidence import DEFAULT_THRESHOLDS, POLICY, decode_evidence, evidence_targets
from train_intent_v6 import ranking_loss


class PermissionEvidenceTests(unittest.TestCase):
    def test_scope_routing_prevents_unconditional_execution(self):
        cap = np.array([
            [[0.9, 0.8, 0.9], [0.85, 0.1, 0.5]],  # A conflict clears every requested action.
            [[0.9, 0.8, 0.1], [0.85, 0.1, 0.5]],  # Disjoint scopes need a scoped executor.
            [[0.9, 0.8, 0.4], [0.85, 0.1, 0.5]],  # Uncertain scope waits.
            [[0.9, 0.1, 0.9], [0.1, 0.8, 0.9]],   # Different capabilities can coexist.
        ])
        intent = np.tile([0.97, 0.01, 0.01, 0.01], (4, 1))
        got = decode_evidence(cap, intent, DEFAULT_THRESHOLDS, POLICY)
        self.assertEqual(got['decision'].tolist(), ['clarify', 'review_scope', 'clarify', 'accept'])
        self.assertEqual(got['status'].tolist(), [1, 0, 0, 0])
        self.assertEqual(got['state'].tolist(), [[2, 0], [2, 0], [2, 0], [1, 2]])
        self.assertTrue(got['requested'][0, 0] and got['prohibited'][0, 0])

    def test_legacy_ambiguous_evidence_is_masked_except_known_templates(self):
        older = {'id': 'old_ambiguous', 'required_capabilities': [], 'forbidden_capabilities': ['camera'],
                 'intent_status': 'ambiguous'}
        template = {**older, 'id': 'v4_train_conflict_00_01_a1', 'split': 'train', 'ambiguity_type': 'conflict',
                    'source': {'kind': 'conflict_vs_other_action', 'composition': {'left': 'camera'}}}
        target, known = evidence_targets([older, template], ['camera'])
        self.assertEqual(target.tolist(), [[[0, 1, 0]], [[1, 1, 1]]])
        self.assertEqual(known.tolist(), [[[False, True, False]], [[True, True, True]]])

    def test_role_ranking_penalizes_sibling_request_within_its_annotated_role(self):
        logits = torch.tensor([[[0., 3., 0.], [1., 0., 0.], [9., 9., 0.]]], requires_grad=True)
        target = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]])
        known = torch.tensor([[[True, True, False], [True, True, False], [False, False, False]]])
        ranking_loss(logits, target, known).backward()
        self.assertLess(float(logits.grad[0, 0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 1, 0]), 0)
        self.assertGreater(float(logits.grad[0, 0, 1]), 0)
        self.assertLess(float(logits.grad[0, 1, 1]), 0)
        self.assertTrue(torch.equal(logits.grad[0, 2], torch.zeros(3)))


if __name__ == '__main__':
    unittest.main()
