from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from videoseek.core.memory import extract_v10_payload
from videoseek.core.p130_runtime import _count_candidate, init_p130_state, next_recovery_spec
from videoseek.tools.frame_verify import (
    _apply_p130_local_receipt_contract, _merge_anchor_audit_into_candidate_assessments,
    _normalize_anchor_assessments, _normalize_candidate_assessments,
    _p130_normalize_typed_event_binding, execute_frame_verify,
)
from test_p130_frame_verify_contract import ORDER_QUESTION, COUNT_QUESTION, _VideoReader, _Batch


class P132BindingTests(unittest.TestCase):
    def row(self, event_id='2'):
        return dict(candidate_id='candidate', target_match='matched', event_match='direct',
                    observed_fact='People walk inside a covered bridge.',
                    candidate_event_ids=[event_id, '4', '3'],
                    best_timestamp_s=82.7, event_span=[81.0, 85.4])

    def normalize(self, row, assigned='2'):
        _p130_normalize_typed_event_binding(row, required_event_ids={'1','2','3','4'},
                                          assigned_target=assigned)
        return row

    def test_context_ids_do_not_erase_assigned_direct_fact(self):
        for assigned in ('1', '2', '3', '4'):
            row = self.row(assigned)
            self.normalize(row, assigned)
            self.normalize(row, assigned)
            self.assertEqual(row['matched_event_ids'], [assigned])
            self.assertFalse(row['event_id_conflict'])
        row = self.row()
        row['target_event_id'] = '2'
        self.normalize(row, '')
        self.assertEqual(row['matched_event_ids'], [])

    def test_conflicts_invalid_ids_and_unsupported_facts_stay_unbound(self):
        changes = [dict(target_event_id='1'), dict(target_event_ids=['1','2']),
                   dict(candidate_event_ids=['2','99']), dict(candidate_event_ids=['2','wrong']),
                   dict(event_match='context_only'), dict(target_match='mismatch'),
                   dict(observed_fact=''), dict(event_span=None, best_timestamp_s=None),
                   dict(event_span=[float('nan'), 85.4], best_timestamp_s=-1),
                   dict(candidate_event_ids=['3','4'])]
        for change in changes:
            with self.subTest(change=change):
                row = {**self.row(), **change}
                self.normalize(row)
                self.normalize(row)
                self.assertEqual(row['matched_event_ids'], [])

    def test_all_normalization_and_anchor_merge_layers_preserve_binding(self):
        candidates = [dict(candidate_id='candidate', target_event_id='2', t_range=[79.4,97.4])]
        assigned = {'candidate':'2'}
        kwargs = dict(event_binding_enabled=True, required_event_ids={'1','2','3','4'},
                      assigned_target_by_candidate=assigned)
        rows, complete = _normalize_candidate_assessments(
            [self.row()], candidates=candidates, allowed_by_candidate={'candidate':[82.7]}, **kwargs)
        self.assertTrue(complete)
        anchor = {**self.row(), 'timestamp_s':82.7}
        anchors, _ = _normalize_anchor_assessments(
            [anchor], allowed_by_candidate={'candidate':[82.7]}, **kwargs)
        _merge_anchor_audit_into_candidate_assessments(rows, anchors, **kwargs)
        payload = dict(candidate_assessments=rows, anchor_assessments=anchors)
        context = dict(active=True, mode='order', p131_active=True, p132_active=True,
                       required_event_ids=['1','2','3','4'], assigned_target_by_candidate=assigned)
        _apply_p130_local_receipt_contract(payload, context=context)
        _apply_p130_local_receipt_contract(payload, context=context)
        # Grouped packets normalize candidate receipts a second time.
        again, _ = _normalize_candidate_assessments(rows, candidates=candidates, **kwargs)
        self.assertEqual(again[0]['matched_event_ids'], ['2'])
        self.assertEqual(payload['anchor_assessments'][0]['matched_event_ids'], ['2'])
        self.assertFalse(payload['decision_sufficient'])

    def test_off_grid_anchor_is_not_relabelled_as_observed(self):
        rows, complete = _normalize_anchor_assessments(
            [{**self.row(), 'timestamp_s':90.0}], allowed_by_candidate={'candidate':[82.7]},
            event_binding_enabled=True, required_event_ids={'1','2','3','4'},
            assigned_target_by_candidate={'candidate':'2'})
        self.assertEqual(rows, [])
        self.assertFalse(complete)

    def test_multiwindow_executor_keeps_candidate_binding(self):
        raw = dict(timestamp_observations=[dict(candidate_id='candidate', timestamp_s=82.7,
                   description='People walking on a bridge.')], candidate_assessments=[self.row()],
                   target_match='matched', detail_sufficient=True)
        with patch('videoseek.tools.frame_verify.scene_aware_timestamps', return_value=[81.,82.7,85.4]), \
             patch('videoseek.tools.frame_verify.observe_content', return_value=(json.dumps(raw),'api')):
            payload = extract_v10_payload(execute_frame_verify(
                dict(p130_minimal_global_fsm_enabled=True, p131_minimal_global_repairs_enabled=True,
                     p132_weak_lead_visible_core_enabled=True, localize_inline_verify_max_frames=3,
                     multiwindow_verify_recovery_enabled=False),
                dict(question=ORDER_QUESTION, query='Inspect the assigned event.', vr=_VideoReader(),
                     video_path='/tmp/test.mp4', duration=100., mode='timeline',
                     candidate_windows=[dict(candidate_id='candidate', target_event_id='2',
                                             t_range=[79.4,97.4], timestamp_anchors=[82.7])])) )
        self.assertEqual(payload['candidate_assessments'][0]['matched_event_ids'], ['2'])
        self.assertEqual(payload['timestamp_observations'][0]['matched_event_ids'], ['2'])


class P132SamplingTests(unittest.TestCase):
    def test_residual_dispatch_has_no_invented_anchors_only_when_enabled(self):
        for enabled in (False, True):
            memory = {}
            init_p130_state(memory, question=COUNT_QUESTION, duration=490., p131_enabled=True)
            spec = next_recovery_spec(memory, question=COUNT_QUESTION, overview_output='',
                duration=490., attempt_index=0, p131_schedule=True,
                p132_weak_lead_visible_core=enabled)
            self.assertEqual(spec['reason'], 'residual_count_cursor')
            self.assertTrue(all(bool(row['timestamp_anchors']) != enabled
                                for row in spec['candidate_windows']))

    def test_actual_frame_indices_keep_scene_sampling_for_residual(self):
        class Reader(_VideoReader):
            def __init__(self): self.seen=[]
            def __len__(self): return 15000
            def get_batch(self, indices):
                self.seen.extend(indices.tolist())
                return _Batch(np.zeros((len(indices),24,32,3), dtype=np.uint8))
        sampled = {}
        for enabled in (False, True):
            vr = Reader()
            candidate = _count_candidate('residual', [405.33,433.573], rank=1,
                                         p132_weak_lead_visible_core=enabled)
            raw = json.dumps(dict(timestamp_observations=[], candidate_assessments=[],
                                  target_match='not_visible', detail_sufficient=False))
            with patch('videoseek.tools.frame_verify.scene_aware_timestamps',
                       return_value=[405.33,412.212,424.825]), \
                 patch('videoseek.tools.frame_verify.observe_content', return_value=(raw,'api')):
                execute_frame_verify(dict(p130_minimal_global_fsm_enabled=True,
                    p131_minimal_global_repairs_enabled=True,
                    p132_weak_lead_visible_core_enabled=enabled, localize_inline_verify_max_frames=3,
                    multiwindow_verify_recovery_enabled=False),
                    dict(question=COUNT_QUESTION, query='Scan residual interval.', vr=vr,
                         video_path='/tmp/test.mp4', duration=500., mode='timeline',
                         candidate_windows=[candidate]))
            sampled[enabled] = vr.seen
        self.assertIn(round(424.825*30), sampled[True])
        self.assertNotIn(round(424.825*30), sampled[False])


if __name__ == '__main__':
    unittest.main()
