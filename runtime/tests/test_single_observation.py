import copy
import json

import numpy as np
import pytest

from videoseek.core import p130_runtime as runtime
from videoseek.core import minimal_global_fsm as fsm
from videoseek.core.memory import extract_v10_payload
from videoseek.tools import frame_verify as verifier
from videoseek.tools.v10_format import format_v10_observation


COUNT = "Throughout this video, what is the total count of occurrences for the scene featuring the 'opening a box' action\n(A) 0\n(B) 1\n(C) 2\n(D) 3"
ORDER = "Arrange the following events in chronological order: (1) a person opens a box; (2) a person closes a door.\n(A) 1->2\n(B) 2->1"


def fixtures(question=COUNT):
    return fsm.parse_global_question(question), [
        {"frame_id": "F100", "frame_index": 100, "timestamp_s": 10., "frame_duration_s": .1, "candidate_ids": ["c"]},
        {"frame_id": "F120", "frame_index": 120, "timestamp_s": 12., "frame_duration_s": .1, "candidate_ids": ["c"]},
    ], [{"candidate_id": "c", "t_range": [9., 13.], "timestamp_anchors": [10., 12.]}]


def observation(target="target", **updates):
    return {**({"target_id": target} if target != "target" else {}), "match": "direct", "fact": "The action is visible.",
            "frame_ids": ["F100", "F120"], **updates}


def canonical(question=COUNT, **updates):
    parsed, frames, candidates = fixtures(question)
    rows, errors = runtime.parse_local_observations({"observations": [observation(**updates)]}, frames=frames, candidates=candidates, parsed=parsed)
    assert not errors
    return rows[0]


@pytest.mark.parametrize("text, expected", [("B","B"),(" b ","B"),("(B)","B"),("B.","B"),
    ("The answer is B",""),("Cannot determine",""),("A or B",""),("","")])
def test_one_strict_answer_parser(text, expected):
    assert fsm._answer_letter(text) == expected


@pytest.mark.parametrize("change", [
    {"target_event_id":"1"}, {"frame_ids":["F999"]}, {"span":[20.,22.]},
    {"span":[False,12.]}, {"match":"context_only"}, {"target_id":"missing"},
    {"frame_ids":[]}, {"fact":""}, {"span":None},
])
def test_invalid_fields_do_not_get_repaired_into_facts(change):
    parsed, frames, candidates = fixtures()
    payload={"observations":[observation(**change)]}
    before=copy.deepcopy(payload)
    rows, errors=runtime.parse_local_observations(payload,frames=frames,candidates=candidates,parsed=parsed)
    assert not rows and errors and payload == before


def test_unknown_and_negative_remain_visible_to_planner_without_increasing_count():
    memory={};runtime.init_p130_state(memory,question=COUNT,duration=100.)
    state=memory['p130_global']
    for match in ['negative','ambiguous']:
        state['evidence_receipts']=[canonical(match=match)]
        runtime.refresh_p130_snapshot(memory,question=COUNT)
        view=runtime.global_planner_state(memory,budget={})
        assert view['verified_observations'][0]['match']==match
        assert view['evidence_state']['observed_count_lower_bound']==0


def test_count_duplicates_merge_only_in_reducer_without_mutating_observations():
    row=canonical(); other=copy.deepcopy(row)
    row.update(event_key='receipt:first',receipt_ids=['first'])
    other.update(event_key='receipt:second',receipt_ids=['second'],candidate_id='renamed')
    rows=[row,other];before=copy.deepcopy(rows)
    state=fsm.reduce_count(COUNT,rows)
    assert state['observed_count_lower_bound']==1
    assert rows==before
    other['t_range']=[50.,52.];other['anchor_timestamps']=[50.,52.]
    assert fsm.reduce_count(COUNT,rows)['observed_count_lower_bound']==2


def test_missing_extent_cannot_add_order_certainty():
    one=canonical(ORDER,target='1'); one['t_range']=None
    two=canonical(ORDER,target='2')
    two['t_range']=[20.,22.];two['anchor_timestamps']=[20.,22.]
    state=fsm.reduce_order(ORDER,[one,two])
    assert state['precedence_edges']==[] and not state['decision_sufficient']
    one['t_range']=[10.,12.]
    state=fsm.reduce_order(ORDER,[one,two])
    assert state['validated_option']=='A'


def test_complete_bindings_with_no_viable_option_show_conflict_to_planner():
    memory={};runtime.init_p130_state(memory,question=ORDER,duration=100.)
    state=memory['p130_global'];state['evidence_receipts']=[canonical(ORDER,target='1'),canonical(ORDER,target='2')]
    for i,row in enumerate(state['evidence_receipts']): row['receipt_ids']=[f'r{i}']
    state['snapshot'].update(required_event_ids=['1','2'],unbound_event_ids=[],viable_options=[],decision_sufficient=False)
    view=runtime.global_planner_state(memory,budget={})
    assert not view['evidence_state']['viable_options']
    assert view['remaining_actions']['frame_verify']>0


def test_replacement_is_scoped_and_raw_observation_survives():
    memory={};runtime.init_p130_state(memory,question=ORDER,duration=100.)
    state=memory['p130_global'];one=canonical(ORDER,target='1');two=canonical(ORDER,target='2')
    one['receipt_ids']=['r1'];two['receipt_ids']=['r2'];state['evidence_receipts']=[one,two]
    incoming=canonical(ORDER,target='1',match='ambiguous');incoming['candidate_id']='review'
    payload={'parse_ok':True,'local_observations':[incoming],'sampled_frames':fixtures()[1]}
    before=copy.deepcopy(payload)
    runtime.append_p130_observation(memory,tool_name='frame_verify',parameters={'replace_receipt_ids':{'review':[{'target_id':'1','receipt_id':'r1'}]}},output=format_v10_observation(payload),action_id='review',question=ORDER,duration=100.)
    assert payload==before
    assert any(r['receipt_ids']==['r2'] for r in state['evidence_receipts'])
    assert not any('r1' in r['receipt_ids'] for r in state['evidence_receipts'])
    assert state['coverage_receipts'][-1]['sampled_timestamps']==[10.,12.]
    assert not state['snapshot']['decision_sufficient']


@pytest.mark.parametrize('proposal',['A','Cannot determine','The answer is B',''])
def test_terminal_fields_are_views_of_one_decision(proposal):
    memory={};runtime.init_p130_state(memory,question=COUNT,duration=100.)
    decision=fsm.force_answer(proposal,memory['p130_global']['snapshot'],reason='no_observation_action')
    answer=runtime.store_global_decision(memory,decision)
    state=memory['p130_global']
    assert answer==(decision['selected_option'] or '')
    assert state['evaluation_prediction']==(answer or None)
    assert state['validated_answer'] is None
    assert state['forced_prediction']==bool(answer)


class FakeVideo:
    def __len__(self): return 1000
    def get_avg_fps(self): return 10.
    def get_batch(self, indices):
        data=np.zeros((len(indices),32,48,3),dtype=np.uint8)
        class Batch:
            def asnumpy(self): return data
        return Batch()


def test_full_observer_path_uses_one_call_and_actual_frames(monkeypatch):
    calls=[]
    def observe(config, *, content, **kwargs):
        calls.append(content)
        frame_text=next(r['text'] for r in content if r['type']=='text' and r['text'].startswith('F'))
        ref,time,_=frame_text.split(' | ')
        timestamp=float(time[:-1])
        return json.dumps({'observations':[observation(frame_ids=[ref])]}),'api'
    monkeypatch.setattr(verifier,'observe_content',observe)
    parameters={'vr':FakeVideo(),'question':COUNT,'candidate_windows':fixtures()[2],'frame_count':4}
    config={'p130_minimal_global_fsm_enabled':True,'localize_inline_verify_max_frames':24}
    output=verifier.execute_frame_verify(config,parameters)
    payload=extract_v10_payload(output)
    assert len(calls)==1 and payload['parse_ok'] and not payload['contract_errors']
    assert len(payload['sampled_frames'])==4
    assert {'F100','F120'}.issubset({r['frame_id'] for r in payload['sampled_frames']})
    assert all(r['image_url']['detail']=='high' for r in calls[0] if r['type']=='image_url')
    assert not {'candidate_assessments','anchor_assessments','decision_sufficient'}.intersection(payload)
    assert payload['local_observations'][0]['frame_ids'][0] in {r['frame_id'] for r in payload['sampled_frames']}


def test_no_new_frames_does_not_make_another_observer_call(monkeypatch):
    monkeypatch.setattr(verifier,'observe_content',lambda *a,**kw: pytest.fail('unexpected API call'))
    params={'vr':FakeVideo(),'question':COUNT,'candidate_windows':fixtures()[2],'frame_count':4,
            'resample':True,'seen_frame_ids':[f'F{i}' for i in range(1000)]}
    out=verifier.execute_frame_verify({'p130_minimal_global_fsm_enabled':True},params)
    assert extract_v10_payload(out)['contract_errors']==['no_new_frames']


def test_overview_seconds_are_normalized_for_search_anchors():
    payload={"timestamp_observations":[{"timestamp_s":"33.1s"},{"timestamp_s":"50.4s"}]}
    original=copy.deepcopy(payload); memory={}
    runtime.init_p130_state(memory,question=COUNT,duration=100.)
    runtime.append_p130_observation(memory,tool_name='overview',parameters={},output=format_v10_observation(payload),action_id='ov',question=COUNT,duration=100.)
    _,params=runtime.global_planner_action(memory,{"reason":"Find target","action":{"tool":"localize_qwen","parameters":{
        "search_windows":[[30,60]],"localization_goal":"Find action"}}})
    assert params['mandatory_timestamps']==[33.1,50.4] and payload==original
    assert runtime._safe_float("not 33.1s")==None
    assert runtime._safe_float("nan")==None
