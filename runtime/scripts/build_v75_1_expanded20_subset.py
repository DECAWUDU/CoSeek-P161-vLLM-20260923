#!/usr/bin/env python3
"""Build the fixed V75.1 expanded validation subset from the first-50 set."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path("/home/zjw/workspace/coseek1_v14base_simplified_planner_20260707")
SOURCE = ROOT / "../ReKV/data/mlvu/dev_first50q_mc.json"
OUTPUT = (
    ROOT
    / "experiments/v75_resilient_multiwindow_verify_runtime/subsets"
    / "v75_1_expanded20_mc.json"
)
RESUME_OUTPUT = OUTPUT.with_name("v75_1_expanded20_resume17_mc.json")
COUNT_RETRY_OUTPUT = OUTPUT.with_name("v75_1_expanded20_count6_retry_mc.json")
TAIL_OUTPUT = OUTPUT.with_name("v75_1_expanded20_tail7_mc.json")
ENTV11_RETRY_OUTPUT = OUTPUT.with_name("v75_1_expanded20_entv11_retry_mc.json")
TAIL4_OUTPUT = OUTPUT.with_name("v75_1_expanded20_tail4_mc.json")
REMAINING23_OUTPUT = OUTPUT.with_name("v75_1_first50_remaining23_mc.json")
MOVIE16_OUTPUT = OUTPUT.with_name("v75_1_first50_movie10116_4_mc.json")
REMAINING19_OUTPUT = OUTPUT.with_name("v75_1_first50_remaining19_mc.json")
COUNT170_RETRY_OUTPUT = OUTPUT.with_name("v75_1_first50_count170_retry_mc.json")
REMAINING17_AFTER_COUNT_OUTPUT = OUTPUT.with_name(
    "v75_1_first50_remaining17_after_count170_mc.json"
)

# Disjoint from the V75.1 diverse-10 and the two count optimization cases.
SELECTED_FLAT_INDICES = {
    0,
    1,
    2,
    4,
    5,
    7,
    8,
    14,
    15,
    17,
    20,
    23,
    24,
    26,
    28,
    29,
    37,
    38,
    42,
    48,
}

# Questions covered by the paired diverse-10 and the completed expanded-17.
VALIDATED_27_FLAT_INDICES = {
    4,
    5,
    7,
    8,
    9,
    11,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    36,
    37,
    38,
    42,
    46,
    48,
}


def main() -> None:
    source = json.loads(SOURCE.resolve().read_text())
    selected_by_video: dict[str, dict] = {}
    flat_index = 0

    for video in source:
        for question in video.get("conversations", []):
            if flat_index in SELECTED_FLAT_INDICES:
                video_id = video["video_id"]
                if video_id not in selected_by_video:
                    selected_video = copy.deepcopy(video)
                    selected_video["conversations"] = []
                    selected_by_video[video_id] = selected_video

                selected_question = copy.deepcopy(question)
                selected_question["_v75_1_eval_flat_index"] = flat_index
                selected_by_video[video_id]["conversations"].append(selected_question)
            flat_index += 1

    selected = list(selected_by_video.values())
    selected_count = sum(len(v["conversations"]) for v in selected)
    if selected_count != len(SELECTED_FLAT_INDICES):
        raise RuntimeError(
            f"Expected {len(SELECTED_FLAT_INDICES)} questions, got {selected_count}"
        )

    OUTPUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {selected_count} questions across {len(selected)} videos to {OUTPUT}")

    resume = [video for video in selected if video["video_id"] != "movie101_16"]
    resume_count = sum(len(video["conversations"]) for video in resume)
    if resume_count != 17:
        raise RuntimeError(f"Expected 17 resume questions, got {resume_count}")
    RESUME_OUTPUT.write_text(json.dumps(resume, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Wrote {resume_count} questions across {len(resume)} videos to {RESUME_OUTPUT}"
    )

    count_retry = []
    tail = []
    for video in resume:
        retry_questions = [
            question
            for question in video["conversations"]
            if question["_v75_1_eval_flat_index"] == 24
        ]
        tail_questions = [
            question
            for question in video["conversations"]
            if question["_v75_1_eval_flat_index"] in {26, 28, 29, 37, 38, 42, 48}
        ]
        if retry_questions:
            retry_video = copy.deepcopy(video)
            retry_video["conversations"] = retry_questions
            count_retry.append(retry_video)
        if tail_questions:
            tail_video = copy.deepcopy(video)
            tail_video["conversations"] = tail_questions
            tail.append(tail_video)

    if sum(len(video["conversations"]) for video in count_retry) != 1:
        raise RuntimeError("Expected one count_6 retry question")
    if sum(len(video["conversations"]) for video in tail) != 7:
        raise RuntimeError("Expected seven tail questions")
    COUNT_RETRY_OUTPUT.write_text(
        json.dumps(count_retry, ensure_ascii=False, indent=2) + "\n"
    )
    TAIL_OUTPUT.write_text(json.dumps(tail, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote count retry subset to {COUNT_RETRY_OUTPUT}")
    print(f"Wrote seven-question tail subset to {TAIL_OUTPUT}")

    entv11_retry = []
    tail4 = []
    for video in tail:
        retry_questions = [
            question
            for question in video["conversations"]
            if question["_v75_1_eval_flat_index"] == 29
        ]
        tail_questions = [
            question
            for question in video["conversations"]
            if question["_v75_1_eval_flat_index"] in {37, 38, 42, 48}
        ]
        if retry_questions:
            retry_video = copy.deepcopy(video)
            retry_video["conversations"] = retry_questions
            entv11_retry.append(retry_video)
        if tail_questions:
            tail_video = copy.deepcopy(video)
            tail_video["conversations"] = tail_questions
            tail4.append(tail_video)

    if sum(len(video["conversations"]) for video in entv11_retry) != 1:
        raise RuntimeError("Expected one en_tv_11 retry question")
    if sum(len(video["conversations"]) for video in tail4) != 4:
        raise RuntimeError("Expected four final tail questions")
    ENTV11_RETRY_OUTPUT.write_text(
        json.dumps(entv11_retry, ensure_ascii=False, indent=2) + "\n"
    )
    TAIL4_OUTPUT.write_text(json.dumps(tail4, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote en_tv_11 retry subset to {ENTV11_RETRY_OUTPUT}")
    print(f"Wrote four-question final tail subset to {TAIL4_OUTPUT}")

    remaining23_by_video: dict[str, dict] = {}
    flat_index = 0
    for video in source:
        for question in video.get("conversations", []):
            if flat_index not in VALIDATED_27_FLAT_INDICES:
                video_id = video["video_id"]
                if video_id not in remaining23_by_video:
                    selected_video = copy.deepcopy(video)
                    selected_video["conversations"] = []
                    remaining23_by_video[video_id] = selected_video
                selected_question = copy.deepcopy(question)
                selected_question["_v75_1_eval_flat_index"] = flat_index
                remaining23_by_video[video_id]["conversations"].append(selected_question)
            flat_index += 1

    remaining23 = list(remaining23_by_video.values())
    remaining23_count = sum(len(video["conversations"]) for video in remaining23)
    if remaining23_count != 23:
        raise RuntimeError(f"Expected 23 remaining questions, got {remaining23_count}")
    REMAINING23_OUTPUT.write_text(
        json.dumps(remaining23, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote first-50 remaining 23 subset to {REMAINING23_OUTPUT}")

    movie16 = [video for video in remaining23 if video["video_id"] == "movie101_16"]
    remaining19 = [video for video in remaining23 if video["video_id"] != "movie101_16"]
    if sum(len(video["conversations"]) for video in movie16) != 4:
        raise RuntimeError("Expected four movie101_16 questions")
    if sum(len(video["conversations"]) for video in remaining19) != 19:
        raise RuntimeError("Expected nineteen questions after excluding movie101_16")
    MOVIE16_OUTPUT.write_text(json.dumps(movie16, ensure_ascii=False, indent=2) + "\n")
    REMAINING19_OUTPUT.write_text(
        json.dumps(remaining19, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote movie101_16 four-question subset to {MOVIE16_OUTPUT}")
    print(f"Wrote first-50 remaining 19 subset to {REMAINING19_OUTPUT}")

    count170_retry = []
    remaining17_after_count = []
    remaining17_indices = {12, 18, 21, 30, 31, 32, 33, 34, 35, 39, 40, 41, 43, 44, 45, 47, 49}
    for video in remaining19:
        retry_questions = [
            question
            for question in video["conversations"]
            if question["_v75_1_eval_flat_index"] == 10
        ]
        remaining_questions = [
            question
            for question in video["conversations"]
            if question["_v75_1_eval_flat_index"] in remaining17_indices
        ]
        if retry_questions:
            retry_video = copy.deepcopy(video)
            retry_video["conversations"] = retry_questions
            count170_retry.append(retry_video)
        if remaining_questions:
            remaining_video = copy.deepcopy(video)
            remaining_video["conversations"] = remaining_questions
            remaining17_after_count.append(remaining_video)

    if sum(len(video["conversations"]) for video in count170_retry) != 1:
        raise RuntimeError("Expected one count_170 retry question")
    if sum(len(video["conversations"]) for video in remaining17_after_count) != 17:
        raise RuntimeError("Expected seventeen questions after count_170")
    COUNT170_RETRY_OUTPUT.write_text(
        json.dumps(count170_retry, ensure_ascii=False, indent=2) + "\n"
    )
    REMAINING17_AFTER_COUNT_OUTPUT.write_text(
        json.dumps(remaining17_after_count, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote count_170 retry subset to {COUNT170_RETRY_OUTPUT}")
    print(
        "Wrote first-50 seventeen-question continuation subset to "
        f"{REMAINING17_AFTER_COUNT_OUTPUT}"
    )


if __name__ == "__main__":
    main()
