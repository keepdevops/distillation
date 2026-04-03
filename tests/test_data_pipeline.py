#!/usr/bin/env python3
"""Smoke test for data_pipeline schema detection and prompt extraction.

Tests _detect_schema + _extract_pair with synthetic examples matching the
field structure of each supported dataset — no network required.

Usage:
    pixi run python tests/test_data_pipeline.py
    # or: python -m pytest tests/test_data_pipeline.py
"""

from __future__ import annotations

import sys

from distill.data_pipeline import (
    _detect_schema,
    _extract_pair,
    format_prompt_full,
    format_prompt_only,
)

CASES = [
    (
        "alpaca                  tatsu-lab/alpaca",
        {
            "instruction": "Explain gravity.",
            "input": "For a 10-year-old.",
            "output": "Gravity pulls things toward each other.",
        },
        "alpaca", "Explain gravity", "Gravity pulls",
    ),
    (
        "alpaca (no input)       bigcode/self-oss-instruct-sc2-exec-filter-50k",
        {
            "instruction": "Write a Python function to reverse a string.",
            "response": "def reverse(s):\n    return s[::-1]",
        },
        "alpaca", "Write a Python", "def reverse",
    ),
    (
        "sharegpt                teknium/OpenHermes-2.5",
        {
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human",  "value": "What is the capital of France?"},
                {"from": "gpt",    "value": "The capital of France is Paris."},
            ]
        },
        "sharegpt", "What is the capital", "Paris",
    ),
    (
        "messages/ChatML         HuggingFaceH4/no_robots",
        {
            "prompt": "Write a haiku about rain.",
            "messages": [
                {"role": "user",      "content": "Write a haiku about rain."},
                {"role": "assistant", "content": "Rain taps on the glass,\nWhispering to those inside."},
            ],
            "category": "Creative Writing",
        },
        "messages", "Write a haiku", "Rain taps",
    ),
    (
        "dpo                     argilla/distilabel-capybara-dpo-7k-binarized",
        {
            "instruction": "Summarise the French Revolution in one sentence.",
            "chosen": [
                {"role": "user",      "content": "Summarise the French Revolution in one sentence."},
                {"role": "assistant", "content": "The French Revolution overthrew the monarchy."},
            ],
            "rejected": [
                {"role": "user",      "content": "Summarise the French Revolution in one sentence."},
                {"role": "assistant", "content": "France had a revolution."},
            ],
        },
        "dpo", "French Revolution", "overthrew",
    ),
    (
        "guanaco                 mlabonne/guanaco-llama2-1k",
        {
            "text": "### Human: What is 2+2?\n### Assistant: 2+2 equals 4.",
        },
        "guanaco", "What is 2+2", "equals 4",
    ),
]


def main() -> None:
    pass_ = fail = 0
    for label, example, exp_schema, prompt_snip, resp_snip in CASES:
        errors = []

        schema = _detect_schema(example)
        if schema != exp_schema:
            errors.append(f"schema: got '{schema}', expected '{exp_schema}'")

        prompt, response = _extract_pair(example)
        if prompt_snip not in prompt:
            errors.append(f"prompt missing '{prompt_snip}': {prompt!r}")
        if resp_snip not in response:
            errors.append(f"response missing '{resp_snip}': {response!r}")

        full = format_prompt_full(example)
        only = format_prompt_only(example)

        if "### Response:" not in full:
            errors.append("format_prompt_full missing '### Response:'")
        if "### Response:" not in only:
            errors.append("format_prompt_only missing '### Response:'")
        if response and response not in full:
            errors.append("format_prompt_full missing response text")
        if response and response in only:
            errors.append("format_prompt_only should NOT contain response")

        if errors:
            print(f"FAIL  {label}")
            for e in errors:
                print(f"        {e}")
            fail += 1
        else:
            print(f"PASS  {label}")
            pass_ += 1

    print(f"\n{pass_ + fail} cases — {pass_} passed, {fail} failed")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
