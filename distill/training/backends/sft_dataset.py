"""SFT dataset class and teacher label generation."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_labels(teacher_model, tokenizer, prompts: list[str],
                    max_new_tokens: int, device, batch_size: int = 8) -> list[str]:
    """Generate teacher responses in batches. Much faster than one-at-a-time."""
    responses = []
    logger.info("Generating teacher labels for %d prompts (batch_size=%d)...", len(prompts), batch_size)
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start: batch_start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        out = teacher_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        for seq in out:
            response = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            responses.append(response)
        done = min(batch_start + batch_size, len(prompts))
        if done % 100 == 0 or done == len(prompts):
            logger.info("  Generated %d/%d labels", done, len(prompts))
    return responses


class SFTDataset(torch.utils.data.Dataset):
    """
    Tokenizes prompt+response with prompt tokens masked to -100 in labels.
    The model only learns to predict response tokens; prompt and padding
    positions are excluded from the cross-entropy loss.
    """
    IGNORE = -100

    def __init__(self, prompts: list[str], responses: list[str], tokenizer, max_length: int):
        input_ids_list, attn_list, label_list = [], [], []
        for prompt, response in zip(prompts, responses):
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))

            full = tokenizer(
                prompt + " " + response,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            ids = full["input_ids"][0]
            mask = full["attention_mask"][0]

            labels = ids.clone()
            labels[:prompt_len] = self.IGNORE
            labels[mask == 0] = self.IGNORE

            input_ids_list.append(ids)
            attn_list.append(mask)
            label_list.append(labels)

        self.input_ids = torch.stack(input_ids_list)
        self.attention_mask = torch.stack(attn_list)
        self.labels = torch.stack(label_list)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
