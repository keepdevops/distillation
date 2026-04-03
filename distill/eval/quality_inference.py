"""Model-dependent inference helpers for quality evaluation."""
from __future__ import annotations

import json
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from .quality_metrics import JUDGE_PROMPT

logger = logging.getLogger(__name__)


@torch.no_grad()
def batch_generate_responses(
    model, tokenizer, prompts, max_new_tokens, temperature, device, batch_size=8
):
    """Generate responses in batches for 10-100x speedup over sequential."""
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True, padding_side="left",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            ),
        )
        for j, out in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            all_responses.append(response)
    return all_responses


@torch.no_grad()
def compute_embedding_diversity(model, tokenizer, texts, device, batch_size=16):
    """Compute semantic diversity using model embeddings."""
    from scipy.spatial.distance import pdist

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(embeddings)
    n = len(embeddings)
    sample_emb = embeddings[np.random.choice(n, size=min(100, n), replace=False)] if n > 100 else embeddings
    distances = pdist(sample_emb, metric="euclidean")
    centroid = embeddings.mean(axis=0)
    radii = np.linalg.norm(embeddings - centroid, axis=1)
    return {
        "mean_pairwise_distance": round(float(np.mean(distances)), 4),
        "std_pairwise_distance": round(float(np.std(distances)), 4),
        "coverage_radius_95": round(float(np.percentile(radii, 95)), 4),
        "embeddings": embeddings,
    }


def create_umap_visualization(embeddings, categories, output_path):
    """Create UMAP 2D projection of embeddings colored by category."""
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("UMAP not installed, skipping visualization. Install with: pip install umap-learn")
        return None

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)
    viz_data = {
        "points": [
            {"x": float(embedding_2d[i, 0]), "y": float(embedding_2d[i, 1]), "category": categories[i]}
            for i in range(len(embedding_2d))
        ],
        "category_counts": {cat: categories.count(cat) for cat in set(categories)},
    }
    with open(output_path, "w") as f:
        json.dump(viz_data, f, indent=2)
    logger.info("UMAP visualization saved to %s", output_path)
    return viz_data


@torch.no_grad()
def batch_judge_responses(judge_model, judge_tok, instructions, responses, device, batch_size=8):
    """Judge multiple responses in batches."""
    all_judgments = []
    for i in range(0, len(instructions), batch_size):
        batch_inst = instructions[i : i + batch_size]
        batch_resp = responses[i : i + batch_size]
        prompts = [
            JUDGE_PROMPT.format(instruction=inst, response=resp)
            for inst, resp in zip(batch_inst, batch_resp)
        ]
        inputs = judge_tok(prompts, return_tensors="pt", truncation=True, max_length=768, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )
        for j, out in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            judgment = judge_tok.decode(out[input_len:], skip_special_tokens=True).strip()
            all_judgments.append(judgment)
    return all_judgments


@torch.no_grad()
def compute_teacher_perplexity_on_responses(
    teacher_model, teacher_tok, prompts, responses, device, batch_size=4
):
    """Compute teacher's perplexity on student-generated responses."""
    perplexities = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_responses = responses[i : i + batch_size]
        full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
        inputs = teacher_tok(full_texts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = teacher_model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        shift_mask = inputs["attention_mask"][:, 1:].float()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        token_losses = token_losses * shift_mask
        sample_losses = token_losses.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        for loss_val in sample_losses.tolist():
            ppl = math.exp(loss_val) if loss_val < 10 else float("inf")
            perplexities.append(round(ppl, 2))
    return perplexities
