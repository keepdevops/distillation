#!/usr/bin/env python3
"""
Vanilla forward-KL knowledge distillation for classification.
Bare-metal, air-gapped. Works with BERT/DistilBERT etc.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


def parse_args():
    p = argparse.ArgumentParser(description="Forward KL distillation (classification)")
    p.add_argument("--teacher", type=str, default="bert-large-uncased")
    p.add_argument("--student", type=str, default="distilbert-base-uncased")
    p.add_argument("--dataset", type=str, default="glue")
    p.add_argument("--dataset_config", type=str, default="sst2")
    p.add_argument("--output_dir", type=str, default="./distilled-forward")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag callback for watchdog")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainer(Trainer):
    """Trainer with forward KL distillation loss."""

    def __init__(self, teacher_model, temperature, alpha, *args, **kwargs):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_outputs = model(**inputs)

        soft_teacher = F.softmax(teacher_outputs.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_outputs.logits / self.temperature, dim=-1)
        soft_loss = (
            F.kl_div(soft_student, soft_teacher, reduction="batchmean")
            * (self.temperature ** 2)
        )

        if labels is not None:
            hard_loss = F.cross_entropy(student_outputs.logits, labels)
            loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        else:
            loss = soft_loss

        return (loss, student_outputs) if return_outputs else loss


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.student, cache_dir=cache_dir)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher, cache_dir=cache_dir
    ).to(device)
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student, cache_dir=cache_dir
    ).to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    if args.dataset == "glue":
        dataset = load_dataset(args.dataset, args.dataset_config, cache_dir=ds_cache)
    else:
        dataset = load_from_disk(args.dataset) if os.path.isdir(args.dataset) else load_dataset(args.dataset, cache_dir=ds_cache)

    def tokenize_fn(examples):
        text = examples.get("sentence", examples.get("text", [""] * len(examples["label"])))
        out = tokenizer(text, truncation=True, padding="max_length", max_length=128)
        out["labels"] = examples["label"]
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() or torch.backends.mps.is_available(),
    )

    callbacks = []
    if args.watchdog:
        from watchdog_callbacks import PauseFlagCallback
        callbacks.append(PauseFlagCallback(args.output_dir))

    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        model=student,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Distilled model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
