#!/usr/bin/env python
import os
import time
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

import wandb  # For logging

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.optimization import get_constant_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler


# ------------------------------------------------------------------------
# 1) Parse command line arguments
# ------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=820, help="Warmup steps for the LR scheduler")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps within an epoch")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Use fewer samples for debugging. -1 for all.")
    parser.add_argument("--output_dir", type=str, default="./llama_pretrain_checkpoints")

    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="my_project", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default="llama_pretrain_run", help="Wandb run name.")

    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Use mixed precision training (fp16 autocast).")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps.")
    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------
# 2) Dataset class and collator for causal LM training
# ------------------------------------------------------------------------
class LLMDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=2048):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        for text in texts:
            tokenized = tokenizer(text, add_special_tokens=True)["input_ids"]
            if tokenized[-1] != tokenizer.eos_token_id:
                tokenized.append(tokenizer.eos_token_id)
            if len(tokenized) > block_size:
                tokenized = tokenized[:block_size]
            self.examples.append(torch.tensor(tokenized, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def lm_collator(batch):
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = batch.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": batch, "labels": labels}


# ------------------------------------------------------------------------
# 3) Main training script
# ------------------------------------------------------------------------
def main():
    args = parse_args()

    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    is_main_process = (not distributed) or (dist.get_rank() == 0)
    if is_main_process:
        print(f"Using device: {device}")
        if distributed:
            print(f"Distributed training on {world_size} GPUs.")
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            wandb.config.update(vars(args))

    # --------------------------------------------------------------------
    # Print Global and Effective Batch Size
    # --------------------------------------------------------------------
    if is_main_process:
        global_batch_size = args.batch_size * world_size
        effective_batch_size = global_batch_size * args.grad_accum_steps
        print(f"Global batch size = {global_batch_size} "
              f"(local_batch_size={args.batch_size} x world_size={world_size})")
        print(f"Effective batch size per optimizer step = {effective_batch_size} "
              f"(global_batch_size x grad_accum_steps={args.grad_accum_steps})")

    # --------------------------------------------------------------------
    # Load tokenizer and configure the model
    # --------------------------------------------------------------------
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None or tokenizer.pad_token_id >= tokenizer.vocab_size:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if is_main_process:
            print("Added pad token. New vocab size:", len(tokenizer))

    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)

    configuration = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=10,
        num_attention_heads=16,
        hidden_act='silu',
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 128000,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=64,
    )

    model = LlamaForCausalLM(configuration).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.train()

    # --------------------------------------------------------------------
    # Load the TinyStories dataset from Hugging Face
    # --------------------------------------------------------------------
    ds = load_dataset("roneneldan/TinyStories")
    train_texts = [item["text"] for item in ds["train"] if item["text"]]
    val_texts = [item["text"] for item in ds["validation"] if item["text"]]
    if args.max_train_samples > 0:
        train_texts = train_texts[:args.max_train_samples]
    val_texts = val_texts[-2500:]

    block_size = 2048
    train_dataset = LLMDataset(train_texts, tokenizer, block_size=block_size)
    val_dataset = LLMDataset(val_texts, tokenizer, block_size=block_size)
    if is_main_process:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

    # --------------------------------------------------------------------
    # DataLoaders with our LM collator
    # --------------------------------------------------------------------
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lm_collator,
        num_workers=0,
        drop_last=True,
    )
    val_dloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=lm_collator,
        num_workers=0,
        drop_last=True,
    )

    # --------------------------------------------------------------------
    # Optimizer, Scheduler, and Mixed Precision Setup
    # --------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)

    steps_per_epoch = len(train_dloader)
    total_steps = (steps_per_epoch // args.grad_accum_steps) * args.epochs
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps
    )
    scaler = GradScaler(enabled=args.use_mixed_precision)

    # --------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------
    global_step = 0
    total_fwd_passes = 0

    model.train()

    for epoch in range(args.epochs):
        if distributed:
            train_dloader.sampler.set_epoch(epoch)

        accumulated_loss = 0.0
        accumulated_steps = 0

        for step, batch in enumerate(tqdm(train_dloader, desc=f"Epoch {epoch+1}")):
            total_fwd_passes += 1  # increment on every mini-batch forward pass

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=args.use_mixed_precision):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            accumulated_steps += 1

            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ---------------------------------------------------------
                # Logging every N steps
                # ---------------------------------------------------------
                if global_step % args.logging_steps == 0 and is_main_process:
                    avg_loss = args.grad_accum_steps * accumulated_loss / accumulated_steps
                    current_lr = scheduler.get_last_lr()[0]

                    samples_seen = total_fwd_passes * args.batch_size
                    if distributed:
                        samples_seen *= world_size

                    print(f"[Epoch {epoch+1}, Global Step {global_step}] "
                          f"Avg Loss = {avg_loss:.4f}, LR = {current_lr:.6f}, "
                          f"Samples Seen = {samples_seen}")
                    if args.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/step": global_step,
                            "train/samples_seen": samples_seen,
                            "train/lr": current_lr,
                        })
                    accumulated_loss = 0.0
                    accumulated_steps = 0

                # ---------------------------------------------------------
                # Evaluate every N steps
                # ---------------------------------------------------------
                if (global_step % args.eval_steps == 0) and (global_step > 0):
                    val_loss_step = evaluate(model, val_dloader, device, distributed)
                    if is_main_process:
                        print(f"[Step {global_step}] Validation Loss = {val_loss_step:.4f}")
                        if args.use_wandb:
                            wandb.log({"eval/loss_steps": val_loss_step,
                                       "eval/step": global_step})

        # -------------------------------------------------------------
        # End of epoch: evaluate, save checkpoint
        # -------------------------------------------------------------
        val_loss_epoch = evaluate(model, val_dloader, device, distributed)
        if is_main_process:
            print(f"[Epoch {epoch+1}] Validation Loss = {val_loss_epoch:.4f}")
            if args.use_wandb:
                wandb.log({"eval/loss_epoch": val_loss_epoch, "eval/epoch": epoch + 1})

            save_full_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch+1,
                step_in_epoch=0,
                global_step=global_step,
                output_dir=args.output_dir
            )

    if distributed:
        dist.barrier()


def evaluate(model, val_dloader, device, distributed):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_dloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast(enabled=False):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            num_tokens = input_ids.ne(tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    if distributed:
        loss_tensor = torch.tensor([total_loss, total_tokens], dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_tokens = loss_tensor[0].item(), loss_tensor[1].item()
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0.0


def save_full_checkpoint(model, optimizer, scheduler, epoch, step_in_epoch, global_step, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    ckpt_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "random_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved full checkpoint to {ckpt_path}")


def load_full_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    random.setstate(ckpt["random_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if "cuda_rng_state_all" in ckpt:
        for i, state in enumerate(ckpt["cuda_rng_state_all"]):
            torch.cuda.set_rng_state(state.cpu(), device=i)
    return {
        "epoch": ckpt["epoch"],
        "step_in_epoch": ckpt["step_in_epoch"],
        "global_step": ckpt["global_step"],
    }


if __name__ == "__main__":
    main()