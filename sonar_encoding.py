import os
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

from datasets import load_dataset

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of text samples processed per GPU step.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for DDP.")
    parser.add_argument("--output_dir", type=str, default="./sonar_embeddings_ts_100_shuffle",
                        help="Where to save the output PT files.")
    return parser.parse_args()


class TinyStoriesTextDataset(Dataset):
    """
    Simple dataset returning a single text (TinyStory) per index.
    """
    def __init__(self, text_list):
        super().__init__()
        self.texts = text_list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_and_encode(batch_texts, sonar_model):
    """
    Tokenize each text into sentences, batch the sentences through Sonar,
    and regroup them per text.
    Returns a list of dicts, each dict has:
      {
         "original_text": str,
         "sentences": [str, ...],
         "embeddings": [ [float, ..., float], ... ]  # 1024-D
      }
    """
    batch_sentences = []
    batch_map = [] 

    for i, txt in enumerate(batch_texts):
        sents = sent_tokenize(txt)
        sents = [s.strip() for s in sents if s.strip()]
        for j, s in enumerate(sents):
            batch_sentences.append(s)
            batch_map.append((i, j))

    if len(batch_sentences) > 0:
        with torch.no_grad():
            emb_tensors = sonar_model.predict(batch_sentences, source_lang="eng_Latn")
        all_embeddings = emb_tensors.cpu().tolist()
    else:
        all_embeddings = []

    results = [
        {"original_text": batch_texts[i], "sentences": [], "embeddings": []}
        for i in range(len(batch_texts))
    ]

    idx_in_sentences = 0
    for (text_i, _), emb_vec in zip(batch_map, all_embeddings):
        results[text_i]["sentences"].append(batch_sentences[idx_in_sentences])
        results[text_i]["embeddings"].append(emb_vec)
        idx_in_sentences += 1

    return results


def ddp_main():
    args = parse_args()

    # ---------------------------------------------------------------
    # 1) Setup DDP
    # ---------------------------------------------------------------
    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = (world_size > 1)

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (not distributed) or (dist.get_rank() == 0)

    if is_main_process:
        print("Running DDP Embedding Script (Online Splitting)")
        if distributed:
            print(f"World Size = {world_size}, local_rank = {local_rank}")
        else:
            print("Single-process run.")

    # ---------------------------------------------------------------
    # 2) Load Sonar text->embedding pipeline
    # ---------------------------------------------------------------
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    ).eval()
    for param in t2vec_model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------------
    # 3) Load TinyStories dataset
    # ---------------------------------------------------------------
    while True:
        try:
            tiny_stories = load_dataset('roneneldan/TinyStories')
            train_texts = [item['text'] for item in tiny_stories['train'] if item['text']]
            val_texts   = [item['text'] for item in tiny_stories['validation'] if item['text']]
            break
        except:
            time.sleep(1)

    num_train = len(train_texts)
    num_val   = len(val_texts)

    if is_main_process:
        print(f"Loaded TinyStories. Train size = {num_train}, Val size = {num_val}")

    # ---------------------------------------------------------------
    # 4) DataLoaders for train & val
    # ---------------------------------------------------------------
    train_dataset = TinyStoriesTextDataset(train_texts)
    val_dataset   = TinyStoriesTextDataset(val_texts)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # ---------------------------------------------------------------
    # 5) Online Splitting of TRAIN into 100 buckets
    # ---------------------------------------------------------------
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # We'll define bucket sizes for the 100 chunks
        bucket_size = num_train // 100
        remainder   = num_train % 100

        def get_bucket_size(idx):
            # Buckets 0..98 each get bucket_size
            # Bucket 99 gets bucket_size + remainder
            return bucket_size if idx < 99 else (bucket_size + remainder)

    if is_main_process:
        print("=== Encoding TRAIN set in 100 online buckets ===")

    train_results_buffer = []
    current_bucket_idx = 0
    items_in_current_bucket = 0
    if is_main_process:
        bucket_target_size = get_bucket_size(current_bucket_idx)
    else:
        bucket_target_size = None

    for batch_texts in tqdm(train_loader, disable=not is_main_process):
        local_batch_res = collate_and_encode(batch_texts, t2vec_model)

        if distributed:
            if is_main_process:
                gather_list = [None for _ in range(world_size)]
                dist.gather_object(local_batch_res, object_gather_list=gather_list, dst=0)
            else:
                dist.gather_object(local_batch_res, dst=0)
        else:
            gather_list = [local_batch_res]

        if is_main_process:
            merged_batch_res = []
            if distributed:
                for part in gather_list:
                    merged_batch_res.extend(part)
            else:
                merged_batch_res.extend(gather_list[0])

            for item in merged_batch_res:
                train_results_buffer.append(item)
                items_in_current_bucket += 1

                if items_in_current_bucket == bucket_target_size:
                    out_path = os.path.join(
                        args.output_dir,
                        f"tiny_stories_sonar_embeddings_train_bucket_{current_bucket_idx}.pt"
                    )
                    torch.save({"train": train_results_buffer}, out_path)
                    print(f"[Bucket {current_bucket_idx}] Saved {bucket_target_size} items → {out_path}")

                    train_results_buffer = []
                    items_in_current_bucket = 0
                    current_bucket_idx += 1

                    if current_bucket_idx < 100:
                        bucket_target_size = get_bucket_size(current_bucket_idx)
                    else:
                        break

            if current_bucket_idx >= 100:
                break

        if distributed:
            dist.barrier()

    if is_main_process:
        while current_bucket_idx < 100:
            if len(train_results_buffer) > 0:
                out_path = os.path.join(
                    args.output_dir,
                    f"tiny_stories_sonar_embeddings_train_bucket_{current_bucket_idx}.pt"
                )
                torch.save({"train": train_results_buffer}, out_path)
                print(f"[Bucket {current_bucket_idx}] leftover {len(train_results_buffer)} → {out_path}")
                train_results_buffer = []
            current_bucket_idx += 1

    if distributed:
        dist.barrier()

    # ---------------------------------------------------------------
    # 6) Encode Validation in a single pass
    # ---------------------------------------------------------------
    if is_main_process:
        print("=== Encoding VALIDATION set ===")

    val_results_local = []
    for batch_texts in tqdm(val_loader, disable=not is_main_process):
        batch_encoded = collate_and_encode(batch_texts, t2vec_model)
        val_results_local.extend(batch_encoded)

    if distributed:
        if is_main_process:
            gather_list = [None for _ in range(world_size)]
            dist.gather_object(val_results_local, object_gather_list=gather_list, dst=0)
        else:
            dist.gather_object(val_results_local, dst=0)
    else:
        gather_list = [val_results_local]

    if is_main_process:
        val_results = []
        if distributed:
            for part in gather_list:
                val_results.extend(part)
        else:
            val_results.extend(gather_list[0])

        val_out_path = os.path.join(args.output_dir, "tiny_stories_sonar_embeddings_val.pt")
        torch.save({"validation": val_results}, val_out_path)
        print(f"Saved validation set with {len(val_results)} items → {val_out_path}")

    if distributed:
        dist.barrier()

    # ---------------------------------------------------------------
    # 7) Save embedding for the special phrase "End of sequence."
    # ---------------------------------------------------------------
    if is_main_process:
        # Generate embedding for the special phrase
        with torch.no_grad():
            eos_tensor = t2vec_model.predict(["End of sequence."], source_lang="eng_Latn")
        eos_embedding = eos_tensor.cpu().tolist()

        # Save to output directory
        eos_path = os.path.join(args.output_dir, "sonar_eos_embedding.pt")
        torch.save(eos_embedding, eos_path)
        print(f"Saved 'End of sequence.' embedding → {eos_path}")

if __name__ == "__main__":
    ddp_main()