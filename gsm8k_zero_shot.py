#!/usr/bin/env python3
# gsm8k_zero_shot.py — Evaluate SONAR‑LLM zero‑shot on GSM8K without fine‑tuning.

import os
import re
import json
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _lazy_imports():
    from tqdm import tqdm
    from datasets import load_dataset
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
    from sonar.inference_pipelines.text import (
        TextToEmbeddingModelPipeline,
        EmbeddingToTextModelPipeline,
    )
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download("punkt", quiet=True)
    return tqdm, load_dataset, AutoTokenizer, LlamaForCausalLM, LlamaConfig, TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline, sent_tokenize

class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    def forward(self, x):
        return self.ln(self.proj(x))

class SonarInferenceWrapper(nn.Module):
    def __init__(self, llama_model, forward_proj, reverse_proj, sonar_decoder):
        super().__init__()
        self.llama_model = llama_model
        self.forward_proj = forward_proj
        self.reverse_proj = reverse_proj
        self.sonar_decoder = sonar_decoder

    @torch.no_grad()
    def inference_step(self, embedded_sents: torch.Tensor) -> Tuple[str, torch.Tensor]:
        if embedded_sents.ndim == 2:
            embedded_sents = embedded_sents.unsqueeze(0)  # [1, T, 1024]
        proj = self.forward_proj(embedded_sents)               # [1, T, hidden]
        out = self.llama_model(inputs_embeds=proj, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1][:, -1:, :]         # [1, 1, hidden]
        pred_embed = self.reverse_proj(last_hidden).squeeze(0) # [1, 1024]
        decoded = self.sonar_decoder.predict(pred_embed, target_lang="eng_Latn")[0]
        return decoded, pred_embed

def build_models_from_config(cfg: dict, device: torch.device):
    _, _, AutoTokenizer, LlamaForCausalLM, LlamaConfig, TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline, _ = _lazy_imports()
    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained_model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    llama_cfg = cfg.get("llama_config", {}).copy()
    llama_cfg["vocab_size"] = len(tokenizer)
    llama_cfg["pad_token_id"] = tokenizer.pad_token_id
    llama_cfg["bos_token_id"] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 128000
    llama_cfg["eos_token_id"] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001

    llama_config = LlamaConfig(**llama_cfg)
    llama = LlamaForCausalLM(llama_config).to(device)

    in_dim = cfg.get("embed_dim", 1024)  # matches generate.py naming
    hidden = llama_cfg["hidden_size"]
    forward_proj = Projector(in_dim, hidden).to(device)
    reverse_proj = Projector(hidden, in_dim).to(device)

    # Match your generate.py constructors (no embedding_dim / dtype args)
    t2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    ).eval()
    e2t = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    ).eval()
    return tokenizer, llama, forward_proj, reverse_proj, t2vec, e2t

def load_checkpoint(ckpt_path: str, llama: nn.Module, forward: nn.Module, reverse: nn.Module):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    llama.load_state_dict({k.replace("llama.", ""): v for k, v in state.items() if k.startswith("llama.")}, strict=False)
    forward.load_state_dict({k.replace("forward_proj.", ""): v for k, v in state.items() if k.startswith("forward_proj.")}, strict=False)
    reverse.load_state_dict({k.replace("reverse_proj.", ""): v for k, v in state.items() if k.startswith("reverse_proj.")}, strict=False)

@torch.no_grad()
def sonar_generate_continuation(inference_model: SonarInferenceWrapper,
                                t2vec_model,
                                prefix_text: str,
                                eos_embedding: torch.Tensor,
                                eos_threshold: float = 0.98,
                                max_sentences: int = 32,
                                add_begin: bool = True) -> str:
    *_, sent_tokenize = _lazy_imports()
    sents = sent_tokenize(prefix_text)
    if add_begin:
        sents = ["Begin of text."] + sents
    sents = [s.strip() for s in sents if s.strip()]
    embeddings = t2vec_model.predict(sents, source_lang="eng_Latn").to(eos_embedding.device)  # keep on same device
    generated_sents = []
    while len(generated_sents) < max_sentences:
        nxt_text, nxt_emb = inference_model.inference_step(embeddings)  # already on correct device
        generated_sents.append(nxt_text)
        re_emb = t2vec_model.predict([nxt_text], source_lang="eng_Latn").to(eos_embedding.device)
        sim = F.cosine_similarity(re_emb, eos_embedding, dim=1).item()
        embeddings = torch.cat([embeddings, re_emb], dim=0)  # no .cpu()
        if sim >= eos_threshold:
            break
    return " ".join(generated_sents).strip()

def extract_gsm8k_answer(text: str) -> str:
    m = re.search(r"####\s*([^\n\r]+)", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:/\d+)?", text)
    return nums[-1] if nums else ""

def normalize_ans(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(",", "").replace(" ", "")
    return s

def make_prompt(question: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step by step and show your reasoning. "
        "At the end, on a new line, write '#### <final answer as an integer or simplified fraction>'.\n\n"
        f"Question: {question}\n\n"
        "Let's think step by step."
    )

def _to_tensor_eos(obj, device):
    if isinstance(obj, torch.Tensor):
        t = obj
    else:
        t = torch.tensor(obj, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)  # [1, 1024]
    return t.to(device)

def main():
    tqdm, load_dataset, *_ = _lazy_imports()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eos_path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"])
    parser.add_argument("--subset", default="main")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--eos_threshold", type=float, default=0.98)
    parser.add_argument("--max_sentences", type=int, default=16)
    parser.add_argument("--add_begin_token", action="store_true")
    parser.add_argument("--save_path", default="results/gsm8k_zeroshot.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer, llama, fproj, rproj, t2v, v2t = build_models_from_config(cfg, device)
    load_checkpoint(args.checkpoint, llama, fproj, rproj)
    for p in llama.parameters(): p.requires_grad_(False)
    for p in fproj.parameters(): p.requires_grad_(False)
    for p in rproj.parameters(): p.requires_grad_(False)

    eos_obj = torch.load(args.eos_path, map_location=device)
    eos_emb = _to_tensor_eos(eos_obj, device)  # robust to list/np/tensor
    infer = SonarInferenceWrapper(llama, fproj, rproj, v2t).to(device).eval()

    ds = load_dataset("gsm8k", args.subset, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    correct = 0
    total = 0
    with open(args.save_path, "w", encoding="utf-8") as fout:
        for item in tqdm(ds, desc="GSM8K zero‑shot"):
            q = item["question"]
            gold_full = item["answer"]
            gold = normalize_ans(extract_gsm8k_answer(gold_full))

            prompt = make_prompt(q)
            gen = sonar_generate_continuation(
                infer, t2v, prompt, eos_emb,
                eos_threshold=args.eos_threshold,
                max_sentences=args.max_sentences,
                add_begin=args.add_begin_token
            )
            pred = normalize_ans(extract_gsm8k_answer(gen))
            ok = (pred == gold) and (pred != "")

            fout.write(json.dumps({
                "question": q,
                "gold": gold_full,
                "gold_extracted": gold,
                "generation": gen,
                "pred_extracted": pred,
                "correct": bool(ok),
            }, ensure_ascii=False) + "\n")

            correct += int(ok)
            total += 1

    acc = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")
    print(f"Saved per‑example outputs to: {args.save_path}")

if __name__ == "__main__":
    main()
