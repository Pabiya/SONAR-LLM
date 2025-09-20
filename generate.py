import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

from datasets import load_dataset
from tqdm import tqdm

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class SonarInferenceWrapper(nn.Module):
    def __init__(self, llama_model, forward_proj, reverse_proj, sonar_decoder):
        super().__init__()
        self.llama_model = llama_model
        self.forward_proj = forward_proj
        self.reverse_proj = reverse_proj
        self.sonar_decoder = sonar_decoder

    @torch.no_grad()
    def inference_step(self, embedded_sents: torch.Tensor):
        """Run one forward step and return next sentence + its embedding."""
        if embedded_sents.ndim == 2:
            embedded_sents = embedded_sents.unsqueeze(0)  # [1, T, dim]

        proj = self.forward_proj(embedded_sents)
        out = self.llama_model(inputs_embeds=proj, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # [1, T, hidden]
        final_hidden = hidden[0, -1, :]
        reversed_emb = self.reverse_proj(final_hidden.unsqueeze(0))
        out_texts = self.sonar_decoder.predict(
            reversed_emb, target_lang="eng_Latn", max_seq_len=256
        )
        return out_texts[0], reversed_emb


def load_checkpoint_for_inference(ckpt_path, model, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"], strict=False)


@torch.no_grad()
def predict_next_sentence(sents, embeddings, eos_emb, threshold):
    if len(embeddings) == 0:
        embeddings = t2vec_model.predict(sents, source_lang="eng_Latn")
    embeddings = embeddings.to(device)

    next_sentence, next_embedding = inference_model.inference_step(embeddings)
    embedding_re = t2vec_model.predict([next_sentence], source_lang="eng_Latn").to(device)
    sim = F.cosine_similarity(embedding_re, eos_emb, dim=1).item()
    stop = sim >= threshold
    embeddings = torch.cat([embeddings, embedding_re])
    return next_sentence, embeddings, stop


@torch.no_grad()
def predict_next_text(text: str, eos_emb, threshold, add_begin):
    sents_original = sent_tokenize(text)
    if add_begin:
        sents_original = ["Begin of text."] + sents_original
    sents = [s.strip() for s in sents_original if s.strip()]
    embeddings = []

    while len(sents) <= 32:
        next_sentence, embeddings, stop = predict_next_sentence(
            sents, embeddings, eos_emb, threshold
        )
        if stop:
            break
        sents_original.append(next_sentence)
        sents.append(next_sentence.strip())
        
    if add_begin:
        sents_original = sents_original[1:]
    return " ".join(sents_original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--num_texts", type=int, default=512, help="How many texts to generate")
    parser.add_argument(
        "--prefix_mode", choices=["start", "half"], required=True, help="Prefix selection mode"
    )
    parser.add_argument("--output", required=True, help="File to save generated texts")
    parser.add_argument("--add_begin_text", action="store_true",
                        help="Prepend 'Begin of text.'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained_model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    llama_cfg_dict = cfg.get("llama_config", {})
    llama_cfg_dict["vocab_size"] = len(tokenizer)
    llama_cfg_dict["pad_token_id"] = tokenizer.pad_token_id
    llama_cfg_dict["bos_token_id"] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 128000
    llama_cfg_dict["eos_token_id"] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001
    llama_cfg = LlamaConfig(**llama_cfg_dict) if "llama_config" in cfg else LlamaConfig()

    llama_model = LlamaForCausalLM(llama_cfg).to(device).eval()

    hidden_size = llama_cfg.hidden_size
    embed_dim = cfg.get("embed_dim", 1024)

    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    ).eval()

    vec2text_model = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    ).eval()

    forward_projector = Projector(embed_dim, hidden_size).to(device).eval()
    reverse_projector = Projector(hidden_size, embed_dim).to(device).eval()

    inference_model = SonarInferenceWrapper(
        llama_model=llama_model,
        forward_proj=forward_projector,
        reverse_proj=reverse_projector,
        sonar_decoder=vec2text_model,
    ).to(device).eval()

    load_checkpoint_for_inference(args.checkpoint, inference_model, device=device)

    eos_emb = t2vec_model.predict(["End of sequence."], source_lang="eng_Latn").to(device)
    threshold = 0.98

    tiny_stories = load_dataset("roneneldan/TinyStories")
    val_texts = [item["text"] for item in tiny_stories["validation"] if item["text"]]
    test_texts = [t for t in val_texts[:-2500] if len(sent_tokenize(t)) <= 32]

    generated_texts = []
    for text in tqdm(test_texts[: args.num_texts]):
        sents = sent_tokenize(text)
        if args.prefix_mode == "start":
            prefix_len = 2
        elif args.prefix_mode == "half":
            prefix_len = max(1, len(sents) // 2)
        prefix = " ".join(sents[:prefix_len])
        generated = predict_next_text(prefix, eos_emb, threshold, args.add_begin_text)
        generated_texts.append({"prefix": prefix, "full_text": generated, "original_text": text})

    with open(args.output, "w", encoding="utf-8") as f_out:
        json.dump(generated_texts, f_out, ensure_ascii=False, indent=2)
