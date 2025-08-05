# LLM & SONAR-LLM Training on Tiny Stories

This repository contains scripts for training Large Language Models (LLM) and SONAR-LLM on the **Tiny Stories** dataset.

---

## 📁 Available Scripts

### `train_llm.py`  
**Description**:  
Trains a Large Language Model (LLM) on the Tiny Stories dataset.

**Example command**:
```bash
torchrun --nproc_per_node=8 train_llm.py \
  --use_wandb \
  --wandb_project ProjectName \
  --wandb_run_name RunName \
  --use_mixed_precision \
  --grad_accum_steps 8 \
  --output_dir OutName \
  --batch_size 8
```

---

### `train_sonarllm.py`  
**Description**:  
Legacy training script for SONAR-LLM. Encodes sentences **on-the-fly** using the Sonar encoder during training.  

**Example command**:
```bash
torchrun --nproc_per_node=8 train_sonarllm.py \
  --use_wandb \
  --wandb_project ProjectName \
  --wandb_run_name RunName \
  --use_mixed_precision \
  --grad_accum_steps 64 \
  --output_dir OutName \
  --batch_size 1
```

---

### `train_sonarllm_fast.py`  
**Description**:  
Main training script for SONAR-LLM. Uses **precomputed** sentence embeddings generated with the Sonar encoder.

**Example command**:
```bash
torchrun --nproc_per_node=8 train_sonarllm_fast.py \
  --use_wandb \
  --wandb_project ProjectName \
  --wandb_run_name RunName \
  --use_mixed_precision \
  --grad_accum_steps 64 \
  --output_dir OutName \
  --batch_size 1
```

---

### `sonar_encoding.py`  
**Description**:  
Generates sentence embeddings using the Sonar encoder. Required for `train_fast.py`.

**Example command**:
```bash
torchrun --nproc_per_node=8 sonar_encoding.py \
  --batch_size 32 \
  --output_dir ./sonar_embeddings_ts_100_shuffle
```

---

### `generate.py`  
**Description**:  
Generates full texts from Tiny Stories prefixes using a trained SONAR-LLM model.

**Example command**:
```bash
python generate.py \
  --config ./SonarLLM_900M.json \
  --checkpoint ./checkpoint.pt \
  --num_texts 512 \
  --prefix_mode start \
  --output ./generated_900M.json \
```

---

### `evaluate_nlg.py`  
**Description**:  
Evaluates BLEU, ROUGE and METEOR metrics.

**Example command**:
```bash
python evaluate_nlg.py generated_900M.json
```

---

## ✅ TinyStories SONAR-LLM Checkpoints

| Model Size        | Checkpoint Link                                                                 |
|-------------------|----------------------------------------------------------------------------------|
| SONAR-LLM 100M     | [Download](https://drive.google.com/file/d/18KRywQXjRQFbbVZhkUp5uaWyCuD3pEga/view?usp=sharing)                  |
| SONAR-LLM 300M     | [Download](https://drive.google.com/file/d/1A4Fqm-w1arBRiU_V67OrfBf_Ymrm71-7/view?usp=sharing)                  |
| SONAR-LLM 600M     | [Download](https://drive.google.com/file/d/1oG1_8whE1USC1lelF7-A9_Lhvfd2tNQm/view?usp=sharing)                  |
| SONAR-LLM 900M     | [Download](https://drive.google.com/file/d/1lzUkUNbFQkeYLp69FYrLA1vAB3--WbrY/view?usp=sharing)                  |
