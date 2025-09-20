import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# Match generate.py constructors
t2v = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
).eval()

emb = t2v.predict(["End of sequence."], source_lang="eng_Latn")  # [1, 1024] tensor
torch.save(emb, "tiny_stories_sonar_eos.pt")
print("saved: tiny_stories_sonar_eos.pt", tuple(emb.shape))
