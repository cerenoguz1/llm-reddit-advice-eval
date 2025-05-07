import os
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# Paths
MODEL_PATH = "/Users/cerenoguz/gpt2-finetune/gpt2-finetuned/checkpoint-224"

DATASET_PATH = "reddit_advice_dataset.csv"
OUTPUT_PATH = "scored_advice_responses.csv"

# Load models
print("Loading Sentence-BERT model...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading fine-tuned GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load data
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["prompt", "suggestion"]).sample(n=100, random_state=42)  # test subset

llm_responses = []
similarities = []

print()
print("DISCLAIMER:")
print("This project uses real prompts scraped from Reddit's r/Advice forum.")
print("As a result, some entries in 'scored_advice_responses.csv' may contain sensitive, disturbing, or sexual content.")
print("Please review the file with caution.\n")

for idx, row in df.iterrows():
    prompt = row["prompt"]

    # Encode prompt and generate response
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Trim prompt from the generated response
    llm_answer = generated_text[len(prompt):].strip()
    llm_responses.append(llm_answer)

    # Similarity
    try:
        emb1 = sbert.encode(row["suggestion"], convert_to_tensor=True)
        emb2 = sbert.encode(llm_answer, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(emb1, emb2).item())
    except:
        sim_score = 0.0
    similarities.append(sim_score)

# Save results
df["llm_response"] = llm_responses
df["similarity"] = similarities
df.to_csv(OUTPUT_PATH, index=False)

# Print stats
df = pd.read_csv(OUTPUT_PATH)
avg_sim = df["similarity"].mean()
print("Highest similarity:", df["similarity"].max())
print("Lowest similarity:", df["similarity"].min())
print(f"Average similarity score: {avg_sim:.4f}")
print(f"Done! Results have been saved to {OUTPUT_PATH}")
