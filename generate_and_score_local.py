import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import subprocess
import time
from sentence_transformers import SentenceTransformer, util



sbert = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("reddit_advice_dataset.csv").head(10)




llm_responses = []
similarities = []

def get_llm_response(prompt):
    try:
        result = subprocess.run(
            ["llm", "-m", "gemma2", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error on prompt: {prompt[:30]} — {e}")
        return ""

def compute_similarity(a, b):
    emb1 = sbert.encode(a, convert_to_tensor=True)
    emb2 = sbert.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

for i, row in df.iterrows():
    prompt = row['prompt']
    human = row['suggestion']
    
    print(f"[{i+1}/{len(df)}] Generating for: {prompt[:40]}...")

    llm_reply = get_llm_response(prompt)
    sim = compute_similarity(human, llm_reply)

    llm_responses.append(llm_reply)
    similarities.append(sim)

    time.sleep(1.0)  # reduce CPU stress

df['llm_response'] = llm_responses
df['similarity'] = similarities

df.to_csv("scored_advice_responses.csv", index=False)
print("Done! Results saved to scored_advice_responses.csv")
