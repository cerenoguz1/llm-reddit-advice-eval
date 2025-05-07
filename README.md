Local LLM Advice Evaluation  
CS 383 Final Project – Spring 2025

This project evaluates the ability of a **fully local large language model** (Gemma 2B Instruct) to generate helpful responses to real-life advice questions. We compare LLM responses to human-written answers using semantic similarity with Sentence-BERT.

---

Files

| File                         | Description                                      |
|-----------------------------|--------------------------------------------------|
| `reddit_advice_dataset.csv` | Real prompts + human advice from r/Advice       |
| `generate_and_score_local.py` | Sends prompts to local LLM, scores with SBERT   |
| `scored_advice_responses.csv` | Output file with LLM completions + scores       |
| `test_one_prompt.py`        | Lets you test any custom question on the model  |
| `requirements.txt`          | Python dependencies                             |

---

Setup Instructions

1. Clone the repo:
"git clone https://github.com/cerenoguz1/llm-reddit-advice-eval.git
cd llm-reddit-advice-eval"


2. Set up the Python environment:
  python3 -m venv myenv
  source myenv/bin/activate
  pip install -r requirements.txt


3. Install and load a local model (GGUF format):
  Used LM Studio to download and run Gemma 2 2B Instruct (Q5_K_M) locally. Register it with:
"llm llama-cpp add-model /path/to/model.gguf --alias gemma2"

 
4. Generate responses and similarity scores (10 prompts):
  python generate_and_score_local.py
  - Outputs willbe saved to: scored_advice_responses.csv


5. Try your own question!
  python test_one_prompt.py

---------------------------------------------------------------------------------------------

Compute similarity between human and model responses using:
  sentence-transformers → all-MiniLM-L6-v2
  Average similarity: ~0.39
  Best performance on neutral/factual advice
  Weakest performance on emotional topics


Prompt:
I feel like I've outgrown my friend group. What should I do?
LLM Response:
It's okay to grow and change. You can start setting boundaries and seeking out new connections that match your current values.
Human Response:
I went through this too. Start spending time doing things you love, and you'll naturally meet people you connect with.
Similarity Score: 0.48


* This project uses no OpenAI or cloud APIs — it runs entirely on local hardware using llm, llm-llama-cpp, and a GGUF model from LM Studio.
