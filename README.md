#Local LLM Advice Evaluation (CS 383 Final Project)

This project evaluates the ability of a local large language model (Gemma 2B) to generate helpful advice using real prompts from Reddit's r/Advice community. Model outputs are compared against human-written suggestions using semantic similarity (Sentence-BERT).

---

##Contents

- `reddit_advice_dataset.csv`: Real prompts + top human responses from Reddit
- `generate_and_score_local.py`: Runs the local model and generates LLM advice
- `scored_advice_responses.csv`: Output file with LLM responses + similarity scores
- `analyze_results.py`: Ranks high/low similarity samples and computes averages

---

##Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-reddit-advice-eval.git
cd llm-reddit-advice-eval
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

Model Info

Model: gemma-2-2b-it (GGUF format)
Run locally with llm + llm-llama-cpp plugin

Prompt: "How do I make new friends?"
LLM:    "Try joining clubs or volunteering..."
Human:  "Meet people through shared activities..."
Score:  0.52

Results Summary:

Average cosine similarity: 0.3926
Highest: ~0.59
Lowest: ~0.12
Best alignment on factual/neutral advice
Divergence on emotional or sensitive topics
