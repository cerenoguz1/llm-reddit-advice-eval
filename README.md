Local LLM Advice Evaluation

This project evaluates the ability of a fully local GPT-2 model to generate helpful responses to real-life advice questions from Reddit. Using Sentence-BERT, we quantitatively compare model responses with human-written advice based on semantic similarity.
⚠️ Disclaimer

This project uses real prompts from Reddit's r/Advice. The file scored_advice_responses.csv may contain sensitive, disturbing, or sexual content. Please review responsibly.
Highlights

Local Inference Only: All experiments are conducted offline using Hugging Face's transformers—no external API calls.
Custom Dataset: Collected and processed ~5,000 advice Q&A pairs from Reddit.
Evaluation: Scoring based on Sentence-BERT cosine similarity for semantic comparison.
🔗 Resources

GitHub: llm-reddit-advice-eval
Google Drive (Download Only): checkpoint-224.zip
Google Drive Contents
File	Description
llm-reddit-advice-eval.zip	Complete codebase for inference + scoring
checkpoint-224.zip	Fine-tuned GPT-2 model
gpt2-finetune-full-backup.zip	All code, training data, and outputs
requirements.txt	Reproducible Python environment
1. Problem Statement

Goal:
Assess whether a fine-tuned GPT-2 can generate meaningful, helpful responses to advice questions sourced from Reddit.
Core Features:
Local fine-tuning and inference
No third-party LLM APIs
Semantic evaluation using Sentence-BERT
2. Dataset

Source: Reddit's r/Advice subreddit (via Reddit API + manual curation)
Stats:
~5,000 entries
Avg prompt: ~100 tokens
Avg response: ~80 tokens
Preprocessing:
Cleaned, formatted, and converted to Hugging Face JSONL
Train/Test Split: 4,500 / 500
3. Prompt Format

Used During Fine-Tuning & Inference:
Prompt: <user advice question>\nAdvice:
Example:
Input:
Prompt: My prom dress got ruined the day of the event. What should I do?\nAdvice:
Output:
If you're not sure what to wear, try wearing a pair of jeans or something similar...
Sampling Settings:
max_new_tokens=100
temperature=0.8
top_p=0.95
do_sample=True
pad_token_id=50256 (eos_token)
4. Evaluation

Metric:
Sentence-BERT (all-MiniLM-L6-v2) cosine similarity
Steps:
Generate advice responses for 500 test prompts
Score against original human-written answers
Output stored in scored_advice_responses.csv
Pros:
Fully offline & reproducible
No need for expensive APIs
Cons:
GPT-2 occasionally misses emotional depth or sensitivity
5. Results

Metric	Value
Highest	0.506
Lowest	-0.046
Average	0.1799
6. Instructor Feedback

Code and results uploaded to GitHub and Drive
Instructor script: test_one_prompt.py
README refined for clarity and reproducibility
7. Author Contribution

Ceren Oguz
Designed dataset scraping & preprocessing pipeline
Fine-tuned GPT-2 using Hugging Face Trainer
Built scoring mechanism with Sentence-BERT
Maintained codebase, documentation, and backups
Authored testing and analysis scripts
Project Structure

llm-reddit-advice-eval/
├── generate_and_score_local.py     # Generates advice and scores it
├── test_one_prompt.py              # Run model on a single input
├── analyze_results.py              # Optional results analysis
├── preprocess.py                   # Prepares dataset for training
├── formatted.jsonl                 # Final dataset (Hugging Face format)
├── reddit_advice_dataset.csv       # Raw Reddit data
├── scored_advice_responses.csv     # Output scores
├── myenv/                          # Local Python environment (ignored)
Model Details

Model: GPT-2 (from Hugging Face)
Training Data: Custom Reddit dataset (formatted.jsonl)
Inference: Local only
Evaluation Model: all-MiniLM-L6-v2 Sentence-BERT
🔍 Try It Out

python test_one_prompt.py
