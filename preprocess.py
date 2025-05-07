import pandas as pd

df = pd.read_csv("reddit_advice_dataset.csv").dropna(subset=["prompt", "suggestion"])

def format(row):
    return {
        "text": f"<start_of_prompt>{row['prompt']}<end_of_prompt><start_of_response>{row['suggestion']}<end_of_response>"
    }

df.apply(format, axis=1).to_json("formatted.json", lines=True, orient="records")
