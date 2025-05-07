import pandas as pd

df = pd.read_csv("scored_advice_responses.csv")

df = df.dropna(subset=["similarity"])

avg = df["similarity"].mean()
print(f"\n🔢 Average similarity: {avg:.4f}")

print("\nTop 3 highest similarity responses:\n")
top = df.sort_values(by="similarity", ascending=False).head(3)
for i, row in top.iterrows():
    print(f"- Prompt: {row['prompt'][:80]}...")
    print(f"  Human: {row['suggestion'][:80]}...")
    print(f"  LLM:   {row['llm_response'][:80]}...")
    print(f"  Score: {row['similarity']:.4f}\n")

print("\nBottom 3 lowest similarity responses:\n")
bottom = df.sort_values(by="similarity").head(3)
for i, row in bottom.iterrows():
    print(f"- Prompt: {row['prompt'][:80]}...")
    print(f"  Human: {row['suggestion'][:80]}...")
    print(f"  LLM:   {row['llm_response'][:80]}...")
    print(f"  Score: {row['similarity']:.4f}\n")
