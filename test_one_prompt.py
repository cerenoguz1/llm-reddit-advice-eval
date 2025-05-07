from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load fine-tuned model
model_path = "/Users/cerenoguz/gpt2-finetune/gpt2-finetuned/checkpoint-224"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

while True:
    prompt = input("Enter your advice question:\n> ").strip()
    if not prompt:
        break

    full_prompt = f"Prompt: {prompt}\nAdvice:"
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            do_sample=True,                  # ✅ Now sampling
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    advice = decoded.split("Advice:")[-1].strip()

    print("\nGenerated Advice:\n", advice)
