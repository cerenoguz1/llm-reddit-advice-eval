from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Load your dataset
dataset = load_dataset("json", data_files="formatted.jsonl", split="train")

# Load tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
trainer.save_model("./gpt2-finetuned")
trainer.save_model("gpt2-finetuned-reddit")
tokenizer.save_pretrained("gpt2-finetuned-reddit")

# Save the model and tokenizer
trainer.save_model("finetuned-model")
tokenizer.save_pretrained("finetuned-model")

