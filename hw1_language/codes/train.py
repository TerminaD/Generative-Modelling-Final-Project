from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import math

# Load the tokenizer and model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

dataset = load_dataset('wikitext', 'wikitext-2-v1')
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

small_train_dataset = tokenized_dataset["train"]
small_eval_dataset = tokenized_dataset["test"]

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=2,
                                  gradient_accumulation_steps=4,
                                  num_train_epochs=1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

eval_results = trainer.evaluate()

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

path = "./ckpts"
trainer.save_model(path)
