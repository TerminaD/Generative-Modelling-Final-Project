from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

prompt = "Two households, both alike in dignity,"	# Set your prompt here

model = AutoModelForCausalLM.from_pretrained("./ckpts", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(generator(prompt))