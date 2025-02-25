from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/flan-t5-base"

# Download and save the model locally
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Save locally
model.save_pretrained("./models/flan_t5_base")
tokenizer.save_pretrained("./models/flan_t5_base")

print("Model downloaded and saved locally.")
