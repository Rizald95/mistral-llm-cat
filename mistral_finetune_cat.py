import json
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Step 1: Memuat file data.json
with open('data.json', 'r') as f:
    dataset_json = json.load(f)

# Step 2: Format ulang dataset agar sesuai dengan model fine-tuning
formatted_dataset = [{"text": f"Pertanyaan: {item['question']} Jawaban: {item['answer']}"} for item in dataset_json]

# Step 3: Convert ke Hugging Face Dataset
dataset = Dataset.from_list(formatted_dataset)

# Step 4: Fine-tuning dengan model Mistral 4-bit
max_seq_length = 2048  # Supports RoPE scaling internally

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",  # Pilih model Mistral yang sudah pre-quantized
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Step 5: Tambahkan fast LoRA ke model untuk optimalisasi training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
)

# Step 6: Mengatur parameter training dan memulai training dengan SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# Mulai proses fine-tuning
trainer.train()
