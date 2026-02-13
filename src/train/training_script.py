import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM

# This tells the trainer to only calculate loss on text after this tag
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template, tokenizer=tokenizer)

# Load credentials from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Login to Hugging Face
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN not found in .env file")

# 1. SETUP & DOWNLOAD
# Example (Updated to a real available ID)
model_id = "Qwen/Qwen2.5-7B-Instruct"
output_name = "my-finetuned-sft-model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. LOCAL DATA LOADING
# Assuming your file is named 'training_data.jsonl' in the same directory
dataset = load_dataset(
    "json", data_files="../Data Generation/dataset.jsonl", split="train")

# 3. QLoRA CONFIGURATION
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj",
                    "o_proj"],  # Expanded for better results
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. TRAINING
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

sft_config = SFTConfig(
    output_dir="./results",
    max_seq_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    save_steps=100,
    push_to_hub=True,  # Automatically pushes checkpoints to HF
    hub_model_id=output_name,
    gradient_checkpointing=True,
    # Required for modern versions
    gradient_checkpointing_kwargs={"use_reentrant": False},
)


def formatting_prompts_func(examples):
    output_texts = []
    # Loop through each list of messages in the batch
    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]
        # Use apply_chat_template to handle system, user, and assistant roles
        # tokenize=False returns a string instead of IDs
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        output_texts.append(formatted_text)
    return output_texts


# When you call the trainer, just pass this function:
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    args=sft_config,
)

trainer.train()

# 5. FINAL PUSH
trainer.push_to_hub()
