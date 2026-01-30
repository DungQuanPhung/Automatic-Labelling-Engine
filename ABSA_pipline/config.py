import torch
from langdetect import DetectorFactory

# Seed cho reproducibility
DetectorFactory.seed = 0

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configurations
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PHI_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ROBERTA_MODEL_NAME = "roberta-base"
POLARITY_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LoRA configurations for Qwen
QWEN_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# LoRA configurations for Phi-3
PHI_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 16,
    "target_modules": ["qkv_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# LoRA configurations for RoBERTa
ROBERTA_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "value"],
    "lora_dropout": 0.05,
    "bias": "none",
}

# BitsAndBytes configuration
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16
}

# Training configurations
TRAINING_CONFIG = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 50,
    "learning_rate": 1.5e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.05,
    "logging_steps": 10,
}

# Pipeline configurations
MAX_NEW_TOKENS_TERM = 20
MAX_NEW_TOKENS_OPINION = 40
MAX_NEW_TOKENS_TRANSLATION = 200
MAX_NEW_TOKENS_CATEGORY = 40
BATCH_SIZE = 100
SAVE_EVERY = 100

# Paths
SAVE_PATH_ROBERTA = "/content/roberta_lora_category_goal"
OUTPUT_DIR_ROBERTA = "./roberta_lora_goal"