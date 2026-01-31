import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sentence_transformers import SentenceTransformer

from config import *

def load_qwen_model():
    """Load Qwen model với LoRA configuration"""
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**QWEN_LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def load_phi3_model():
    """Load Phi-3 model với LoRA configuration"""
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    
    tokenizer = AutoTokenizer.from_pretrained(
        PHI_MODEL_ID,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**PHI_LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # Tắt cache
    model.config.use_cache = False
    model.generation_config.use_cache = False
    
    return model, tokenizer

def load_roberta_for_classification(num_labels, label2id, id2label):
    """Load RoBERTa model cho classification task"""
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    lora_config = LoraConfig(
        **ROBERTA_LORA_CONFIG,
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)
    
    return model, tokenizer

def load_polarity_classifier():
    """Load polarity classification pipeline"""
    return pipeline(
        "text-classification",
        model=POLARITY_MODEL,
        top_k=None,
        truncation=True
    )

def load_embedding_model():
    """Load sentence embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL)

def chat(model, tokenizer, messages, max_new_tokens=100):
    """Chat function cho Qwen model"""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

def chat_phi3(model, tokenizer, prompt, max_new_tokens=50, temperature=0.2, top_p=0.9):
    """Chat function cho Phi-3 model"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    model.config.use_cache = False
    model.generation_config.use_cache = False

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            use_cache=False
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )