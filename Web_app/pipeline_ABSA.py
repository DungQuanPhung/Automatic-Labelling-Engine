import os
import pandas as pd
from typing import Dict, Any, Tuple, Union
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import torch
from llama_cpp import Llama
from config import (
    MAX_NEW_TOKENS_SPLIT,
    MAX_NEW_TOKENS_OPINION,
    QWEN_MODEL_PATH,
    CATEGORY_MODEL_PATH,
    POLARITY_MODEL_PATH,
    OPINION_BATCH_SIZE,
    CATEGORY_BATCH_SIZE,
    POLARITY_BATCH_SIZE,
    QWEN_N_CTX,
    QWEN_N_GPU_LAYERS,
    QWEN_N_THREADS,
    DEBUG_MODE,
    DEVICE,
    USE_FAST_TOKENIZER,
    TORCH_DTYPE
)
from peft import PeftModel, PeftConfig

from Extract_Clause_Term import split_and_term_extraction
from Extract_Opinion import extract_opinions_only_from_clauses
from Extract_Category import get_predicted_categories, DEFAULT_CATEGORY_LABELS
from Extract_Polarity import detect_polarity

# Import quantization config nếu có
try:
    from config import QWEN_LORA_QUANTIZATION
except ImportError:
    QWEN_LORA_QUANTIZATION = None

# ---------------- QWEN LOADER (HỖ TRỢ CẢ GGUF VÀ LORA) ---------------- #
def load_qwen_model() -> Tuple[Union[Llama, Any], Union[None, Any], str]:
    # Kiểm tra QWEN_MODEL_PATH là file hay folder
        model = Llama(
            model_path=QWEN_MODEL_PATH,
            n_gpu_layers=QWEN_N_GPU_LAYERS,
            n_ctx=QWEN_N_CTX,
            n_threads=QWEN_N_THREADS,
            verbose=DEBUG_MODE
        )
        return model, None, "gguf"

def load_category_model():
    '''
    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(CATEGORY_MODEL_PATH, use_fast=USE_FAST_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(
        CATEGORY_MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    '''
    adapter_path = CATEGORY_MODEL_PATH
    peft_cfg = PeftConfig.from_pretrained(adapter_path)

    base = AutoModelForSequenceClassification.from_pretrained(
        peft_cfg.base_model_name_or_path,
        num_labels=6,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    # optional: model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)

    return model, tokenizer, DEFAULT_CATEGORY_LABELS

# ---------------- POLARITY MODEL (DistilBERT LoRA) ---------------- #
def load_polarity_model():
    """
    Load polarity model đã fine-tune bằng LoRA.
    Adapter được train với 3 labels (Negative, Neutral, Positive).
    """
    adapter_path = POLARITY_MODEL_PATH
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    
    # Load base model với num_labels=3 để match với adapter đã train
    base = AutoModelForSequenceClassification.from_pretrained(
        peft_cfg.base_model_name_or_path,
        num_labels=3,  # Match với adapter (3 classes: Negative, Neutral, Positive)
        ignore_mismatched_sizes=True,  # Bỏ qua mismatch từ pretrained weights
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        peft_cfg.base_model_name_or_path,
        use_fast=USE_FAST_TOKENIZER
    )
    
    # Create pipeline
    polarity_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        device=DEVICE
    )
    return polarity_classifier

# ---------------- MASTER LOADER ---------------- #
def load_all_models():
    """Load tất cả models và cache."""
    qwen_model, qwen_tokenizer, qwen_model_type = load_qwen_model()
    cat_model, cat_tokenizer, cat_id2label = load_category_model()
    polarity_classifier = load_polarity_model()
    return {
        "qwen_model": qwen_model,
        "qwen_tokenizer": qwen_tokenizer,
        "qwen_model_type": qwen_model_type,
        "cat_model": cat_model,
        "cat_tokenizer": cat_tokenizer,
        "cat_id2label": cat_id2label,
        "polarity_classifier": polarity_classifier,
    }

# ---------------- FULL PIPELINE ---------------- #
def run_full_pipeline(sentence: str, models: Dict[str, Any]) -> pd.DataFrame:
    """Chạy toàn bộ pipeline ABSA trên 1 câu review."""
    sentence = (sentence or "").strip()
    if not sentence:
        return pd.DataFrame([])

    qwen_model = models["qwen_model"]
    qwen_tokenizer = models["qwen_tokenizer"]
    qwen_model_type = models.get("qwen_model_type", "gguf")
    cat_model = models["cat_model"]
    cat_tokenizer = models["cat_tokenizer"]
    cat_id2label = models["cat_id2label"]
    polarity_classifier = models["polarity_classifier"]

    # Bước 1 & 2: Tách Clause, Term và Opinion (Qwen)
    clauses_with_details = split_and_term_extraction(
        sentence,
        qwen_model,
        qwen_tokenizer,
        max_new_tokens=MAX_NEW_TOKENS_SPLIT,
        model_type=qwen_model_type
    )
    clauses_with_details = extract_opinions_only_from_clauses(
        clauses_with_details,
        qwen_model,
        qwen_tokenizer,
        max_new_tokens=MAX_NEW_TOKENS_OPINION,
        batch_size=OPINION_BATCH_SIZE,
        model_type=qwen_model_type
    )

    # Bước 3: Category (RoBERTa)
    clauses_categories = get_predicted_categories(
        clauses_with_details,
        cat_model,
        cat_tokenizer,
        cat_id2label,
        batch_size=CATEGORY_BATCH_SIZE
    )

    # Bước 4: Polarity (DeBERTa)
    final_results = detect_polarity(
        clauses_categories,
        polarity_classifier,
        batch_size=POLARITY_BATCH_SIZE
    )

    df = pd.DataFrame(final_results)
    
    # Map từ keys trong dict sang column names cho hiển thị
    column_mapping = {
        "clause": "Clause",
        "term": "Term", 
        "opinion": "Opinion",
        "category": "Category",
        "category_score": "Category Score",
        "polarity": "Polarity",
        "polarity_score": "Polarity Score",
        "sentence_original": "Original Sentence"
    }
    
    # Rename columns theo mapping
    df = df.rename(columns=column_mapping)
    
    # Đảm bảo thứ tự cột: Original Sentence ở cuối
    columns_order = [
        "Clause",
        "Term",
        "Opinion",
        "Category",
        "Category Score",
        "Polarity",
        "Polarity Score",
        "Original Sentence",
    ]
    final_columns = [c for c in columns_order if c in df.columns]
    return df[final_columns] if final_columns else df