import torch
import pandas as pd
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from langdetect import detect

from config import DEVICE, TRAINING_CONFIG, OUTPUT_DIR_ROBERTA, SAVE_PATH_ROBERTA
from model_loader import load_roberta_for_classification

def train_category_classifier(goal_df):
    """
    Fine-tune RoBERTa cho category classification với LoRA
    
    Args:
        goal_df: DataFrame chứa columns ['clause', 'category']
    
    Returns:
        model, tokenizer, label2id, id2label
    """
    # Preprocess data
    goal_df["sentence_original"] = goal_df["sentence_original"].astype(str).str.strip()
    goal_df["word_count"] = goal_df["sentence_original"].apply(lambda x: len(x.split()))
    
    removed_short = goal_df[goal_df["word_count"] <= 1]
    count_removed_short = len(removed_short)
    
    goal_df = goal_df[goal_df["word_count"] > 1]
    goal_df = goal_df[goal_df["sentence_original"].apply(lambda x: detect(x) == "en")]
    
    print(f"Số câu có ≤ 1 từ bị loại: {count_removed_short}")
    print(f"goal_df: {len(goal_df)} mẫu")
    
    # Split train/test
    train_df, eval_df = train_test_split(
        goal_df, 
        test_size=0.1, 
        random_state=42, 
        stratify=goal_df["category"]
    )
    
    # Create label mappings
    label_list = sorted(goal_df["category"].unique().tolist())
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    
    # Load model
    model, tokenizer = load_roberta_for_classification(
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label
    )
    
    # Encode function
    def encode_fn(batch):
        enc = tokenizer(batch["clause"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = label2id[batch["category"]]
        return enc
    
    # Create datasets
    train_ds = Dataset.from_pandas(train_df).map(encode_fn)
    eval_ds = Dataset.from_pandas(eval_df).map(encode_fn)
    
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR_ROBERTA,
        **TRAINING_CONFIG,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    
    print("\n🚀 Bắt đầu fine-tuning với GOAL dataset ...")
    trainer.train()
    
    # Save model
    model.save_pretrained(SAVE_PATH_ROBERTA)
    tokenizer.save_pretrained(SAVE_PATH_ROBERTA)
    
    print(f"\n✅ Fine-tuning hoàn tất! Model đã được lưu tại: {SAVE_PATH_ROBERTA}")
    
    return model, tokenizer, label2id, id2label

def extract_category(clauses, model, tokenizer, id2label):
    """
    Predict category cho mỗi clause
    
    Args:
        clauses: List of dict chứa clause information
        model: Trained classification model
        tokenizer: Tokenizer
        id2label: Mapping từ label id sang label name
    
    Returns:
        clauses with 'category' field added
    """
    model.eval()
    model.to(DEVICE)

    for c in clauses:
        text = str(c.get("clause", c.get("sentence_original", ""))).strip()
        if text == "":
            c["category"] = "Unknown"
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(DEVICE)

        try:
            with torch.no_grad():
                outputs = model(**inputs)
                pred_id = torch.argmax(outputs.logits, dim=1).item()
                c["category"] = id2label[pred_id]
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý clause='{text}': {e}")
            c["category"] = "Unknown"

    return clauses