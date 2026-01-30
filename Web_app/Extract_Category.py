import torch
from typing import List, Dict, Any
from config import CATEGORY_BATCH_SIZE
MAX_LENGTH = 128

# ĐỊNH NGHĨA MAP NHÃN CHUẨN (HARDCODED)
# Đảm bảo thứ tự này khớp với lúc bạn train model RoBERTa
DEFAULT_CATEGORY_LABELS = {
    0: "Amenity",
    1: "Branding",
    2: "Experience",
    3: "Facility",
    4: "Loyalty",
    5: "Service",
}

def get_predicted_categories(
    clauses: List[Dict[str, Any]],
    model,
    tokenizer,
    id2label: Dict[int, str] = None, # Tham số này có thể bỏ qua để dùng mặc định
    batch_size: int = CATEGORY_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    
    # Ưu tiên dùng map chuẩn đã định nghĩa ở trên
    final_id2label = DEFAULT_CATEGORY_LABELS

    if not clauses:
        return clauses

    device = next(model.parameters()).device
    model.eval()

    texts = []
    clause_indices = []

    for idx, clause in enumerate(clauses):
        text = str(clause.get("clause", "")).strip()
        if not text:
            clause["category"] = "Unknown"
            clause["category_score"] = 0.0
            continue
        texts.append(text)
        clause_indices.append(idx)

    if not texts:
        return clauses

    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_idx = clause_indices[start : start + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                ).to(device)

                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                scores, preds = torch.max(probs, dim=1)

                for idx_local, pred_id, score in zip(batch_idx, preds.tolist(), scores.tolist()):
                    # Map ID sang tên (VD: 5 -> Service)
                    # Xử lý trường hợp model trả về LABEL_5 thì parse lấy số 5
                    label_name = final_id2label.get(pred_id, "Unknown")
                    
                    clauses[idx_local]["category"] = label_name
                    clauses[idx_local]["category_score"] = float(score)

    except Exception as exc:
        print(f"[Category Error] {exc}")
        for idx in clause_indices:
            clauses[idx].setdefault("category", "Unknown")
            clauses[idx].setdefault("category_score", 0.0)

    return clauses