from typing import List, Dict, Any
from config import POLARITY_BATCH_SIZE

def detect_polarity(clauses: List[Dict[str, Any]], polarity_classifier, batch_size=POLARITY_BATCH_SIZE):
    """
    Dự đoán cảm xúc (Positive/Negative/Neutral) theo batch.
    Input: List dict chứa 'clause'.
    Output: List dict đã bổ sung 'polarity' và 'polarity_score'.
    """
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    if not clauses:
        return clauses

    # 1. Chuẩn bị dữ liệu đầu vào cho model
    texts = [str(c.get("clause", "")).strip() for c in clauses]
    
    # Lưu lại index của những câu hợp lệ (không rỗng)
    valid_indices = [i for i, t in enumerate(texts) if t]
    valid_texts = [texts[i] for i in valid_indices]
    
    # Gán giá trị mặc định cho các clause rỗng
    for i, text in enumerate(texts):
        if not text:
            clauses[i]["polarity"] = "Neutral"
            clauses[i]["polarity_score"] = 0.0

    if not valid_texts:
        return clauses

    try:
        predictions = polarity_classifier(valid_texts, batch_size=batch_size, truncation=True)
        for idx, pred in zip(valid_indices, predictions):
            # Xử lý format trả về của pipeline
            if isinstance(pred, list):
                pred = max(pred, key=lambda x: x['score'])
                
            raw_label = pred.get('label', 'Neutral')
            score = pred.get('score', 0.0)

            # Map label sang nhãn thực tế
            if raw_label.startswith('LABEL_'):
                label_id = int(raw_label.split('_')[1])
                label = id2label.get(label_id, "Neutral")
            else:
                # Model trả về trực tiếp 'Negative', 'Neutral', 'Positive'
                label = raw_label.capitalize()

            clauses[idx]["polarity"] = label if label else "Neutral"
            clauses[idx]["polarity_score"] = round(float(score), 4)
            
    except Exception as e:
        print(f"[Polarity Error] Batch processing failed: {e}")
        # Fallback: Gán giá trị mặc định nếu lỗi
        for i in valid_indices:
            clauses[i]["polarity"] = "Neutral"
            clauses[i]["polarity_score"] = 0.0
            
    return clauses