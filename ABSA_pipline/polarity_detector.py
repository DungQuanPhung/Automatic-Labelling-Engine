from model_loader import load_polarity_classifier

def detect_polarity(clauses):
    """
    Detect polarity (Positive/Negative/Neutral) cho mỗi clause
    
    Args:
        clauses: List of dict chứa clause information
    
    Returns:
        clauses with 'polarity' and 'polarity_score' fields added
    """
    polarity_classifier = load_polarity_classifier()
    
    results = []
    for item in clauses:
        clause = str(item.get("clause", "")).strip()
        if clause == "":
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0
            results.append(item)
            continue

        try:
            res = polarity_classifier(clause)
            # Xử lý kết quả từ model
            if isinstance(res, list) and isinstance(res[0], list):
                res = res[0]
            top = max(res, key=lambda x: x["score"])
            item["polarity"] = top["label"].capitalize()
            item["polarity_score"] = round(top["score"], 4)
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý câu '{clause}': {e}")
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0

        results.append(item)
    
    return results