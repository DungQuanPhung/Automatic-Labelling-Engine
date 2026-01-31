import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model_loader import load_embedding_model
from text_processing import normalize

def exact_match_f1(pred_list, gold_list):
    """Exact match F1 score"""
    tp = sum([1 for g, p in zip(gold_list, pred_list) if g == p])
    fp = sum([1 for g, p in zip(gold_list, pred_list) if g != p])
    fn = fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def token_f1(pred_list, gold_list):
    """Token-level F1 score"""
    tp = 0
    for g, p in zip(gold_list, pred_list):
        g_tokens = set(g.split())
        p_tokens = set(p.split())
        if len(g_tokens & p_tokens) > 0:
            tp += 1
    fp = len(pred_list) - tp
    fn = len(gold_list) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def rouge_l(pred_list, gold_list):
    """ROUGE-L score"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    f_scores = []
    for g, p in zip(gold_list, pred_list):
        f_scores.append(scorer.score(g, p)['rougeL'].fmeasure)
    return sum(f_scores) / len(f_scores) if f_scores else 0.0

def embedding_similarity(pred_list, gold_list, model):
    """Cosine similarity của sentence embeddings"""
    sims = []
    for g, p in zip(gold_list, pred_list):
        emb_g = model.encode(g)
        emb_p = model.encode(p)
        sims.append(cosine_similarity([emb_g], [emb_p])[0][0])
    return sum(sims) / len(sims) if sims else 0.0

def evaluate_absa(df, pred_prefix='pred_'):
    """
    Đánh giá toàn diện kết quả ABSA
    
    Args:
        df: DataFrame chứa cả ground truth và predictions
            Columns: term, opinion, category, polarity, pred_term, pred_opinion, pred_category, pred_polarity
        pred_prefix: Prefix cho các cột prediction
    
    Returns:
        dict chứa tất cả metrics
    """
    # Normalize text columns
    df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
    
    # Initialize embedding model
    embedding_model = load_embedding_model()
    
    results = {}
    
    # ===== Non-discrete metrics (Term & Opinion) =====
    for key in ['term', 'opinion']:
        gold_list = [normalize(str(t)) for t in df[key]]
        pred_list = [normalize(str(t)) for t in df[pred_prefix + key]]
        
        results[f"{key}_exact_match_f1"] = exact_match_f1(pred_list, gold_list)
        results[f"{key}_token_f1"] = token_f1(pred_list, gold_list)
        results[f"{key}_rouge_l"] = rouge_l(pred_list, gold_list)
        results[f"{key}_embedding_sim"] = embedding_similarity(pred_list, gold_list, embedding_model)
    
    # ===== Discrete metrics (Category & Polarity) =====
    for key in ['category', 'polarity']:
        y_true = df[key]
        y_pred = df[pred_prefix + key]
        
        results[f"{key}_accuracy"] = accuracy_score(y_true, y_pred)
        results[f"{key}_macro_f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results[f"{key}_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results[f"{key}_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return results

def print_evaluation_results(results):
    """In kết quả đánh giá dạng table"""
    non_disc_metrics = []
    disc_metrics = []
    
    for key, value in results.items():
        if 'term' in key or 'opinion' in key:
            metric_name = key.replace('_', ' ').title()
            non_disc_metrics.append([metric_name, round(value, 4)])
        else:
            metric_name = key.replace('_', ' ').title()
            disc_metrics.append([metric_name, round(value, 4)])
    
    non_disc_df = pd.DataFrame(non_disc_metrics, columns=["Metric", "Score"])
    disc_df = pd.DataFrame(disc_metrics, columns=["Metric", "Score"])
    
    print("📌 Non-discrete Metrics (Term & Opinion)\n")
    print(non_disc_df.to_string(index=False))
    print("\n🎯 Discrete Classification Metrics (Category & Polarity)\n")
    print(disc_df.to_string(index=False))
    
    return non_disc_df, disc_df

def analyze_errors(df, pred_prefix='pred_'):
    """
    Phân tích các lỗi prediction
    
    Args:
        df: DataFrame chứa predictions
        pred_prefix: Prefix cho prediction columns
    
    Returns:
        dict chứa thông tin về errors
    """
    errors = {}
    
    for key in ['category', 'polarity']:
        wrong = df[df[key] != df[pred_prefix + key]]
        errors[f"{key}_wrong_count"] = len(wrong)
        errors[f"{key}_total_count"] = len(df)
        errors[f"{key}_accuracy"] = 100 - (len(wrong) / len(df) * 100)
        errors[f"{key}_wrong_samples"] = wrong[[key, pred_prefix + key, 'clause']].head(10)
    
    return errors