from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd

df = pd.read_csv("datasets/Test_Evaluation_6.csv")

# Chuyển tất cả labels về lowercase để so sánh
df['category'] = df['category'].str.lower()
df['pred_category'] = df['pred_category'].str.lower()
df['polarity'] = df['polarity'].str.lower()
df['pred_polarity'] = df['pred_polarity'].str.lower()

# Tính metrics cho Category
category_accuracy = accuracy_score(df['category'], df['pred_category'])
category_precision, category_recall, category_f1, _ = precision_recall_fscore_support(
    df['category'], df['pred_category'], average='weighted', zero_division=0
)

print("=== CATEGORY METRICS ===")
print(f"Accuracy: {category_accuracy:.4f}")
print(f"Precision: {category_precision:.4f}")
print(f"Recall: {category_recall:.4f}")
print(f"F1-Score: {category_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(df['category'], df['pred_category'], zero_division=0))

# Tính metrics cho Polarity
polarity_accuracy = accuracy_score(df['polarity'], df['pred_polarity'])
polarity_precision, polarity_recall, polarity_f1, _ = precision_recall_fscore_support(
    df['polarity'], df['pred_polarity'], average='weighted', zero_division=0
)

print("\n=== POLARITY METRICS ===")
print(f"Accuracy: {polarity_accuracy:.4f}")
print(f"Precision: {polarity_precision:.4f}")
print(f"Recall: {polarity_recall:.4f}")
print(f"F1-Score: {polarity_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(df['polarity'], df['pred_polarity'], zero_division=0))