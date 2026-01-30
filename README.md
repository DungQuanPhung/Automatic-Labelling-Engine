# ABSA (Aspect-Based Sentiment Analysis) Pipeline

Hệ thống phân tích cảm xúc theo khía cạnh cho domain khách sạn sử dụng LLM (Qwen/Phi-3) và RoBERTa.

## 📁 Cấu trúc Project

```
absa_project/
├── config.py                  # Cấu hình chung
├── model_loader.py           # Load và khởi tạo models
├── text_processing.py        # Xử lý văn bản
├── term_extraction.py        # Trích xuất terms/aspects
├── opinion_extraction.py     # Trích xuất opinions
├── category_classifier.py    # Phân loại category
├── polarity_detector.py      # Phát hiện polarity
├── absa_pipeline.py         # Pipeline chính
├── evaluation.py            # Đánh giá kết quả
├── main.py                  # File chính để chạy
├── requirements.txt         # Dependencies
└── README.md               # File này
```

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
```

## 📖 Chức năng từng module

### 1. **config.py**
Chứa tất cả cấu hình:
- Model IDs (Qwen, Phi-3, RoBERTa)
- LoRA configurations
- Training parameters
- Device settings

### 2. **model_loader.py**
Load các models:
- `load_qwen_model()`: Load Qwen với LoRA
- `load_phi3_model()`: Load Phi-3 với LoRA
- `load_roberta_for_classification()`: Load RoBERTa cho classification
- `load_polarity_classifier()`: Load model phát hiện polarity
- `chat()` và `chat_phi3()`: Hàm chat với LLM

### 3. **text_processing.py**
Xử lý văn bản:
- `translate_to_english()`: Dịch sang tiếng Anh
- `clean_text()`: Làm sạch văn bản
- `normalize()`: Chuẩn hóa text

### 4. **term_extraction.py**
Trích xuất terms:
- `split_sentence_with_terms_llm()`: Split câu và extract terms
- `extract_terms_only_from_sentence()`: Extract terms không split (Qwen)
- `extract_terms_only_from_sentence_phi()`: Extract terms không split (Phi-3)

### 5. **opinion_extraction.py**
Trích xuất opinions:
- `extract_opinions_only_from_clauses()`: Extract opinion expressions

### 6. **category_classifier.py**
Phân loại category:
- `train_category_classifier()`: Fine-tune RoBERTa
- `extract_category()`: Predict category cho clauses

### 7. **polarity_detector.py**
Phát hiện polarity:
- `detect_polarity()`: Phát hiện Positive/Negative/Neutral

### 8. **absa_pipeline.py**
Pipeline chính:
- `absa_pipeline()`: Pipeline đầy đủ với category classification
- `absa_pipeline_batch_save()`: Pipeline đơn giản, auto-save theo batch

### 9. **evaluation.py**
Đánh giá kết quả:
- `exact_match_f1()`: Exact match F1
- `token_f1()`: Token-level F1
- `rouge_l()`: ROUGE-L score
- `embedding_similarity()`: Embedding similarity
- `evaluate_absa()`: Đánh giá toàn diện
- `analyze_errors()`: Phân tích lỗi

## 💡 Cách sử dụng

### 1. Training Category Classifier

```python
from main import main_train_category_model

model_cat, tokenizer_cat, label2id, id2label = main_train_category_model()
```

### 2. Chạy Full Pipeline với Qwen

```python
from main import main_run_pipeline_with_qwen

main_run_pipeline_with_qwen()
```

### 3. Chạy Full Pipeline với Phi-3

```python
from main import main_run_pipeline_with_phi3

main_run_pipeline_with_phi3()
```

### 4. Chạy Simple Pipeline (không category)

```python
from main import main_simple_pipeline

main_simple_pipeline()
```

### 5. Test với một câu

```python
from main import main_single_sentence_test

main_single_sentence_test()
```

### 6. Đánh giá kết quả

```python
from main import main_evaluate_results

results, errors = main_evaluate_results()
```

## 🔧 Custom Usage

### Sử dụng từng component riêng lẻ:

```python
from model_loader import load_phi3_model
from term_extraction import extract_terms_only_from_sentence_phi
from opinion_extraction import extract_opinions_only_from_clauses

# Load model
model, tokenizer = load_phi3_model()

# Extract terms
sentence = "The room was clean and comfortable."
clauses = extract_terms_only_from_sentence_phi(sentence, model, tokenizer)

# Extract opinions
clauses = extract_opinions_only_from_clauses(clauses, model, tokenizer)

# Xem kết quả
for c in clauses:
    print(f"Term: {c['term']}, Opinion: {c['opinion']}")
```

## 📊 Output Format

Kết quả được lưu dưới dạng CSV với các columns:
- `sentence_index`: Chỉ số câu
- `sentence_original`: Câu gốc
- `clause`: Clause được trích xuất
- `term`: Term/aspect
- `opinion`: Opinion expression
- `category`: Category (Service, Amenity, Facility, Experience)
- `polarity`: Polarity (Positive, Negative, Neutral)
- `polarity_score`: Confidence score

## 🎯 Metrics

### Non-discrete (Term, Opinion):
- Exact Match F1
- Token F1
- ROUGE-L
- Embedding Similarity

### Discrete (Category, Polarity):
- Accuracy
- Macro F1
- Precision
- Recall

## ⚙️ Configuration

Có thể thay đổi cấu hình trong `config.py`:

```python
# Model selection
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PHI_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# LoRA parameters
QWEN_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 16,
    ...
}

# Training parameters
TRAINING_CONFIG = {
    "num_train_epochs": 50,
    "learning_rate": 1.5e-4,
    ...
}
```

## 📝 Notes

- Pipeline tự động dịch câu sang tiếng Anh nếu cần
- Hỗ trợ auto-save theo batch để tránh mất dữ liệu
- Có thể chọn split hoặc không split câu thành clauses
- Hỗ trợ cả Qwen và Phi-3 models

## 🐛 Troubleshooting

### Out of Memory:
- Giảm `batch_size` trong config
- Sử dụng `gradient_accumulation_steps` lớn hơn

### Model loading errors:
- Kiểm tra GPU availability
- Verify model IDs trong config
- Đảm bảo đã cài đủ dependencies