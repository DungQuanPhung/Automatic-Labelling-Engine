# Web Application - ABSA Intelligence Dashboard

Web application cho hệ thống Aspect-Based Sentiment Analysis (ABSA) sử dụng FastAPI backend và giao diện web hiện đại.

## 📋 Tổng quan

Web app này cung cấp giao diện người dùng trực quan để phân tích cảm xúc theo khía cạnh (ABSA) cho domain khách sạn. Hệ thống sử dụng các mô hình AI tiên tiến như Qwen, RoBERTa, và DeBERTa để trích xuất và phân loại các khía cạnh, ý kiến, và cảm xúc từ đánh giá của khách hàng.

## 🏗️ Kiến trúc

### Backend (FastAPI)
- **API Server**: Xử lý requests và responses
- **ABSA Pipeline**: Xử lý logic phân tích chính
- **Model Management**: Quản lý và load các mô hình AI

### Frontend (HTML/CSS/JavaScript)
- Giao diện dashboard hiện đại với hiệu ứng trực quan
- Real-time analysis results
- Word cloud visualization
- Statistics và metrics display

## 📁 Cấu trúc Project

```
Web_app/
├── api.py                              # FastAPI server và endpoints
├── index.html                          # Frontend dashboard
├── config.py                           # Cấu hình mô hình và parameters
├── pipeline_ABSA.py                    # Pipeline ABSA chính
├── Extract_Clause_Term.py              # Trích xuất clauses và terms
├── Extract_Opinion.py                  # Trích xuất opinions
├── Extract_Category.py                 # Phân loại category
├── Extract_Polarity.py                 # Phát hiện polarity
├── Calculate_Metrics_Evaluation.py     # Đánh giá metrics
├── requirements.txt                    # Dependencies
└── README.md                          # Documentation
```

## 🚀 Cài đặt

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Tải Models

Đảm bảo các models sau được đặt trong thư mục `Web_app/`:

- **Qwen Model**: `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`
- **Category Model**: `roberta_lora_category_goal/`
- **Polarity Model**: `deberta_lora_polarity_goal_distilbert/`

## ▶️ Chạy Application

### Khởi động Server

```bash
cd Web_app
python api.py
```

Server sẽ chạy tại: `http://localhost:8000`

### Mở Web Interface

Mở file `index.html` trong trình duyệt hoặc truy cập:
```
http://localhost:8000
```

## 📖 API Endpoints

### 1. Health Check
```
GET /
```
Response:
```json
{
  "message": "Hotel ABSA Engine is Running",
  "status": "ready"
}
```

### 2. Analyze Text
```
POST /analyze
Content-Type: application/json

{
  "text": "The room was clean and spacious but the service was slow."
}
```

Response:
```json
[
  {
    "clause": "The room was clean and spacious",
    "Term": "room",
    "Opinion": "clean and spacious",
    "Category": "Facility",
    "Category Score": 0.95,
    "Polarity": "Positive",
    "Polarity Score": 0.98
  },
  {
    "clause": "the service was slow",
    "Term": "service",
    "Opinion": "slow",
    "Category": "Service",
    "Category Score": 0.92,
    "Polarity": "Negative",
    "Polarity Score": 0.87
  }
]
```

### 3. Upload File
```
POST /upload
Content-Type: multipart/form-data

file: reviews.csv (hoặc .txt)
```

Response: JSON với kết quả phân tích cho từng review

## 🔧 Configuration

File `config.py` chứa các cấu hình quan trọng:

### Model Paths
```python
# Actual implementation uses os.path.join with BASE_DIR
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "roberta_lora_category_goal")
POLARITY_MODEL_PATH = os.path.join(BASE_DIR, "deberta_lora_polarity_goal_distilbert")
```

### Batch Sizes
```python
CATEGORY_BATCH_SIZE = 32
POLARITY_BATCH_SIZE = 64
OPINION_BATCH_SIZE = 32
```

### Token Limits
```python
MAX_NEW_TOKENS_OPINION = 128
MAX_NEW_TOKENS_SPLIT = 256
```

### Hardware Settings
```python
QWEN_N_GPU_LAYERS = -1      # Sử dụng tất cả GPU layers
QWEN_N_CTX = 2048           # Context window size
QWEN_N_THREADS = 8          # CPU threads
DEVICE = "cuda"             # cuda hoặc cpu
```

## 💡 Modules Chi tiết

### 1. **api.py** - FastAPI Server
- Khởi tạo FastAPI app với CORS middleware
- Định nghĩa `ABSA_Engine` class để quản lý models
- Cung cấp endpoints cho text analysis và CSV upload
- Xử lý errors và exceptions

### 2. **pipeline_ABSA.py** - ABSA Pipeline
- `load_all_models()`: Load tất cả models cần thiết
- `load_qwen_model()`: Load Qwen GGUF model
- `load_category_model()`: Load RoBERTa category classifier
- `load_polarity_model()`: Load DeBERTa polarity detector
- `run_full_pipeline()`: Chạy pipeline đầy đủ từ text đến kết quả

### 3. **Extract_Clause_Term.py**
- Split câu thành clauses
- Trích xuất aspect terms từ mỗi clause
- Sử dụng Qwen LLM với prompt engineering

### 4. **Extract_Opinion.py**
- Trích xuất opinion expressions cho mỗi term
- Context-aware extraction
- Batch processing support

### 5. **Extract_Category.py**
- Phân loại category cho clauses
- Categories: Service, Amenity, Facility, Experience
- Sử dụng RoBERTa fine-tuned với LoRA
- Trả về category và confidence score

### 6. **Extract_Polarity.py**
- Phát hiện polarity (Positive/Negative/Neutral)
- Sử dụng DeBERTa fine-tuned model
- Trả về polarity label và confidence score

### 7. **Calculate_Metrics_Evaluation.py**
- Tính toán metrics đánh giá
- F1-score, Precision, Recall
- Support cho evaluation tasks

### 8. **index.html** - Frontend Dashboard
Features:
- 🎨 Modern UI với gradient effects
- 📊 Real-time analysis results
- ☁️ Word cloud visualization
- 📈 Statistics display
- 🎯 Category và polarity filters
- 📤 CSV upload support
- 💾 Export results

## 📊 Output Format

Kết quả phân tích bao gồm các trường:

| Field | Description |
|-------|-------------|
| `clause` | Clause được trích xuất từ câu |
| `Term` | Aspect term (ví dụ: "room", "service") |
| `Opinion` | Opinion expression (ví dụ: "clean", "slow") |
| `Category` | Category (Service/Amenity/Facility/Experience) |
| `Category Score` | Confidence score cho category (0-1) |
| `Polarity` | Polarity (Positive/Negative/Neutral) |
| `Polarity Score` | Confidence score cho polarity (0-1) |

## 🎯 Use Cases

### 1. Single Text Analysis
Phân tích một câu review đơn lẻ để hiểu chi tiết các aspects và cảm xúc.

### 2. Batch CSV Processing
Upload file CSV chứa nhiều reviews để phân tích hàng loạt.

### 3. Real-time Monitoring
Tích hợp vào hệ thống để phân tích real-time customer feedback.

### 4. Statistical Analysis
Tổng hợp thống kê về các categories và polarities để có insights.

## 🔍 Examples

### Example 1: Positive Review
```
Input: "The hotel room was very clean and the staff were extremely helpful."

Output:
- Clause: "The hotel room was very clean"
  - Term: room
  - Opinion: very clean
  - Category: Facility
  - Polarity: Positive

- Clause: "the staff were extremely helpful"
  - Term: staff
  - Opinion: extremely helpful
  - Category: Service
  - Polarity: Positive
```

### Example 2: Mixed Review
```
Input: "Great location but the WiFi was terrible."

Output:
- Clause: "Great location"
  - Term: location
  - Opinion: Great
  - Category: Facility
  - Polarity: Positive

- Clause: "the WiFi was terrible"
  - Term: WiFi
  - Opinion: terrible
  - Category: Amenity
  - Polarity: Negative
```

## ⚙️ Advanced Configuration

### Tối ưu Performance

1. **GPU Memory**:
   ```python
   QWEN_N_GPU_LAYERS = -1  # Use all layers on GPU
   ```

2. **Batch Processing**:
   ```python
   CATEGORY_BATCH_SIZE = 32  # Increase for faster processing
   POLARITY_BATCH_SIZE = 64
   ```

3. **Context Window**:
   ```python
   QWEN_N_CTX = 2048  # Increase for longer texts
   ```

### Debug Mode
```python
DEBUG_MODE = True  # Enable verbose logging
```

## 🐛 Troubleshooting

### Issue: Models không load được
**Solution**: 
- Kiểm tra đường dẫn models trong `config.py`
- Đảm bảo models đã được tải và đặt đúng thư mục

### Issue: Out of memory
**Solution**:
- Giảm `BATCH_SIZE` trong config
- Giảm `QWEN_N_GPU_LAYERS` nếu GPU memory không đủ
- Sử dụng quantization (đã enable 4bit)

### Issue: Slow inference
**Solution**:
- Tăng `QWEN_N_GPU_LAYERS` để sử dụng GPU nhiều hơn
- Tăng batch sizes để xử lý nhiều items cùng lúc
- Đảm bảo `USE_FAST_TOKENIZER = True`

### Issue: CORS errors
**Solution**:
- CORS đã được cấu hình trong `api.py`
- Nếu vẫn có lỗi, kiểm tra browser console

## 📝 Notes

- Web app hỗ trợ cả Vietnamese và English inputs
- Tự động dịch sang tiếng Anh nếu cần thiết
- Models được load một lần khi server khởi động
- Hỗ trợ batch processing với auto-save
- Real-time progress updates cho batch processing

## 🔐 Security Notes

- Không expose sensitive model files
- Validate input trước khi xử lý
- Limit file upload sizes
- Sanitize output trước khi trả về client

## 📚 Dependencies

Các dependencies chính:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `transformers`: HuggingFace models
- `torch`: PyTorch
- `llama-cpp-python`: Qwen GGUF inference
- `peft`: LoRA fine-tuning
- `pandas`: Data processing
- `sentence-transformers`: Embeddings

Xem đầy đủ trong `requirements.txt`

## 🤝 Contributing

Khi thêm features mới:
1. Tuân thủ cấu trúc code hiện tại
2. Update API documentation
3. Test thoroughly với các edge cases
4. Update README nếu cần

## 📄 License

Xem LICENSE file trong repository chính.

---

**Developed for Hotel Review Analysis**
*Powered by Qwen, RoBERTa, and DeBERTa*
