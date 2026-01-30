import os
import torch

# ============== CẤU HÌNH MÔ HÌNH ==============

# Thư mục gốc của app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mô hình LLM chính (Qwen)
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")

# Mô hình Category (RoBERTa)
CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "roberta_lora_category_goal")

# Mô hình Polarity (DeBERTa)
POLARITY_MODEL_PATH = os.path.join(BASE_DIR, "deberta_lora_polarity_goal_distilbert")

# ============== CẤU HÌNH BATCH SIZE ==============

# Batch size cho các mô hình phân loại
CATEGORY_BATCH_SIZE = 32
POLARITY_BATCH_SIZE = 64
OPINION_BATCH_SIZE = 32
# ============== CẤU HÌNH TOKEN ==============

# Số token mới sinh ra tối đa cho từng task
MAX_NEW_TOKENS_OPINION = 128
MAX_NEW_TOKENS_SPLIT = 256
MAX_NEW_TOKENS_DEFAULT = 256

# ============== CẤU HÌNH QWEN GGUF ==============
QWEN_N_GPU_LAYERS = -1
QWEN_N_CTX = 2048

# Số threads CPU cho processing
QWEN_N_THREADS = 8

# Inference optimization
USE_FAST_TOKENIZER = True        # Dùng Rust-based fast tokenizers
PREFETCH_FACTOR = 2              # Prefetch data cho GPU
ENABLE_TORCH_COMPILE = False     # Tắt vì Triton không hỗ trợ tốt trên Windows
# Debug mode
DEBUG_MODE = True                # Bật để xem chi tiết logs

# Quantization cho LoRA model (giảm VRAM usage)
QWEN_LORA_QUANTIZATION = "4bit"  # Dùng 4bit để tiết kiệm VRAM

# Tự động chọn precision tối ưu (bfloat16 cho Ampere+, float16 cho cũ hơn)
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"