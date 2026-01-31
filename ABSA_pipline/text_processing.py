import re
from langdetect import detect
from model_loader import chat

def translate_to_english(sentence, model, tokenizer, max_new_tokens=200):
    """
    Dịch một câu sang tiếng Anh nếu nó KHÔNG phải tiếng Anh.
    - Nếu là tiếng Anh rồi thì giữ nguyên.
    - Nếu không phải, dịch sang tiếng Anh bằng LLM.
    """
    if not isinstance(sentence, str) or sentence.strip() == "":
        return ""

    text = sentence.strip()

    def is_english(text):
        try:
            return detect(text) == "en"
        except:
            return False

    if not is_english(text):
        prompt = f"""
You are a professional translator.

Task: Translate the following text into natural English, preserving its meaning.
Do not explain, comment, or add anything else.

Text:
{text}

Translation:
"""
        messages = [{"role": "user", "content": prompt}]
        translated = chat(model, tokenizer, messages, max_new_tokens=max_new_tokens).strip()
        return translated

    return text

def clean_text(text):
    """Làm sạch văn bản"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)          # Bỏ link
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # Bỏ ký tự đặc biệt
    text = re.sub(r"\s+", " ", text).strip()     # Gộp khoảng trắng
    return text

def normalize(text):
    """Normalize text về lowercase và strip"""
    return text.lower().strip()