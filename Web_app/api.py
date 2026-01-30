from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import io
import random
import os

# 1. Import trực tiếp pipeline xử lý
import pipeline_ABSA as absa_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Phải False khi dùng allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. ĐỊNH NGHĨA ENGINE (Đã sửa logic thêm Score)
class ABSA_Engine:
    def __init__(self):
        print("⏳ Đang tải các model AI...")
        try:
            self.models = absa_pipeline.load_all_models()
            print("✅ Đã tải xong model.")
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")
            self.models = None

    def predict(self, text):
        if self.models is None:
            return []
            
        try:
            # Gọi pipeline gốc để lấy kết quả (DataFrame)
            df = absa_pipeline.run_full_pipeline(text, self.models)
            
            # Chuyển đổi DataFrame sang danh sách Dictionary
            records = df.to_dict(orient="records")
            
            # Đảm bảo các trường số được chuyển thành float và xử lý NaN
            for item in records:
                # Xử lý Category Score
                if 'Category Score' in item:
                    if pd.isna(item['Category Score']):
                        item['Category Score'] = 0.0
                    else:
                        item['Category Score'] = float(item['Category Score'])
                else:
                    item['Category Score'] = 0.0

                # Xử lý Polarity Score
                if 'Polarity Score' in item:
                    if pd.isna(item['Polarity Score']):
                        item['Polarity Score'] = 0.0
                    else:
                        item['Polarity Score'] = float(item['Polarity Score'])
                else:
                    item['Polarity Score'] = 0.0
                
                # Đảm bảo các trường text không bị None
                if 'Term' not in item or pd.isna(item.get('Term')):
                    item['Term'] = ""
                if 'Opinion' not in item or pd.isna(item.get('Opinion')):
                    item['Opinion'] = ""
                if 'Category' not in item or pd.isna(item.get('Category')):
                    item['Category'] = ""
                if 'Polarity' not in item or pd.isna(item.get('Polarity')):
                    item['Polarity'] = ""

            return records
            
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            return []

# Khởi tạo model
model = ABSA_Engine()

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Hotel ABSA Engine is Running", "status": "ready"}

@app.post("/analyze")
def analyze(data: InputText):
    if model.models is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải thành công.")
    result = model.predict(data.text)
    return result

# ============================================================
# PHẦN UPLOAD FILE
# ============================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if model.models is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải thành công.")

    if file.filename is None:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ.")

    contents = await file.read()
    results_list = []
    
    try:
        # Xử lý file TXT
        if file.filename.endswith('.txt'):
            text_content = contents.decode('utf-8')
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            
            # Xử lý tối đa 200 dòng đầu
            for text in lines[:200]:
                if len(text) > 2:
                    try:
                        aspects = model.predict(text)
                        for aspect in aspects:
                            aspect['Original Sentence'] = text
                        
                        results_list.append({
                            "Review": text,
                            "Aspects": aspects
                        })
                    except Exception as e:
                        print(f"Lỗi xử lý dòng: {e}")
                        continue
            
            return {"results": results_list}
        
        # Xử lý file CSV/Excel
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .csv, .xlsx hoặc .txt")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Lỗi encoding file. Vui lòng sử dụng UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {str(e)}")

    # Tìm cột Review cho CSV/Excel
    possible_columns = ['Review', 'review', 'Text', 'text', 'Content', 'content']
    target_col = None
    for col in possible_columns:
        if col in df.columns:
            target_col = col
            break
    
    if not target_col:
        target_col = df.columns[0]

    # Xử lý 200 dòng đầu
    df_processed = df.head(200).fillna("") 

    for index, row in df_processed.iterrows():
        text = str(row[target_col]).strip()
        if len(text) > 2: 
            try:
                # Gọi hàm predict (đã có logic tự thêm score ở trên)
                aspects = model.predict(text)
                
                # Gắn thêm câu gốc
                for aspect in aspects:
                    aspect['Original Sentence'] = text
                
                results_list.append({
                    "Review": text,
                    "Aspects": aspects
                })
            except Exception as e:
                print(f"Lỗi dòng {index}: {e}")
                continue

    return {"results": results_list}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)