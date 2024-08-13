import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# 미리 학습된 모델 로드
model = tf.keras.models.load_model('model_1.keras')  # 모델 경로를 적절히 변경

# Pydantic 모델 정의
class SensorData(BaseModel):
    accX: float
    accY: float
    accZ: float
    gyroX: float
    gyroY: float
    gyroZ: float

class RequestBody(BaseModel):
    fall: List[SensorData]

def preprocess_input(data: List[SensorData]):
    data = np.array([[item.accX, item.accY, item.accZ, item.gyroX, item.gyroY, item.gyroZ] for item in data])
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    max_sequence_length = 100 
    padded_data = pad_sequences([data], maxlen=max_sequence_length, padding='post', dtype='float32')
    
    return padded_data

@app.post("/api/ai/predict")
async def predict(item: RequestBody):
    try:
        # 입력 데이터를 전처리
        input_data = preprocess_input(item.fall)

        # 모델로 예측 수행
        prediction = model.predict(input_data)

        # 예측 결과를 이진 분류 (True/False)로 변환
        result = bool(prediction[0] > 0.5)

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
