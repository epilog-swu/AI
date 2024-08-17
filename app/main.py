import numpy as np
from tflite_runtime.interpreter import Interpreter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from mangum import Mangum
from sklearn.preprocessing import StandardScaler
from keras_preprocessing.sequence import pad_sequences

app = FastAPI()

# TensorFlow Lite 모델 로드
interpreter = Interpreter(model_path="./static/LSTM_model/model_1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    # 데이터 전처리 (정규화 포함)
    data = np.array([[item.accX, item.accY, item.accZ, item.gyroX, item.gyroY, item.gyroZ] for item in data])

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    max_sequence_length = 100
    padded_data = pad_sequences([data], maxlen=max_sequence_length, padding='post', dtype='float32')

    return padded_data

@app.get("/")
def read_root():
    return {"Message": "Hello World"}

@app.post("/api/ai/predict")
async def predict(item: RequestBody):
    try:
        input_data = preprocess_input(item.fall)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        result = bool(prediction[0] > 0.5)

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

handler = Mangum(app)
