# 스마트워치와 AI를 활용한 낙상 감지 및 혈당 관리 서비스 Dialog
![ezgif-3-2446187e13](https://github.com/user-attachments/assets/7f6bb330-65dd-41ef-8882-7edb91489c0e)

## 사용 기술
> Python, tensorflow keras LSTM <br/>
> fastAPI, Docker, AWS ECR, AWS Lambda

## 프로젝트 실행 방법
깃 레포지토리 클론
```
git clone
cd AI
```
파이썬 가상환경 세팅
```
python3 -m venv venv
source venv/bin/activate
```
라이브러리 설치
```
pip install -r /app/requirements.txt
```
서버 실행
```
uvicorn main:app -reload
```

## 시스템 아키텍처
![lambda](https://github.com/user-attachments/assets/3a13b67d-7992-4fb5-a415-8e72a977b223)


## 디렉토리 구조
```
📁 Dialog
├── Dockerfile
└── app
    ├── main.py
    ├── requirements.txt
    └── static
        ├── LSTM_model
        ├── images
        ├── learning
        └── training_data
```
