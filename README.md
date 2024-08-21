## 스마트워치와 AI를 활용한 낙상 감지 및 혈당 관리 서비스 Dialog
![ezgif-3-2446187e13](https://github.com/user-attachments/assets/7f6bb330-65dd-41ef-8882-7edb91489c0e)

## 사용 기술
> Python, tensorflow keras LSTM <br/>
> fastAPI, Docker, AWS ECR, AWS Lambda

## 프로젝트 실행 방법
깃 레포지토리 클론
```bash
git clone https://github.com/epilog-swu/AI.git
cd AI
```
파이썬 가상환경 세팅
```bash
python3 -m venv venv
source venv/bin/activate (mac) 
source venv/Scripts/activate (window)
```
라이브러리 설치
```bash
pip install -r /app/requirements.txt
```
서버 실행
```bash
uvicorn main:app -reload
```

## 시스템 아키텍처
![lambda](https://github.com/user-attachments/assets/3a13b67d-7992-4fb5-a415-8e72a977b223)

## 모델 구조 및 평가
<img src="https://github.com/user-attachments/assets/77049221-6b01-493e-a230-af659312aa99" height="300px" alert="모델 정보"> <img src="https://github.com/user-attachments/assets/2cb534b8-56c3-4420-bc2a-5f35640ed1f6" height="340px" /> </img>


## 디렉토리 구조
```
📁 AI
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
