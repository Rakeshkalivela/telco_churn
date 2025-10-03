FROM python:3.12-slim
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501"]
