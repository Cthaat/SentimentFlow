FROM python:3.11-slim
WORKDIR /workspace
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY sentiment_scale.py ordinal_loss.py ./
COPY BERT ./BERT
COPY training ./training
COPY backend ./backend
ENV PYTHONPATH=/workspace:/workspace/backend \
    SENTIMENTFLOW_PROJECT_ROOT=/workspace
WORKDIR /workspace/backend
EXPOSE 8846
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8846"]
