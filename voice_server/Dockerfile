FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 60"]
