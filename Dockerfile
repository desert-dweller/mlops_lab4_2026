FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the API code and Pydantic schemas
COPY serve.py .
COPY schemas.py .

# Copy the MLflow tracking data and SQLite database
COPY mlruns/ ./mlruns/
COPY mlflow.db .

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]