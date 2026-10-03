FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --default-timeout=300 -r requirements-docker.txt

COPY api ./api
COPY src ./src
COPY models ./models
COPY app ./app

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]