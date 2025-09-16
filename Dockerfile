FROM ubuntu:latest
LABEL authors="Xavier"

ENTRYPOINT ["top", "-b"]
FROM python:3.9-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd -m -u 1000 appuser

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY gunicorn_conf.py ./
COPY static ./static

USER appuser

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:create_app()"]