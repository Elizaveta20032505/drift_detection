FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Устанавливаем PYTHONPATH для импортов
ENV PYTHONPATH=/app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


