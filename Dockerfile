# Dockerfile
FROM python:3.11-slim

# Cleaner Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 1. Install dependencies first (this layer is cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the app code and the promoted model into the image
COPY src/ ./src/
COPY models/ ./models/

# 3. The port the app listens on inside the container
EXPOSE 8000

# 4. What runs when the container starts
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]