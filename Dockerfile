# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip

# Install critical packages first in correct order
RUN pip install numpy==1.24.3
RUN pip install scipy==1.11.4
RUN pip install scikit-learn==1.3.2
RUN pip install pandas==2.1.3
RUN pip install xgboost==2.0.0

# Install the rest from requirements
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:10000/health', timeout=2)"

# Run the application
CMD ["gunicorn", "main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:10000", "--timeout", "120", "--keep-alive", "5"]