# Use official Python 3.11 
FROM docker.io/library/python:3.11

# Set working directory
WORKDIR /app

# Install system-level deps (needed for matplotlib, pandas, shap, xgboost, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Render’s default port
EXPOSE 10000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
