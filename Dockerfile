# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port (Render sets PORT env, but we declare for clarity)
EXPOSE 8080

# Start the app with Gunicorn + Uvicorn worker
CMD exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}
