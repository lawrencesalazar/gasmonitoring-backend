# Start from Debian
FROM debian:bookworm-slim

# Set working directory
WORKDIR /app

# Install Python 3.11 + pip + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Verify version
RUN python --version && pip --version

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Render port
EXPOSE 8080

# Run with Gunicorn + Uvicorn worker
CMD exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}
