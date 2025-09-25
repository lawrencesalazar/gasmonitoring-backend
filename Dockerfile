# Use Debian slim base (always available on Render)
FROM debian:bookworm-slim

# Set working directory
WORKDIR /app

# Install Python, pip, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make "python" command point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Render’s default port
EXPOSE 8080

# Run with Gunicorn + Uvicorn worker
CMD exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT}
