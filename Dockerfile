FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the two calculators and their bundled theme and fonts
COPY futures.py options.py ./
COPY assets/ ./assets/

# Default command (docker-compose sets the command for each calculator)
CMD ["python", "-u", "futures.py"]
