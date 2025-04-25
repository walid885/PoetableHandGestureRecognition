FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/req.txt .
RUN pip install --no-cache-dir -r req.txt

# Create output directory for demo frames
RUN mkdir -p /app/output

COPY . .

CMD ["python", "app/script.py"]