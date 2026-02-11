FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.server.txt ./requirements.server.txt
RUN pip install --no-cache-dir -r requirements.server.txt

COPY src ./src
COPY config ./config
COPY frontend ./frontend
COPY scripts ./scripts

RUN mkdir -p /app/data

EXPOSE 8088

CMD ["python", "src/mcp_server_http.py", "--host", "0.0.0.0", "--port", "8088"]
