# ── Dockerfile ───────────────────────────────────────────────
FROM python:3.11-slim

# Install system tools (optional: remove build deps later)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "gui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
