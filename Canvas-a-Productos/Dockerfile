# Canvas-a-Productos — servicio en contenedor con Chrome + Selenium
FROM python:3.11-slim

# Instalar Chrome y dependencias (sin apt-key, deprecado en Debian reciente)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    unzip \
    ca-certificates \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Usuario no root
RUN adduser --disabled-password --gecos "" app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && chown -R app:app /app

COPY script.py app.py ./
RUN chown app:app script.py app.py

# Directorio de salida (volumen para persistir crops si se desea)
RUN mkdir -p /app/output && chown app:app /app/output

USER app
EXPOSE 3000

ENV PORT=3000
ENV OUTPUT_BASE=/app/output

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget -q -O - http://127.0.0.1:3000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
