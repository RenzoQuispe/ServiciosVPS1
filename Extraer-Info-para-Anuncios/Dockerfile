FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN adduser --disabled-password --gecos "" app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && chown -R app:app /app

COPY app.py ./
COPY scrapers/ ./scrapers/
COPY .env ./.env
RUN chown -R app:app /app

USER app
EXPOSE 3000

ENV PORT=3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget -q -O - http://127.0.0.1:3000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
