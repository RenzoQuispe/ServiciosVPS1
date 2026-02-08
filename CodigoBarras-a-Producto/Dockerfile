# CodigoBarras-a-Producto — servicio en contenedor
FROM node:20-alpine

WORKDIR /app

# wget para HEALTHCHECK (node:alpine no lo incluye por defecto)
RUN apk add --no-cache wget

# Usuario no root
RUN addgroup -g 1001 app && adduser -u 1001 -G app -D app

COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm install --omit=dev; fi && chown -R app:app /app

COPY script.js ./
RUN chown app:app script.js

USER app
EXPOSE 3000

ENV NODE_ENV=production
ENV PORT=3000

# start-period 15s: da tiempo a Node + pool DB a arrancar
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget -q -O - http://127.0.0.1:3000/health || exit 1

CMD ["node", "script.js"]
