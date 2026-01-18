FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY webhook_sr_receiver.py .

# Crear directorios para datos persistentes
RUN mkdir -p /app/data/sr_reales /app/data/db

# Variables de entorno
ENV PORT=5000
ENV PRODUCTION=true

# Puerto
EXPOSE 5000

# Ejecutar con gunicorn (producción)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "webhook_sr_receiver:app"]

