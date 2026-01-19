FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código (webhook + ML signal generator)
COPY webhook_sr_receiver.py .
COPY ml_signal_generator.py .

# Crear directorios para datos persistentes
# En EasyPanel estos directorios deben montarse como volúmenes
RUN mkdir -p /app/data/sr_reales /app/data/db /app/data/possible /app/data/raw /app/logs

# Variables de entorno
ENV PORT=5000
ENV PRODUCTION=true
ENV TZ=America/New_York
ENV SR_DATA_DIR=/app/data/sr_reales
ENV DB_PATH=/app/data/db/forex_bot.db

# Puerto
EXPOSE 5000

# Ejecutar con gunicorn (producción)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "--timeout", "120", "webhook_sr_receiver:app"]
