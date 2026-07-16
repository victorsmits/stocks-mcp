FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --break-system-packages .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=10 \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1',8000),3); s.close()"

CMD ["python", "persistent_server_v2.py"]
