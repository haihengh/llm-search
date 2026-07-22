FROM python:3.11-slim

WORKDIR /app

# Single pip install — only 3 dependencies, all already used by the main project
RUN pip install --no-cache-dir "fastapi>=0.110.0" "uvicorn[standard]>=0.27.0" "httpx>=0.27.0"

# Copy static files and server
COPY server.py .
COPY static/ ./static/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/api/health').raise_for_status()"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
