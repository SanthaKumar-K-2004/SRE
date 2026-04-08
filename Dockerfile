FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate dataset if not present
RUN python generate_dataset.py

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
