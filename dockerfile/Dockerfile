FROM python:3.9-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Install core dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "python-multipart==0.0.6" && \
    pip check

# 4. Verification
RUN python -c "import numpy as np; print(f'NumPy: {np.__version__}')" && \
    python -c "import fastapi, uvicorn; print(f'FastAPI: {fastapi.__version__}, Uvicorn: {uvicorn.__version__}')"

# 5. Copy application
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]