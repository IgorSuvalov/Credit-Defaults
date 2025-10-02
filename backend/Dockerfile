FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# create non-root user
RUN useradd -m appuser
USER appuser

# app code
COPY backend/ ./backend/



EXPOSE 8004

CMD ["uvicorn", "backend.service:app", "--host", "0.0.0.0", "--port", "8004"]