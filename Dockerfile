FROM python:3.11-slim

RUN useradd -m appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app

USER appuser

# CMD ["python", "-m", "src.train_evaluate"]

ENTRYPOINT ["python", "-m"]
CMD ["src.train_evaluate"]