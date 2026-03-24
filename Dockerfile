FROM python:3.11-slim

RUN useradd -m appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir .

RUN chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m"]
CMD ["iris_production_project.train_evaluate"]