# Base image shared by all stages
FROM python:3.11-slim AS base

RUN useradd -m appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir --no-deps .

# -------------------------
# Production dependencies
# -------------------------
FROM base AS prod

RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m"]
CMD ["iris_production_project.train_evaluate"]

# -------------------------
# Development / test stage
# -------------------------
FROM base AS dev

COPY requirements_dev.txt .
RUN pip install --no-cache-dir -r requirements_dev.txt

COPY tests/ ./tests/

RUN chown -R appuser:appuser /app
USER appuser

CMD ["pytest"]