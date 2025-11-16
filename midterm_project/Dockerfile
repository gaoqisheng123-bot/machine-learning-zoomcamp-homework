FROM python:3.9-slim as builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip sync uv.lock

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY predict.py .
COPY heart_disease_model.pkl .

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 9696

CMD ["python", "predict.py"]
