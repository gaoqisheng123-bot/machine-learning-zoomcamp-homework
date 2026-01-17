FROM python:3.11-slim as builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip sync uv.lock

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY predict.py .
COPY traffic_sign_model.h5 .

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 9696

CMD ["python", "predict.py"]