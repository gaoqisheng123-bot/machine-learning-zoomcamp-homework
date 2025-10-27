FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml .

RUN pip install uv && uv pip install --system -r <(uv pip compile pyproject.toml -q -o -)

COPY app.py .

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
