FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml .

RUN pip install uv \
    && uv pip compile pyproject.toml -q -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY app.py .

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
