FROM agrigorev/model-2025-hairstyle:v1

RUN pip install --no-cache-dir numpy onnxruntime pillow

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]
