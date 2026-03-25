FROM python:3.12-slim-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY binance-mcp.py .

CMD ["python", "binance-mcp.py"]
