FROM python:3.11

WORKDIR /app/ml-server/

COPY requirements.txt .
COPY *.py ./

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
