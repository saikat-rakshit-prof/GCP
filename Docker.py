FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install gunicorn

RUN pip install pydantic==2.9.1

RUN pip install -r requirements.txt

ENV PORT=8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app