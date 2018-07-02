FROM python:alpine

EXPOSE 80

RUN pip install gunicorn

COPY ./app /app

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["gunicorn", "-b", "0.0.0.0:80", "main:app"]
