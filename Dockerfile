FROM python:3.7-buster
COPY . .
RUN pip install -r requirements.txt
CMD flask run --host 0.0.0.0 --port $PORT
