FROM python:3.12

RUN apt-get -y update && apt-get -y upgrade && apt-get install build-essential

WORKDIR /app

COPY ./requirements.txt .

RUN python3 -m pip install -r requirements.txt

COPY . .


CMD ["python", "main.py"]
