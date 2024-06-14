FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y build-essential && \
    apt-get install -y libopenblas-dev liblapack-dev && \
    apt-get install -y libx11-dev libgtk-3-dev libboost-python-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/uploads /app/registered_faces

EXPOSE 5001

CMD ["python", "rest.py"]
