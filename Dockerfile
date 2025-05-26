FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    gcc \
    libpython3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pip.conf /etc/pip.conf

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN rm -fr requirements.txt /etc/pip.conf


EXPOSE 8080

CMD [ "python", "-m", "app.main" ]
