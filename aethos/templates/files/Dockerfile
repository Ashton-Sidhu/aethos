FROM tiangolo/uvicorn-gunicorn:python3.6-alpine3.8

# Make directories suited to your application
RUN mkdir -p /var/www/{{ name }}
WORKDIR /var/www/{{ name }}

# Copy and install requirements
COPY requirements.txt /var/www/{{ name }}
RUN pip install --no-cache-dir -r requirements.txt

# Copy contents from your local to your docker container
COPY . /var/www/{{ name }}
