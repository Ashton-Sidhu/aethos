FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Make directories suited to your application
RUN mkdir -p /var/www/{{ name }}
WORKDIR /var/www/{{ name }}

# Copy and install requirements
COPY ./app /var/www/{{ name }}
RUN pip install --no-cache-dir -r requirements.txt
