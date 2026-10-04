# Metaculus forecasting bot. Build:  docker build -t metaculus-bot .
# Secrets are NOT in the image: .env, config.toml, state/ and logs/ are mounted at run time.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY run.py config.toml ./
COPY bot ./bot
RUN mkdir -p /app/state /app/logs && chmod 0777 /app/state /app/logs \
    && python -m compileall -q bot run.py >/dev/null

# One pass over the scored tournaments. Does nothing unless armed = true in the mounted config.toml.
ENTRYPOINT ["python", "run.py"]
CMD ["run", "--live"]
