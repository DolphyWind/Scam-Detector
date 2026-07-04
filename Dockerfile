FROM python:3.14
COPY --from=docker.io/astral/uv:0.8.9 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml .
COPY .python-version .
COPY uv.lock .
COPY main.py .
COPY bot_config.py .
COPY scam_detector.py .

RUN ["uv", "sync"]
CMD ["uv", "run", "main.py"]
