FROM python:3.14-slim
COPY --from=docker.io/astral/uv:0.8.9 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml .
COPY .python-version .
COPY uv.lock .
RUN ["uv", "sync"]

COPY . .

STOPSIGNAL SIGINT
CMD ["uv", "run", "main.py"]
