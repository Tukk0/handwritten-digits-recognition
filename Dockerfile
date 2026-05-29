# PyTorch image for training, inference and conversion
FROM python:3.11-slim

RUN pip install --no-cache-dir poetry && rm -rf /root/.cache/pip

WORKDIR /app
COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root && \
    rm -rf /root/.cache/pip

COPY . .

# Install system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Make run.sh executable
RUN chmod +x run.sh entrypoint.sh

# Default command is the one-command pipeline
CMD ["./run.sh"]
