FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone catanatron
RUN git clone --depth 1 https://github.com/bcollazo/catanatron.git catanatron_repo \
    && pip install --no-cache-dir -e catanatron_repo/catanatron

# Copy project
COPY catan_ai/ catan_ai/
COPY scripts/ scripts/
COPY pyproject.toml .
COPY runpod_handler.py .

# Install project
RUN pip install --no-cache-dir -e .

# Create dirs
RUN mkdir -p checkpoints logs demos

# Default: run training
ENV PYTHONPATH=/app:/app/catanatron_repo/catanatron
ENV WANDB_MODE=offline

ENTRYPOINT ["python"]
CMD ["scripts/train.py"]
