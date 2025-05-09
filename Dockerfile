FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Install FastAPI and related dependencies
RUN pip install fastapi uvicorn requests pydantic python-dotenv

# Clone the LLM repository
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

# Copy application code
COPY . .

# Install additional requirements for LLaMA-Factory
RUN cd LLaMA-Factory && pip install -e ".[torch,metrics]" --no-cache-dir && cd ../

# Set Python path to include the current directory
ENV PYTHONPATH=/app

# Configure CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

# Expose the API port
EXPOSE 23477

# Change to api directory and run the script
CMD ["sh", "-c", "cd api && python train_server.py"]