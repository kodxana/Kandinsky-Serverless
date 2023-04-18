FROM runpod/pytorch:3.10-2.0.0-117

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python packages
RUN pip install --no-cache-dir runpod opencv-python "git+https://github.com/ai-forever/Kandinsky-2.git" "git+https://github.com/openai/CLIP.git"

# copy models
COPY kandinsky2 /app/kandinsky2

# Copy the serverless implementation file into the container
COPY rp_handler.py /app/rp_handler.py

# Set the entry point
CMD ["python", "/app/rp_handler.py"]
