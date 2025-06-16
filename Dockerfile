# Use Python 3.9 base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port 5000 for Flask app
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app/app.py"]

ENV CUDA_VISIBLE_DEVICES=""
