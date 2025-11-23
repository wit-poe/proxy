# Use an official Python runtime as a parent image
# We use 3.10-slim for a balance of size and compatibility
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other libs
# (libgl1-mesa-glx is often needed for image processing libs)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy the rest of the application code
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for Python buffering
ENV PYTHONUNBUFFERED=1

# Run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]