# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libreadline-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libffi-dev \
    libsqlite3-dev \
    sqlite3 \
    python3.11-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify sqlite3 version to ensure compatibility
RUN sqlite3 --version

# Create a virtual environment named rooton-be
RUN python3.11 -m venv /opt/rooton-be

# Activate the virtual environment and upgrade pip
RUN /opt/rooton-be/bin/pip install --upgrade pip

# Copy the application code into the container
COPY . /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies in the virtual environment
RUN /opt/rooton-be/bin/pip install --no-cache-dir -r requirements.txt

# Install pysqlite3-binary in the virtual environment
RUN /opt/rooton-be/bin/pip install --no-cache-dir pysqlite3-binary

# Ensure the virtual environment is activated by default
ENV PATH="/opt/rooton-be/bin:$PATH"

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
