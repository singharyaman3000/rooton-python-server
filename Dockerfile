# Use the official Python 3.11.3 slim image
FROM python:3.11.3-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies and remove any pre-installed sqlite3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libreadline-dev \
    libsqlite3-dev && \
    apt-get remove -y sqlite3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and compile SQLite 3.41.2
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3410200.tar.gz && \
    tar xvfz sqlite-autoconf-3410200.tar.gz && \
    cd sqlite-autoconf-3410200 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3410200*

# Verify sqlite3 version
RUN sqlite3 --version

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Copy the .env file
COPY .env .

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
