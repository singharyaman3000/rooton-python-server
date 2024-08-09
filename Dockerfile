# Use the official Python image from the Docker Hub
FROM python:3.11.3

# Set the working directory
WORKDIR /app

# Install system dependencies and update sqlite3
RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-dev && \
    apt-get upgrade -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code into the container
COPY . .

# Copy the .env file
COPY .env .

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
