# Use Python 3.9 or newer
FROM python:3.12-slim

# Set working directory
WORKDIR /api

# Copy files
COPY ./api /api/api
COPY requirements.txt /api

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Command to run the app
CMD ["bash", "-c", "flask --app api/api --debug run --host 0.0.0.0"]
