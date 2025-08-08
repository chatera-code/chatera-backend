# Step 1: Update the base image to the desired Python version.
# Using '3.13-slim' is a good practice as it will automatically
# get patch releases like 3.13.2 while keeping the image small.
FROM python:3.13-slim

# Step 2: Set the working directory inside the container.
WORKDIR /app

# Step 3: Copy requirements and install dependencies (leveraging cache).
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the rest of your application's source code.
COPY . .

# Step 5: Expose the port the app runs on.
EXPOSE 8000

# Step 6: Define the command to run your application.
# The 'main:main' part should be 'filename:fastapi_app_variable_name'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]