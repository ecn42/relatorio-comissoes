# Use a lightweight official Python runtime as a parent image
FROM python:3.12.11-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on
EXPOSE 8080


# Define the command to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "relatorio_comissoes.py", "--server.port=8080", "--server.address=0.0.0.0"]