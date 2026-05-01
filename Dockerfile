# Base image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# copy project files
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose port
EXPOSE 8000

# run fastAPI
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]