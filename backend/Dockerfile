# Dockerfile
FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./backend/

#Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies.
RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir -r backend/deployment.txt

COPY startup.sh /startup.sh
RUN sed -i 's/\r$//' /startup.sh && chmod +x /startup.sh

CMD ["/startup.sh"]
