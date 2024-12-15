#!/bin/bash

# Build the Docker image
docker build -t router-expert-system .

# Run the Docker container
docker run -p 5000:5000 router-expert-system
