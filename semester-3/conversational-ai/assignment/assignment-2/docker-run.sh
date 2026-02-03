#!/bin/bash
# Helper script to run tasks inside Docker

echo "============================================================"
echo "WIKIPEDIA HYBRID RAG - DOCKER RUNNER"
echo "============================================================"
echo ""

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "Choose a task to run inside Docker:"
echo "  1. Run Web UI (localhost:5000)"
echo "  2. Run Evaluation Pipeline"
echo "  3. Build/Rebuild Docker Image"
echo "  4. Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Launching Web UI in Docker..."
        docker-compose up
        ;;
    2)
        echo "Running Evaluation Pipeline in Docker..."
        docker-compose run --rm rag-system python run_evaluation.py
        ;;
    3)
        echo "Building Docker Image..."
        docker-compose build
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
