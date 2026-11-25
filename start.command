#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

echo "🚀 Starting Pricing Engines..."
echo "======================================"

# Check that Docker is installed
if ! command -v docker > /dev/null 2>&1; then
    echo "❌ Docker is not installed."
    echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop and try again."
    read -p "Press any key to exit..."
    exit 1
fi

# Check that Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker Desktop is not running!"
    echo "Please start Docker Desktop and try again."
    read -p "Press any key to exit..."
    exit 1
fi

echo "✅ Docker is running"
echo "🔨 Building and starting both calculators..."
echo ""

# Start both calculators (futures and options)
if ! docker-compose up -d --build; then
    echo ""
    echo "❌ FAILED: the calculators did not start. See the messages above."
    read -p "Press any key to exit..."
    exit 1
fi

# Wait for apps to be ready
echo "⏳ Waiting for calculators to start..."
sleep 6

# Open both calculators in the browser
open http://localhost:8050
sleep 1
open http://localhost:8051

echo ""
echo "✅ Both calculators are running!"
echo "📍 Futures Calculator: http://localhost:8050"
echo "📍 Options Calculator: http://localhost:8051"
echo ""
echo "🛑 To stop: run stop.command"
echo "======================================"
echo ""
read -p "Press any key to exit..."
