#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

echo "🛑 Stopping Pricing Engines..."
echo "========================================"

# Stop both calculators' Docker containers
if ! docker-compose down; then
    echo ""
    echo "❌ FAILED: the calculators were not stopped. Is Docker Desktop running?"
    echo ""
    read -p "Press any key to exit..."
    exit 1
fi

echo ""
echo "✅ Both calculators stopped"
echo "🔒 Ports 8050 and 8051 freed"
echo ""
echo "You can now safely close this window."
echo ""
read -p "Press any key to exit..."
