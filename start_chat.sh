#!/bin/bash
#
# Start the ARC Challenge Solver Chat Interface
#

echo "🧩 ARC Challenge Solver - Chat Interface"
echo "========================================"
echo ""

# Check if dependencies are installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip3 install --break-system-packages -q flask flask-cors matplotlib
    echo "✓ Dependencies installed"
    echo ""
fi

echo "🚀 Starting chat server..."
echo ""
echo "Keyboard shortcuts:"
echo "  Ctrl+Enter  - Send message"
echo "  Ctrl+L      - Clear conversation"
echo "  Ctrl+K      - Command palette"
echo "  Ctrl+↑/↓    - Navigate history"
echo "  Ctrl+/      - Show all shortcuts"
echo ""
echo "Open http://localhost:5000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 chat_server.py
