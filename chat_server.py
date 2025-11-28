"""
Flask-based chat server for ARC Challenge Solver.
Provides a conversational interface to interact with the solver.
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from arc_core.grid import Grid
from dsl.interpreter import Interpreter, Environment
from dsl.ast import Program, Expr
from synth.enumerator import EnumerativeSynthesizer, StochasticSynthesizer
from scripts.visualize import grid_to_base64

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Store conversation history in memory (in production, use a database)
conversations: Dict[str, List[Dict[str, Any]]] = {}


class ARCChatbot:
    """Chatbot interface for ARC solver."""

    def __init__(self):
        self.interpreter = Interpreter()
        self.env = Environment()

    def process_message(self, message: str, conversation_id: str) -> Dict[str, Any]:
        """Process a user message and return a response."""

        # Initialize conversation if new
        if conversation_id not in conversations:
            conversations[conversation_id] = []

        # Add user message to history
        user_msg = {
            'id': str(uuid.uuid4()),
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        conversations[conversation_id].append(user_msg)

        # Process command
        response = self._generate_response(message)

        # Add assistant response to history
        assistant_msg = {
            'id': str(uuid.uuid4()),
            'role': 'assistant',
            'content': response['text'],
            'data': response.get('data'),
            'timestamp': datetime.now().isoformat()
        }
        conversations[conversation_id].append(assistant_msg)

        return assistant_msg

    def _generate_response(self, message: str) -> Dict[str, Any]:
        """Generate a response based on the user message."""

        message_lower = message.lower().strip()

        # Help command
        if message_lower in ['help', '/help', '?']:
            return {
                'text': self._get_help_text(),
                'data': None
            }

        # Solve command
        if message_lower.startswith('solve') or message_lower.startswith('/solve'):
            return self._handle_solve_command(message)

        # Visualize grid
        if message_lower.startswith('visualize') or message_lower.startswith('/visualize'):
            return self._handle_visualize_command(message)

        # Show primitives
        if message_lower.startswith('primitives') or message_lower.startswith('/primitives'):
            return self._handle_primitives_command()

        # Default response
        return {
            'text': f"I'm an ARC Challenge Solver assistant. I can help you:\n\n"
                   f"• Solve ARC challenges\n"
                   f"• Visualize grids\n"
                   f"• Explain primitives and operations\n\n"
                   f"Type 'help' for more information or use keyboard shortcuts (Ctrl+/ to see all).",
            'data': None
        }

    def _get_help_text(self) -> str:
        """Get help text."""
        return """**ARC Challenge Solver - Help**

**Commands:**
• `help` - Show this help message
• `solve <task>` - Solve an ARC challenge
• `visualize <grid>` - Visualize a grid
• `primitives` - List available primitives

**Keyboard Shortcuts:**
• `Ctrl+Enter` - Send message
• `Ctrl+L` - Clear conversation
• `Ctrl+K` - Open command palette
• `Ctrl+↑/↓` - Navigate message history
• `Ctrl+N` - New conversation
• `Ctrl+/` - Show keyboard shortcuts
• `Escape` - Clear input
• `Alt+↑/↓` - Scroll messages

**Example Usage:**
```
solve task_001
visualize [[0,1,0],[1,1,1],[0,1,0]]
primitives
```
"""

    def _handle_solve_command(self, message: str) -> Dict[str, Any]:
        """Handle solve command."""
        return {
            'text': "Solving ARC challenges requires training examples. Please provide:\n\n"
                   "1. Input/output grid pairs for training\n"
                   "2. Test input grid\n\n"
                   "Example format:\n"
                   "```json\n"
                   '{"train": [{"input": [[0,1]], "output": [[1,0]]}], "test": {"input": [[0,1,0]]}}\n'
                   "```",
            'data': None
        }

    def _handle_visualize_command(self, message: str) -> Dict[str, Any]:
        """Handle visualize command."""
        try:
            # Extract grid data from message
            # For now, return a sample visualization
            sample_grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            img_base64 = grid_to_base64(sample_grid)

            return {
                'text': "Here's a visualization of the grid:",
                'data': {
                    'type': 'grid_visualization',
                    'image': img_base64,
                    'grid': sample_grid.tolist()
                }
            }
        except Exception as e:
            return {
                'text': f"Error visualizing grid: {str(e)}",
                'data': None
            }

    def _handle_primitives_command(self) -> Dict[str, Any]:
        """Handle primitives command."""
        from dsl import primitives

        # Get all primitive functions
        primitive_list = []
        for name in dir(primitives):
            if not name.startswith('_'):
                obj = getattr(primitives, name)
                if callable(obj):
                    primitive_list.append(name)

        primitive_list.sort()

        text = f"**Available Primitives ({len(primitive_list)}):**\n\n"

        # Group by category (basic heuristic)
        categories = {
            'Transform': [],
            'Filter': [],
            'Spatial': [],
            'Color': [],
            'Object': [],
            'Other': []
        }

        for prim in primitive_list:
            if 'rotate' in prim or 'flip' in prim or 'reflect' in prim:
                categories['Transform'].append(prim)
            elif 'filter' in prim or 'select' in prim:
                categories['Filter'].append(prim)
            elif 'move' in prim or 'shift' in prim or 'position' in prim:
                categories['Spatial'].append(prim)
            elif 'color' in prim or 'recolor' in prim:
                categories['Color'].append(prim)
            elif 'object' in prim:
                categories['Object'].append(prim)
            else:
                categories['Other'].append(prim)

        for category, prims in categories.items():
            if prims:
                text += f"**{category}:** {', '.join(sorted(prims))}\n\n"

        return {
            'text': text,
            'data': {
                'type': 'primitives_list',
                'primitives': primitive_list,
                'categories': categories
            }
        }


chatbot = ARCChatbot()


@app.route('/')
def index():
    """Serve the chat interface."""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.json
    message = data.get('message', '')
    conversation_id = data.get('conversation_id', str(uuid.uuid4()))

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response = chatbot.process_message(message, conversation_id)
        return jsonify({
            'response': response,
            'conversation_id': conversation_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id: str):
    """Get conversation history."""
    if conversation_id not in conversations:
        return jsonify({'messages': []})

    return jsonify({'messages': conversations[conversation_id]})


@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id: str):
    """Clear conversation history."""
    if conversation_id in conversations:
        conversations[conversation_id] = []

    return jsonify({'success': True})


@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all conversations."""
    return jsonify({
        'conversations': [
            {
                'id': conv_id,
                'message_count': len(messages),
                'last_updated': messages[-1]['timestamp'] if messages else None
            }
            for conv_id, messages in conversations.items()
        ]
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    print("Starting ARC Challenge Solver Chat Interface...")
    print("Open http://localhost:5000 in your browser")
    print("\nKeyboard Shortcuts:")
    print("  Ctrl+Enter  - Send message")
    print("  Ctrl+L      - Clear conversation")
    print("  Ctrl+K      - Command palette")
    print("  Ctrl+↑/↓    - Navigate history")
    print("  Ctrl+/      - Show all shortcuts")

    app.run(debug=True, host='0.0.0.0', port=5000)
