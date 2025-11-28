# ARC Challenge Solver - Chat Interface

A modern, keyboard-first chat interface for interacting with the ARC Challenge Solver, inspired by the best features of Cursor and Claude Code.

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements-chat.txt
```

2. Start the chat server:
```bash
python3 chat_server.py
```

3. Open your browser to: `http://localhost:5000`

## ⌨️ Keyboard Shortcuts

The chat interface features comprehensive keyboard navigation for power users:

### Message Controls
- **`Ctrl+Enter`** - Send message
- **`Ctrl+↑`** / **`Ctrl+↓`** - Navigate through message history
- **`Escape`** - Clear input field

### Navigation
- **`Ctrl+L`** - Clear conversation
- **`Ctrl+N`** - Start new conversation
- **`Ctrl+K`** - Open command palette
- **`Alt+↑`** / **`Alt+↓`** - Scroll messages up/down
- **`/`** - Focus input field (when not already focused)

### UI Controls
- **`Ctrl+/`** - Show keyboard shortcuts help
- **`Ctrl+Shift+T`** - Toggle dark/light theme

### Command Palette (`Ctrl+K`)
- **`↑`** / **`↓`** - Navigate commands
- **`Enter`** - Execute selected command
- **`Escape`** - Close palette
- Type to search/filter commands

## 🎨 Features

### 1. Command Palette
Press `Ctrl+K` to access the quick command palette:
- Search all available commands
- Keyboard navigation with arrow keys
- Quick access to common actions
- Displays keyboard shortcuts for each command

### 2. Message History Navigation
Navigate through previously sent messages:
- `Ctrl+↑` to go to previous messages
- `Ctrl+↓` to go to next messages
- Automatically fills the input field
- Perfect for editing and resending commands

### 3. Smart Scroll Controls
- `Alt+↑` / `Alt+↓` for smooth scrolling
- Auto-scroll to latest message
- Smooth animations

### 4. Theme Support
- Light and dark modes
- Automatic theme persistence
- Toggle with `Ctrl+Shift+T`
- Smooth transitions

### 5. Modern UI
- Clean, minimalist design
- Responsive layout
- Smooth animations
- Clear visual hierarchy
- Accessibility-focused

### 6. Grid Visualizations
- Inline ARC grid visualizations
- Base64-encoded images
- Proper grid color palette
- Responsive sizing

## 🤖 Chat Commands

The chatbot understands several commands:

### Help Commands
- `help` or `/help` - Show help information
- `?` - Quick help

### Solver Commands
- `solve <task>` - Solve an ARC challenge
- `primitives` or `/primitives` - List available DSL primitives
- `visualize <grid>` - Visualize a grid

### Examples
```
help
primitives
solve task_001
visualize [[0,1,0],[1,1,1],[0,1,0]]
```

## 🏗️ Architecture

### Backend (`chat_server.py`)
- Flask-based REST API
- Conversation management
- Integration with ARC solver
- Grid visualization support

### Frontend
- Pure JavaScript (no frameworks)
- Modular class-based architecture
- Event-driven design
- Minimal dependencies

### Key Files
```
chat_server.py          # Flask server and chatbot logic
templates/chat.html     # Main chat interface
static/chat.js         # JavaScript application
scripts/visualize.py    # Grid visualization utilities
```

## 🔧 API Endpoints

### `POST /api/chat`
Send a message to the chatbot.

**Request:**
```json
{
  "message": "help",
  "conversation_id": "conv_123"
}
```

**Response:**
```json
{
  "response": {
    "id": "msg_456",
    "role": "assistant",
    "content": "I can help you...",
    "data": null,
    "timestamp": "2025-11-28T03:15:00"
  },
  "conversation_id": "conv_123"
}
```

### `GET /api/conversation/<conversation_id>`
Get conversation history.

### `DELETE /api/conversation/<conversation_id>`
Clear conversation history.

### `GET /api/conversations`
List all conversations.

## 💡 Design Philosophy

### Keyboard-First
Inspired by Cursor and Claude Code, every action has a keyboard shortcut. Power users can navigate the entire interface without touching the mouse.

### Useful, Not Just Different
Features are designed to solve real problems:
- **Message history navigation** - Easily retry or modify previous commands
- **Command palette** - Discover and execute commands quickly
- **Smart scrolling** - Navigate long conversations efficiently
- **Theme support** - Comfortable viewing in any lighting

### Accessibility
- Semantic HTML
- ARIA labels where needed
- Keyboard navigation
- Clear visual indicators
- Responsive design

## 🎯 Keyboard Navigation Inspiration

### From Cursor
- `Ctrl+K` command palette
- Fuzzy search in command palette
- Clean, minimalist design
- Fast keyboard navigation

### From Claude Code
- Message history navigation
- Clean chat interface
- Markdown rendering
- Code block highlighting
- Smooth scrolling

## 🚧 Future Enhancements

Potential improvements:
- [ ] Syntax highlighting for code blocks
- [ ] Export conversation to markdown
- [ ] Conversation search
- [ ] Multi-conversation tabs
- [ ] File upload for ARC tasks
- [ ] Persistent conversation storage
- [ ] Streaming responses
- [ ] Code execution in chat
- [ ] Interactive grid editor
- [ ] Solution comparison view

## 🐛 Troubleshooting

### Server won't start
```bash
# Make sure Flask is installed
pip install flask flask-cors

# Check for port conflicts
lsof -i :5000
```

### Keyboard shortcuts not working
- Check if you're in an input field (some shortcuts are context-aware)
- Make sure the chat window has focus
- Try pressing `Ctrl+/` to see all available shortcuts

### Theme not persisting
- Check browser localStorage permissions
- Clear browser cache and try again

## 📝 License

Same as the main ARC Challenge Solver project (MIT License).

## 🙏 Acknowledgments

- Inspired by [Cursor](https://cursor.sh/) keyboard navigation
- Inspired by [Claude Code](https://claude.ai/code) chat interface
- Built with Flask, vanilla JavaScript, and CSS
