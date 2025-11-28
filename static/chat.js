/**
 * ARC Challenge Solver - Chat Interface JavaScript
 * Comprehensive keyboard navigation inspired by Cursor and Claude Code
 */

class ChatInterface {
    constructor() {
        this.conversationId = this.generateId();
        this.messages = [];
        this.messageHistory = [];
        this.historyIndex = -1;
        this.isWaitingForResponse = false;

        // DOM elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.newConversationBtn = document.getElementById('newConversationBtn');
        this.themeToggle = document.getElementById('themeToggle');
        this.shortcutsBtn = document.getElementById('shortcutsBtn');
        this.shortcutsModal = document.getElementById('shortcutsModal');
        this.closeShortcutsBtn = document.getElementById('closeShortcutsBtn');
        this.commandPalette = document.getElementById('commandPalette');
        this.commandSearch = document.getElementById('commandSearch');
        this.commandList = document.getElementById('commandList');

        // Commands for command palette
        this.commands = [
            { id: 'clear', title: 'Clear conversation', shortcut: 'Ctrl+L', action: () => this.clearConversation() },
            { id: 'new', title: 'New conversation', shortcut: 'Ctrl+N', action: () => this.newConversation() },
            { id: 'theme', title: 'Toggle theme', shortcut: 'Ctrl+Shift+T', action: () => this.toggleTheme() },
            { id: 'shortcuts', title: 'Show keyboard shortcuts', shortcut: 'Ctrl+/', action: () => this.showShortcuts() },
            { id: 'help', title: 'Show help', shortcut: '', action: () => this.sendMessage('help') },
            { id: 'primitives', title: 'List primitives', shortcut: '', action: () => this.sendMessage('primitives') },
            { id: 'solve', title: 'Solve ARC challenge', shortcut: '', action: () => this.sendMessage('solve') },
        ];

        this.filteredCommands = [...this.commands];
        this.selectedCommandIndex = 0;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadTheme();
        this.autoResizeTextarea();
        this.messageInput.focus();

        // Show welcome message
        this.showWelcomeMessage();
    }

    setupEventListeners() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());

        // Clear conversation
        this.clearBtn.addEventListener('click', () => this.clearConversation());

        // New conversation
        this.newConversationBtn.addEventListener('click', () => this.newConversation());

        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // Shortcuts modal
        this.shortcutsBtn.addEventListener('click', () => this.showShortcuts());
        this.closeShortcutsBtn.addEventListener('click', () => this.hideShortcuts());

        // Click outside modals to close
        this.shortcutsModal.addEventListener('click', (e) => {
            if (e.target === this.shortcutsModal) {
                this.hideShortcuts();
            }
        });

        this.commandPalette.addEventListener('click', (e) => {
            if (e.target === this.commandPalette) {
                this.hideCommandPalette();
            }
        });

        // Command search
        this.commandSearch.addEventListener('input', (e) => this.filterCommands(e.target.value));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcut(e));

        // Input handling
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
        this.messageInput.addEventListener('keydown', (e) => this.handleInputKeydown(e));
    }

    handleKeyboardShortcut(e) {
        // Check if we're in a modal or command palette
        const inModal = this.shortcutsModal.classList.contains('active');
        const inCommandPalette = this.commandPalette.classList.contains('active');

        // Command palette navigation
        if (inCommandPalette) {
            if (e.key === 'Escape') {
                e.preventDefault();
                this.hideCommandPalette();
                return;
            }

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                this.selectNextCommand();
                return;
            }

            if (e.key === 'ArrowUp') {
                e.preventDefault();
                this.selectPreviousCommand();
                return;
            }

            if (e.key === 'Enter') {
                e.preventDefault();
                this.executeSelectedCommand();
                return;
            }

            return;
        }

        // Close modals with Escape
        if (e.key === 'Escape') {
            if (inModal) {
                this.hideShortcuts();
                return;
            }

            // Clear input if not in modal
            if (document.activeElement === this.messageInput) {
                e.preventDefault();
                this.messageInput.value = '';
                this.autoResizeTextarea();
                return;
            }
        }

        // Ctrl+Enter - Send message
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            this.handleSendMessage();
            return;
        }

        // Ctrl+L - Clear conversation
        if (e.ctrlKey && e.key === 'l') {
            e.preventDefault();
            this.clearConversation();
            return;
        }

        // Ctrl+K - Command palette
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            this.showCommandPalette();
            return;
        }

        // Ctrl+N - New conversation
        if (e.ctrlKey && e.key === 'n') {
            e.preventDefault();
            this.newConversation();
            return;
        }

        // Ctrl+/ - Show shortcuts
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            this.showShortcuts();
            return;
        }

        // Ctrl+Shift+T - Toggle theme
        if (e.ctrlKey && e.shiftKey && e.key === 'T') {
            e.preventDefault();
            this.toggleTheme();
            return;
        }

        // Ctrl+↑ - Previous message in history
        if (e.ctrlKey && e.key === 'ArrowUp') {
            e.preventDefault();
            this.navigateHistory(-1);
            return;
        }

        // Ctrl+↓ - Next message in history
        if (e.ctrlKey && e.key === 'ArrowDown') {
            e.preventDefault();
            this.navigateHistory(1);
            return;
        }

        // Alt+↑ - Scroll messages up
        if (e.altKey && e.key === 'ArrowUp') {
            e.preventDefault();
            this.scrollMessages(-100);
            return;
        }

        // Alt+↓ - Scroll messages down
        if (e.altKey && e.key === 'ArrowDown') {
            e.preventDefault();
            this.scrollMessages(100);
            return;
        }

        // / - Focus input (only if not already focused)
        if (e.key === '/' && document.activeElement !== this.messageInput) {
            e.preventDefault();
            this.messageInput.focus();
            return;
        }
    }

    handleInputKeydown(e) {
        // Don't handle special key combinations here - let handleKeyboardShortcut deal with them
        if (e.ctrlKey || e.altKey || e.metaKey) {
            return;
        }

        // Regular Enter (without Ctrl) adds a new line
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
            // Allow default behavior for regular Enter
            return;
        }
    }

    navigateHistory(direction) {
        if (this.messageHistory.length === 0) return;

        // Update history index
        this.historyIndex += direction;

        // Clamp to valid range
        if (this.historyIndex < 0) {
            this.historyIndex = 0;
        } else if (this.historyIndex >= this.messageHistory.length) {
            this.historyIndex = this.messageHistory.length - 1;
        }

        // Set input value
        this.messageInput.value = this.messageHistory[this.historyIndex] || '';
        this.autoResizeTextarea();

        // Move cursor to end
        this.messageInput.setSelectionRange(this.messageInput.value.length, this.messageInput.value.length);
    }

    scrollMessages(amount) {
        this.messagesContainer.scrollTop += amount;
    }

    showCommandPalette() {
        this.commandPalette.classList.add('active');
        this.commandSearch.value = '';
        this.commandSearch.focus();
        this.filteredCommands = [...this.commands];
        this.selectedCommandIndex = 0;
        this.renderCommandList();
    }

    hideCommandPalette() {
        this.commandPalette.classList.remove('active');
        this.messageInput.focus();
    }

    filterCommands(query) {
        const lowerQuery = query.toLowerCase();
        this.filteredCommands = this.commands.filter(cmd =>
            cmd.title.toLowerCase().includes(lowerQuery) ||
            cmd.shortcut.toLowerCase().includes(lowerQuery)
        );
        this.selectedCommandIndex = 0;
        this.renderCommandList();
    }

    renderCommandList() {
        this.commandList.innerHTML = '';

        if (this.filteredCommands.length === 0) {
            this.commandList.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-secondary);">No commands found</div>';
            return;
        }

        this.filteredCommands.forEach((cmd, index) => {
            const item = document.createElement('div');
            item.className = 'command-item' + (index === this.selectedCommandIndex ? ' selected' : '');
            item.innerHTML = `
                <span class="command-title">${cmd.title}</span>
                ${cmd.shortcut ? `<span class="command-shortcut">${cmd.shortcut}</span>` : ''}
            `;

            item.addEventListener('click', () => {
                cmd.action();
                this.hideCommandPalette();
            });

            this.commandList.appendChild(item);
        });
    }

    selectNextCommand() {
        this.selectedCommandIndex = (this.selectedCommandIndex + 1) % this.filteredCommands.length;
        this.renderCommandList();
    }

    selectPreviousCommand() {
        this.selectedCommandIndex = (this.selectedCommandIndex - 1 + this.filteredCommands.length) % this.filteredCommands.length;
        this.renderCommandList();
    }

    executeSelectedCommand() {
        if (this.filteredCommands[this.selectedCommandIndex]) {
            this.filteredCommands[this.selectedCommandIndex].action();
            this.hideCommandPalette();
        }
    }

    showShortcuts() {
        this.shortcutsModal.classList.add('active');
    }

    hideShortcuts() {
        this.shortcutsModal.classList.remove('active');
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }

    loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    showWelcomeMessage() {
        // Welcome message is in HTML, just ensure messages container is visible
    }

    clearEmptyState() {
        const emptyState = this.messagesContainer.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
    }

    async handleSendMessage() {
        const message = this.messageInput.value.trim();

        if (!message || this.isWaitingForResponse) {
            return;
        }

        // Add to history
        this.messageHistory.push(message);
        this.historyIndex = this.messageHistory.length;

        // Clear input
        this.messageInput.value = '';
        this.autoResizeTextarea();

        // Remove empty state
        this.clearEmptyState();

        // Add user message to UI
        this.addMessage('user', message);

        // Show typing indicator
        this.showTypingIndicator();

        // Send to server
        this.isWaitingForResponse = true;
        this.sendBtn.disabled = true;

        try {
            const response = await this.sendMessage(message);
            this.hideTypingIndicator();

            if (response && response.content) {
                this.addMessage('assistant', response.content, response.data);
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('assistant', `Error: ${error.message}`);
        } finally {
            this.isWaitingForResponse = false;
            this.sendBtn.disabled = false;
            this.messageInput.focus();
        }
    }

    async sendMessage(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: this.conversationId
            })
        });

        if (!response.ok) {
            throw new Error('Failed to send message');
        }

        const data = await response.json();
        return data.response;
    }

    addMessage(role, content, data = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'A';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        // Parse markdown-like content
        bubble.innerHTML = this.parseContent(content);

        const timestamp = document.createElement('div');
        timestamp.className = 'message-time';
        timestamp.textContent = new Date().toLocaleTimeString();

        messageContent.appendChild(bubble);
        messageContent.appendChild(timestamp);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        // Add visualizations if present
        if (data && data.type === 'grid_visualization' && data.image) {
            const img = document.createElement('img');
            img.src = data.image;
            img.alt = 'Grid visualization';
            messageContent.appendChild(img);
        }

        this.messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        this.scrollToBottom();

        // Store message
        this.messages.push({ role, content, data, timestamp: new Date() });
    }

    parseContent(content) {
        // Simple markdown-like parsing
        let html = content;

        // Code blocks
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    showTypingIndicator() {
        if (document.querySelector('.typing-indicator')) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'A';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator active';
        indicator.innerHTML = '<span></span><span></span><span></span>';

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(indicator);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const indicator = document.querySelector('.message.assistant:last-child');
        if (indicator && indicator.querySelector('.typing-indicator')) {
            indicator.remove();
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    clearConversation() {
        if (!confirm('Clear this conversation?')) {
            return;
        }

        this.messages = [];
        this.messagesContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🧩</div>
                <div class="empty-state-text">Conversation cleared</div>
                <div class="empty-state-hint">
                    Type a message or press <kbd>Ctrl</kbd> + <kbd>K</kbd> for commands
                </div>
            </div>
        `;

        // Clear on server
        fetch(`/api/conversation/${this.conversationId}`, {
            method: 'DELETE'
        });

        this.messageInput.focus();
    }

    newConversation() {
        if (this.messages.length > 0 && !confirm('Start a new conversation?')) {
            return;
        }

        this.conversationId = this.generateId();
        this.messages = [];
        this.messageHistory = [];
        this.historyIndex = -1;

        this.messagesContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🧩</div>
                <div class="empty-state-text">New conversation started</div>
                <div class="empty-state-hint">
                    Type a message or press <kbd>Ctrl</kbd> + <kbd>K</kbd> for commands
                </div>
            </div>
        `;

        this.messageInput.focus();
    }

    generateId() {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

// Initialize chat interface when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
});
