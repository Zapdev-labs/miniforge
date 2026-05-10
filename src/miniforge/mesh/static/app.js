// Miniforge Mesh Dashboard JavaScript

const socket = io();
let currentStreamMessage = null;

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to mesh');
    addSystemMessage('Connected to mesh dashboard');
});

socket.on('disconnect', () => {
    console.log('Disconnected from mesh');
    addSystemMessage('Disconnected from mesh');
});

socket.on('status', (data) => {
    updateStatus(data);
});

socket.on('chat_chunk', (data) => {
    if (currentStreamMessage) {
        const content = currentStreamMessage.querySelector('.message-content');
        content.textContent += data.chunk;
        scrollToBottom();
    }
});

socket.on('chat_complete', () => {
    currentStreamMessage = null;
});

socket.on('chat_response', (data) => {
    addAssistantMessage(data.response);
});

socket.on('chat_error', (data) => {
    addErrorMessage(data.error);
    currentStreamMessage = null;
});

// UI Functions
function updateStatus(data) {
    // Update resource displays
    if (data.resources) {
        document.getElementById('total-nodes').textContent = data.resources.nodes;
        document.getElementById('nodes-count').textContent = `${data.resources.nodes} node${data.resources.nodes !== 1 ? 's' : ''}`;
        document.getElementById('total-ram').textContent = `${Math.round(data.resources.total_ram_gb)} GB`;
        document.getElementById('available-ram').textContent = `${Math.round(data.resources.available_ram_gb)} GB`;
        document.getElementById('total-cpus').textContent = data.resources.total_cpu_cores;
    }
    
    // Update nodes table
    updateNodesTable();
}

async function updateNodesTable() {
    try {
        const response = await fetch('/api/nodes');
        const data = await response.json();
        
        const tbody = document.querySelector('#nodes-table tbody');
        tbody.innerHTML = '';
        
        data.nodes.forEach(node => {
            const row = document.createElement('tr');
            row.className = `node-row ${node.node_id === window.NODE_ID ? 'self' : ''}`;
            
            const isSelf = node.node_id === window.NODE_ID;
            const name = isSelf ? `${node.node_name} (You)` : node.node_name;
            const isLeader = node.is_leader ? ' 👑' : '';
            
            let statusClass = 'active';
            if (node.status === 'busy') statusClass = 'busy';
            if (node.status === 'offline') statusClass = 'offline';
            
            row.innerHTML = `
                <td>${name}${isLeader}</td>
                <td>${node.ip}:${node.port}</td>
                <td>${Math.round(node.ram_gb)} GB (${Math.round(node.ram_available)} GB free)</td>
                <td>${node.cpu_cores} cores (${Math.round(node.cpu_percent)}%)</td>
                <td><span class="status-badge ${statusClass}">${node.status}</span></td>
            `;
            
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Failed to update nodes:', error);
    }
}

function refreshStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => updateStatus(data))
        .catch(e => console.error('Refresh failed:', e));
    
    updateNodesTable();
}

// Chat Functions
function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addUserMessage(message);
    input.value = '';
    
    const streamMode = document.getElementById('stream-mode').checked;
    
    if (streamMode) {
        // Create placeholder for streaming response
        currentStreamMessage = addAssistantMessage('', true);
        socket.emit('chat_message', {
            message: message,
            stream: true
        });
    } else {
        // Non-streaming
        socket.emit('chat_message', {
            message: message,
            stream: false
        });
    }
}

function addUserMessage(text) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
        <div class="message-sender">You</div>
        <div class="message-content">${escapeHtml(text)}</div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

function addAssistantMessage(text, isStreaming = false) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-sender">Assistant ${isStreaming ? '<span class="loading"></span>' : ''}</div>
        <div class="message-content">${escapeHtml(text)}</div>
    `;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function addSystemMessage(text) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message system';
    div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    container.appendChild(div);
    scrollToBottom();
}

function addErrorMessage(text) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message system';
    div.style.borderLeftColor = 'var(--error)';
    div.innerHTML = `<div class="message-content">Error: ${escapeHtml(text)}</div>`;
    container.appendChild(div);
    scrollToBottom();
}

function scrollToBottom() {
    const container = document.getElementById('chat-messages');
    container.scrollTop = container.scrollHeight;
}

// Connect Modal Functions
function showConnectModal() {
    document.getElementById('connect-modal').style.display = 'block';
}

function hideConnectModal() {
    document.getElementById('connect-modal').style.display = 'none';
    document.getElementById('connect-status').textContent = '';
}

async function connectToPeer() {
    const ip = document.getElementById('peer-ip').value.trim();
    const port = parseInt(document.getElementById('peer-port').value) || 9999;
    const statusDiv = document.getElementById('connect-status');
    
    if (!ip) {
        statusDiv.textContent = 'Please enter an IP address';
        statusDiv.style.color = 'var(--error)';
        return;
    }
    
    statusDiv.textContent = 'Connecting...';
    statusDiv.style.color = 'var(--text-secondary)';
    
    try {
        const response = await fetch('/api/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ip, port })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            statusDiv.textContent = `Connected to ${ip}:${port}`;
            statusDiv.style.color = 'var(--success)';
            setTimeout(hideConnectModal, 1000);
        } else {
            statusDiv.textContent = data.error || 'Connection failed';
            statusDiv.style.color = 'var(--error)';
        }
    } catch (error) {
        statusDiv.textContent = 'Connection failed: ' + error.message;
        statusDiv.style.color = 'var(--error)';
    }
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event Listeners
document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Close modal on outside click
window.onclick = function(event) {
    const modal = document.getElementById('connect-modal');
    if (event.target === modal) {
        hideConnectModal();
    }
}

// Periodic refresh
setInterval(() => {
    refreshStatus();
}, 5000);

// Initial load
refreshStatus();
