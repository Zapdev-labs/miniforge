(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  let conversations = JSON.parse(localStorage.getItem('mf_conversations') || '[]');
  let settings = JSON.parse(localStorage.getItem('mf_settings') || '{}');
  let currentId = null;
  let isGenerating = false;

  const chatContainer = $('#chatContainer');
  const messageInput = $('#messageInput');
  const sendBtn = $('#sendBtn');
  const conversationList = $('#conversationList');
  const welcome = $('#welcome');
  const statusText = $('#statusText');
  const statusMeta = $('#statusMeta');
  const runtimePill = $('#runtimePill');
  const runtimeModel = $('#runtimeModel');
  const runtimeDetails = $('#runtimeDetails');

  function saveSettings() {
    localStorage.setItem('mf_settings', JSON.stringify(settings));
  }

  function loadSettings() {
    $('#systemPrompt').value = settings.systemPrompt || 'You are a helpful assistant.';
    $('#maxTokens').value = settings.maxTokens || 512;
    $('#temperature').value = settings.temperature || 1.0;
    $('#topP').value = settings.topP || 0.95;
  }

  function persistSettings() {
    settings = {
      systemPrompt: $('#systemPrompt').value,
      maxTokens: $('#maxTokens').value,
      temperature: $('#temperature').value,
      topP: $('#topP').value,
    };
    saveSettings();
  }

  function saveConversations() {
    localStorage.setItem('mf_conversations', JSON.stringify(conversations));
  }

  function renderSidebar() {
    conversationList.innerHTML = '';
    conversations.slice().reverse().forEach(conv => {
      const div = document.createElement('div');
      div.className = 'conversation-item' + (conv.id === currentId ? ' active' : '');
      div.textContent = conv.title || 'New Chat';
      div.onclick = () => loadConversation(conv.id);
      conversationList.appendChild(div);
    });
  }

  function createConversation() {
    const id = Date.now().toString();
    const conv = { id, title: 'New Chat', messages: [] };
    conversations.push(conv);
    currentId = id;
    saveConversations();
    renderSidebar();
    renderMessages();
    return conv;
  }

  function loadConversation(id) {
    currentId = id;
    renderSidebar();
    renderMessages();
  }

  function getCurrentConversation() {
    if (!currentId) return createConversation();
    return conversations.find(c => c.id === currentId) || createConversation();
  }

  function addMessage(role, content) {
    const conv = getCurrentConversation();
    conv.messages.push({ role, content });
    if (conv.messages.length === 1 && role === 'user') {
      conv.title = content.slice(0, 40) + (content.length > 40 ? '...' : '');
    }
    saveConversations();
    renderSidebar();
    renderMessages();
  }

  function updateLastMessage(content) {
    const conv = getCurrentConversation();
    if (conv.messages.length && conv.messages[conv.messages.length - 1].role === 'assistant') {
      conv.messages[conv.messages.length - 1].content = content;
      saveConversations();
    }
  }

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[m]));
  }

  function markdownToHtml(text) {
    // Simple markdown: code blocks, inline code, bold, italic, paragraphs
    let html = escapeHtml(text);
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.split(/\n\n+/).map(p => '<p>' + p.replace(/\n/g, '<br>') + '</p>').join('');
    return html;
  }

  function renderMessages() {
    const conv = getCurrentConversation();
    chatContainer.innerHTML = '';
    if (conv.messages.length === 0) {
      chatContainer.appendChild(welcome);
      return;
    }
    conv.messages.forEach(msg => {
      const div = document.createElement('div');
      div.className = 'message ' + msg.role;
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = msg.role === 'user' ? 'You' : 'AI';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerHTML = markdownToHtml(msg.content);
      div.appendChild(avatar);
      div.appendChild(bubble);
      chatContainer.appendChild(div);
    });
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isGenerating) return;
    messageInput.value = '';
    messageInput.style.height = 'auto';
    addMessage('user', text);

    isGenerating = true;
    sendBtn.disabled = true;

    const systemPrompt = $('#systemPrompt').value;
    const maxTokens = parseInt($('#maxTokens').value, 10) || 512;
    const temperature = parseFloat($('#temperature').value) || 1.0;
    const topP = parseFloat($('#topP').value) || 0.95;

    const messages = [{ role: 'system', content: systemPrompt }];
    const conv = getCurrentConversation();
    messages.push(...conv.messages.slice(-20));

    // Add placeholder assistant message
    conv.messages.push({ role: 'assistant', content: '' });
    saveConversations();
    renderMessages();

    const placeholderBubble = chatContainer.querySelector('.message.assistant:last-child .bubble');

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'miniforge',
          messages,
          max_tokens: maxTokens,
          temperature,
          top_p: topP,
          stream: true,
        }),
      });

      if (!response.ok || !response.body) {
        let errorMessage = 'Request failed';
        try {
          const errorBody = await response.json();
          errorMessage = errorBody.error || errorMessage;
        } catch (e) {
          errorMessage = response.statusText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') continue;
          try {
            const data = JSON.parse(dataStr);
            const delta = data.choices?.[0]?.delta?.content;
            if (delta) {
              assistantText += delta;
              updateLastMessage(assistantText);
              if (placeholderBubble) {
                placeholderBubble.innerHTML = markdownToHtml(assistantText);
              }
              chatContainer.scrollTop = chatContainer.scrollHeight;
            }
          } catch (e) {
            // ignore malformed json
          }
        }
      }
    } catch (err) {
      console.error(err);
      if (placeholderBubble) {
        placeholderBubble.innerHTML = '<p style="color:#ff6b6b">Error: ' + escapeHtml(String(err)) + '</p>';
      }
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      renderSidebar();
    }
  }

  // Event listeners
  sendBtn.addEventListener('click', sendMessage);
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
  });

  $('#newChatBtn').addEventListener('click', () => {
    createConversation();
    welcome.style.display = '';
    renderMessages();
  });

  $('#clearBtn').addEventListener('click', () => {
    const conv = getCurrentConversation();
    conv.messages = [];
    saveConversations();
    renderMessages();
  });

  $('#menuToggle').addEventListener('click', () => {
    $('#sidebar').classList.toggle('open');
  });

  // Settings modal
  const modal = $('#settingsModal');
  $('#settingsBtn').addEventListener('click', () => modal.classList.add('open'));
  $('#closeSettings').addEventListener('click', () => modal.classList.remove('open'));
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.remove('open');
  });
  ['systemPrompt', 'maxTokens', 'temperature', 'topP'].forEach((id) => {
    $('#' + id).addEventListener('change', persistSettings);
  });

  // Health check
  function renderRuntime(runtime) {
    const healthy = runtime.status === 'healthy';
    const config = runtime.config || {};
    const modelId = runtime.model?.id || config.model_id || 'Miniforge';
    const backend = runtime.model?.backend || config.backend || 'unknown';
    const quantization = runtime.model?.quantization || config.quantization || 'unknown';
    const context = config.n_ctx ? `${config.n_ctx.toLocaleString()} ctx` : 'context unknown';

    runtimePill.textContent = healthy ? 'Ready' : 'Degraded';
    runtimePill.classList.toggle('degraded', !healthy);
    runtimeModel.textContent = modelId;
    runtimeDetails.textContent = `${backend} | ${quantization} | ${context}`;
    statusText.textContent = healthy ? 'Ready' : 'Runtime needs attention';
    statusText.style.color = healthy ? '#10a37f' : '#cc9900';
    statusMeta.textContent = runtime.load_error || `${backend} | ${quantization}`;
  }

  async function checkHealth() {
    try {
      const res = await fetch('/api/runtime');
      const data = await res.json();
      renderRuntime(data);
    } catch {
      statusText.textContent = 'Offline';
      statusText.style.color = '#cc3333';
      statusMeta.textContent = 'Could not reach server';
      runtimePill.textContent = 'Offline';
      runtimePill.classList.add('degraded');
    }
  }
  checkHealth();
  setInterval(checkHealth, 5000);

  // Init
  loadSettings();
  if (conversations.length === 0) createConversation();
  else { currentId = conversations[conversations.length - 1].id; renderSidebar(); renderMessages(); }
})();
