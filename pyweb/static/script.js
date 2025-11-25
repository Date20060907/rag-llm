// Global variable to store selected database IDs
let selectedDatabases = [];
// Array to store selected files
let selectedFiles = [];

// Settings functionality
let isSettingsOpen = false;
let chatSettings = {
    n_predict: 512,
    temperature: 0.7,
    top_k: 40,
    rag_k: 3,
    rag_sim_threshold: 0.3
};

// Auto-refresh database list every 30 seconds
let databaseRefreshInterval;

document.getElementById('chat-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (!message) return;

    // Add user message (plaintext)
    addMessage(message, 'user', false);

    // Send to backend with all parameters
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            message: message,
            selected_databases: selectedDatabases,
            rag_parameters: {
                n_predict: chatSettings.n_predict,
                temperature: chatSettings.temperature,
                top_k: chatSettings.top_k,
                rag_k: chatSettings.rag_k,
                rag_sim_threshold: chatSettings.rag_sim_threshold
            }
        })
    });

    const data = await response.json();
    // Add AI response (render as HTML) with typing animation
    addMessage(data.response, 'ai', true);

    input.value = '';
});

function addMessage(text, sender, isHtml = false) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'ai-message');

    if (isHtml) {
        // For AI messages, use typing animation
        if (sender === 'ai') {
            messageDiv.innerHTML = '<div class="typing-indicator">Afina печатает<span>.</span><span>.</span><span>.</span></div>';
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // Start typing animation after a short delay
            setTimeout(() => {
                typeText(messageDiv, text);
            }, 500);
        } else {
            messageDiv.innerHTML = text;
            chatBox.appendChild(messageDiv);
        }
    } else {
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

// Typing animation for AI responses
function typeText(container, text) {
    // Remove typing indicator
    container.innerHTML = '';
    
    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    container.appendChild(contentDiv);
    
    let i = 0;
    const speed = 10; // typing speed in ms
    
    function typeWriter() {
        if (i < text.length) {
            // Handle HTML tags properly
            if (text.charAt(i) === '<') {
                // Find the end of the tag
                const tagEnd = text.indexOf('>', i);
                if (tagEnd !== -1) {
                    contentDiv.innerHTML += text.substring(i, tagEnd + 1);
                    i = tagEnd + 1;
                } else {
                    contentDiv.innerHTML += text.charAt(i);
                    i++;
                }
            } else {
                contentDiv.innerHTML += text.charAt(i);
                i++;
            }
            
            // Scroll to bottom as text is added
            const chatBox = document.getElementById('chat-box');
            chatBox.scrollTop = chatBox.scrollHeight;
            
            setTimeout(typeWriter, speed);
        }
    }
    
    typeWriter();
}

// Drag and drop functionality
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const fileList = document.getElementById('file-list');
const uploadBtn = document.getElementById('upload-btn');
const dbNameInput = document.getElementById('db-name');
const selectedFilesContainer = document.getElementById('selected-files');

browseBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', handleFiles);

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files } });
}

function handleFiles(e) {
    const files = e.target.files;
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
            // Add to selected files if not already present
            if (!selectedFiles.some(f => f.name === file.name)) {
                selectedFiles.push(file);
            }
        }
    }
    updateSelectedFilesDisplay();
    updateUploadButtonState();
}

function updateSelectedFilesDisplay() {
    selectedFilesContainer.innerHTML = '';
    if (selectedFiles.length > 0) {
        selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'selected-file-item';
            fileItem.innerHTML = `
                <span>${file.name}</span>
                <span class="remove-file" data-index="${index}" style="color: red; cursor: pointer;">&times;</span>
            `;
            selectedFilesContainer.appendChild(fileItem);
            
            // Add remove event
            fileItem.querySelector('.remove-file').addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                selectedFiles.splice(index, 1);
                updateSelectedFilesDisplay();
                updateUploadButtonState();
            });
        });
    }
}

function updateUploadButtonState() {
    const dbName = dbNameInput.value.trim();
    uploadBtn.disabled = selectedFiles.length === 0 || !dbName;
}

dbNameInput.addEventListener('input', updateUploadButtonState);

uploadBtn.addEventListener('click', uploadFiles);

async function uploadFiles() {
    if (selectedFiles.length === 0 || !dbNameInput.value.trim()) return;
    
    const formData = new FormData();
    formData.append('db_name', dbNameInput.value.trim());
    
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    // Show uploading status
    const statusItem = showFileStatus(`"${dbNameInput.value.trim()}"`, 'Загрузка...', 'info');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (response.ok) {
            showFileStatus(`"${dbNameInput.value.trim()}"`, result.message, 'success', statusItem);
            // Refresh database list after successful upload
            loadDatabaseList();
            // Clear selected files and reset form
            selectedFiles = [];
            updateSelectedFilesDisplay();
            dbNameInput.value = '';
            updateUploadButtonState();
        } else {
            showFileStatus(`"${dbNameInput.value.trim()}"`, result.error, 'error', statusItem);
        }
    } catch (error) {
        showFileStatus(`"${dbNameInput.value.trim()}"`, 'Ошибка загрузки файлов', 'error', statusItem);
    }
}

function showFileStatus(filename, message, status, existingItem = null) {
    let fileItem;
    
    if (existingItem) {
        fileItem = existingItem;
        const messageSpan = fileItem.querySelector('span:nth-child(2)');
        messageSpan.textContent = message;
        messageSpan.className = status === 'success' ? 'upload-success' : 
                               status === 'error' ? 'upload-error' : '';
    } else {
        fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>${filename}</span>
            <span class="${status === 'success' ? 'upload-success' : 
                          status === 'error' ? 'upload-error' : ''}">${message}</span>
        `;
        fileList.appendChild(fileItem);
    }
    
    return fileItem;
}

// Database list functionality
async function loadDatabaseList() {
    try {
        const response = await fetch('/databases');
        const databases = await response.json();
        
        const databaseList = document.getElementById('database-list');
        databaseList.innerHTML = '';
        
        databases.forEach(db => {
            const item = document.createElement('div');
            item.className = 'database-item';
            item.innerHTML = `
                <input type="checkbox" id="db-${db.id}" data-id="${db.id}" ${selectedDatabases.includes(db.id) ? 'checked' : ''}>
                <label for="db-${db.id}">${db.filename}</label>
            `;
            databaseList.appendChild(item);
            
            // Add event listener to checkbox
            const checkbox = item.querySelector('input');
            checkbox.addEventListener('change', handleDatabaseSelection);
        });
    } catch (error) {
        console.error('Error loading database list:', error);
    }
}

function handleDatabaseSelection(e) {
    const id = parseInt(e.target.dataset.id);
    
    if (e.target.checked) {
        // Add to selected databases if not already present
        if (!selectedDatabases.includes(id)) {
            selectedDatabases.push(id);
        }
    } else {
        // Remove from selected databases
        selectedDatabases = selectedDatabases.filter(dbId => dbId !== id);
    }
    
    // Send updated selection to backend
    updateSelectedDatabases();
}

async function updateSelectedDatabases() {
    try {
        await fetch('/selected_databases', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ selected: selectedDatabases })
        });
    } catch (error) {
        console.error('Error updating selected databases:', error);
    }
}

// Settings functionality
function createSettingsUI() {
    // Add settings button to chat container
    const chatContainer = document.querySelector('.chat-container');
    const settingsBtn = document.createElement('button');
    settingsBtn.id = 'settings-btn';
    settingsBtn.innerHTML = '⚙️ Настройки';
    settingsBtn.className = 'settings-btn';
    chatContainer.insertBefore(settingsBtn, chatContainer.firstChild.nextSibling); // After h1

    // Create settings panel
    const settingsPanel = document.createElement('div');
    settingsPanel.id = 'settings-panel';
    settingsPanel.className = 'settings-panel';
    settingsPanel.innerHTML = `
        <h3>Параметры генерации</h3>
        <div class="setting-group">
            <label for="n-predict">Количество токенов:</label>
            <input type="number" id="n-predict" min="1" max="2048" value="${chatSettings.n_predict}">
        </div>
        <div class="setting-group">
            <label for="temperature">Температура:</label>
            <input type="number" id="temperature" min="0" max="2" step="0.1" value="${chatSettings.temperature}">
        </div>
        <div class="setting-group">
            <label for="top-k">Top-K:</label>
            <input type="number" id="top-k" min="1" max="100" value="${chatSettings.top_k}">
        </div>
        <div class="setting-group">
            <label for="rag-k">RAG K (количество контекстов):</label>
            <input type="number" id="rag-k" min="1" max="10" value="${chatSettings.rag_k}">
        </div>
        <div class="setting-group">
            <label for="rag-sim-threshold">Порог схожести RAG:</label>
            <input type="number" id="rag-sim-threshold" min="0" max="1" step="0.1" value="${chatSettings.rag_sim_threshold}">
        </div>
        <div class="settings-actions">
            <button id="save-settings">Сохранить</button>
            <button id="close-settings">Закрыть</button>
        </div>
    `;
    chatContainer.appendChild(settingsPanel);

    // Add event listeners
    settingsBtn.addEventListener('click', toggleSettings);
    document.getElementById('close-settings').addEventListener('click', toggleSettings);
    document.getElementById('save-settings').addEventListener('click', saveSettings);
}

function toggleSettings() {
    const panel = document.getElementById('settings-panel');
    isSettingsOpen = !isSettingsOpen;
    panel.style.display = isSettingsOpen ? 'block' : 'none';
}

function saveSettings() {
    chatSettings.n_predict = parseInt(document.getElementById('n-predict').value);
    chatSettings.temperature = parseFloat(document.getElementById('temperature').value);
    chatSettings.top_k = parseInt(document.getElementById('top-k').value);
    chatSettings.rag_k = parseInt(document.getElementById('rag-k').value);
    chatSettings.rag_sim_threshold = parseFloat(document.getElementById('rag-sim-threshold').value);
    
    // Save to localStorage
    localStorage.setItem('chatSettings', JSON.stringify(chatSettings));
    
    toggleSettings();
}

// Load settings from localStorage
function loadSettings() {
    const saved = localStorage.getItem('chatSettings');
    if (saved) {
        chatSettings = JSON.parse(saved);
        if (document.getElementById('n-predict')) {
            document.getElementById('n-predict').value = chatSettings.n_predict;
            document.getElementById('temperature').value = chatSettings.temperature;
            document.getElementById('top-k').value = chatSettings.top_k;
            document.getElementById('rag-k').value = chatSettings.rag_k;
            document.getElementById('rag-sim-threshold').value = chatSettings.rag_sim_threshold;
        }
    }
}

// Start auto-refresh of database list
function startDatabaseAutoRefresh() {
    // Load initial database list
    loadDatabaseList();
    
    // Set up periodic refresh every 30 seconds
    databaseRefreshInterval = setInterval(loadDatabaseList, 30000);
}

// Stop auto-refresh when needed
function stopDatabaseAutoRefresh() {
    if (databaseRefreshInterval) {
        clearInterval(databaseRefreshInterval);
    }
}

// Load database list when page loads
document.addEventListener('DOMContentLoaded', function() {
    createSettingsUI();
    loadSettings();
    startDatabaseAutoRefresh();
});

// Clean up interval when page is unloaded
window.addEventListener('beforeunload', stopDatabaseAutoRefresh);
