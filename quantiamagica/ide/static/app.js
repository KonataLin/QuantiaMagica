/**
 * QuantiaMagica IDE - 主应用程序
 */

// 全局状态
const state = {
    workspace: null,
    files: [],
    openTabs: [],
    activeTab: null,
    editor: null,
    completions: [],
    isRunning: false,
    undoStack: {},  // 每个文件的撤销栈
};

// 简化版 - 先不用Monaco，用textarea测试
console.log('app.js 已加载');

// 隐藏加载提示
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM加载完成');
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
    
    // 显示欢迎页
    showWelcome();
    
    // 加载补全信息
    loadCompletions();
});

// 创建简单编辑器（临时用textarea代替Monaco）
function createSimpleEditor() {
    const container = document.getElementById('editor-container');
    container.innerHTML = '';
    
    const textarea = document.createElement('textarea');
    textarea.id = 'simple-editor';
    textarea.style.cssText = `
        width: 100%; height: 100%; 
        background: #1e1e1e; color: #d4d4d4;
        border: none; outline: none; resize: none;
        font-family: 'Consolas', monospace; font-size: 14px;
        padding: 16px; line-height: 1.5;
        tab-size: 4;
    `;
    textarea.spellcheck = false;
    
    // 支持Tab键
    textarea.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = this.selectionStart;
            const end = this.selectionEnd;
            this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
            this.selectionStart = this.selectionEnd = start + 4;
        }
    });
    
    // 监听变化
    textarea.addEventListener('input', function() {
        if (state.activeTab) {
            const tab = state.openTabs.find(t => t.path === state.activeTab);
            if (tab) {
                tab.modified = true;
                tab.content = this.value;
                updateTabUI();
            }
        }
    });
    
    container.appendChild(textarea);
    return textarea;
}

// 模拟state.editor的API
state.editor = {
    _textarea: null,
    getValue: function() { return this._textarea ? this._textarea.value : ''; },
    setValue: function(v) { if (this._textarea) this._textarea.value = v; },
    setModel: function() {},
    onDidChangeModelContent: function() {},
    onDidChangeCursorPosition: function() {},
};

// 获取补全类型（简化版不需要）
function getMonacoKind(kind) {
    return 0;
}

// 显示欢迎页面
function showWelcome() {
    const container = document.getElementById('editor-container');
    container.innerHTML = `
        <div class="welcome-page">
            <div class="welcome-logo">QM</div>
            <h1 class="welcome-title">QuantiaMagica IDE</h1>
            <p class="welcome-subtitle">专为ADC仿真设计的开发环境</p>
            <div class="welcome-actions">
                <button class="welcome-btn" onclick="showModal('modal-open-folder')">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                    </svg>
                    打开文件夹
                </button>
                <button class="welcome-btn" onclick="createNewFile()">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="12" y1="18" x2="12" y2="12"></line>
                        <line x1="9" y1="15" x2="15" y2="15"></line>
                    </svg>
                    新建文件
                </button>
            </div>
        </div>
    `;
}

// 加载补全信息
async function loadCompletions() {
    try {
        const res = await fetch('/api/completions');
        const data = await res.json();
        state.completions = data.completions || [];
    } catch (e) {
        console.error('Failed to load completions:', e);
    }
}

// 刷新文件列表
async function refreshFiles() {
    try {
        const res = await fetch('/api/files');
        const data = await res.json();
        state.files = data.files || [];
        state.workspace = data.workspace;
        
        document.getElementById('workspace-path').textContent = 
            state.workspace ? state.workspace : '未打开文件夹';
        
        renderFileTree();
    } catch (e) {
        console.error('Failed to refresh files:', e);
    }
}

// 渲染文件树
function renderFileTree() {
    const container = document.getElementById('file-tree');
    
    if (!state.files.length) {
        container.innerHTML = '<div style="padding: 20px; color: var(--text-muted); font-size: 12px;">没有文件</div>';
        return;
    }
    
    container.innerHTML = renderTreeItems(state.files, 0);
}

// 渲染树节点
function renderTreeItems(items, depth) {
    return items.map(item => {
        const indent = depth * 16;
        const isDir = item.type === 'directory';
        const icon = getFileIcon(item.name, isDir);
        
        if (isDir) {
            return `
                <div class="tree-item" style="padding-left: ${indent + 8}px" 
                     onclick="toggleFolder(this, '${item.path}')" 
                     oncontextmenu="showContextMenu(event, '${item.path}', true)">
                    <span class="arrow">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M8 5l8 7-8 7V5z"/>
                        </svg>
                    </span>
                    ${icon}
                    <span class="name">${item.name}</span>
                </div>
                <div class="tree-children" data-path="${item.path}">
                    ${item.children ? renderTreeItems(item.children, depth + 1) : ''}
                </div>
            `;
        } else {
            return `
                <div class="tree-item" style="padding-left: ${indent + 24}px" 
                     onclick="openFile('${item.path}')"
                     oncontextmenu="showContextMenu(event, '${item.path}', false)">
                    ${icon}
                    <span class="name">${item.name}</span>
                </div>
            `;
        }
    }).join('');
}

// 获取文件图标
function getFileIcon(name, isDir) {
    if (isDir) {
        return `<span class="icon icon-folder">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
            </svg>
        </span>`;
    }
    
    const ext = name.split('.').pop().toLowerCase();
    const icons = {
        py: `<span class="icon icon-python">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
            </svg>
        </span>`,
        json: `<span class="icon icon-json">{ }</span>`,
        md: `<span class="icon icon-md">M</span>`,
    };
    
    return icons[ext] || `<span class="icon icon-file">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
    </span>`;
}

// 切换文件夹展开/折叠
function toggleFolder(element, path) {
    const arrow = element.querySelector('.arrow');
    const children = document.querySelector(`.tree-children[data-path="${path}"]`);
    
    if (children) {
        arrow.classList.toggle('expanded');
        children.classList.toggle('expanded');
    }
}

// 打开文件
async function openFile(path) {
    // 检查是否已打开
    let tab = state.openTabs.find(t => t.path === path);
    
    if (!tab) {
        try {
            const res = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
            const data = await res.json();
            
            if (data.error) {
                showOutput(`错误: ${data.error}`, 'error');
                return;
            }
            
            tab = {
                path: path,
                name: path.split(/[/\\]/).pop(),
                content: data.content,
                originalContent: data.content,
                modified: false,
            };
            
            state.openTabs.push(tab);
        } catch (e) {
            showOutput(`无法打开文件: ${e.message}`, 'error');
            return;
        }
    }
    
    // 激活标签
    state.activeTab = path;
    
    // 恢复编辑器
    const container = document.getElementById('editor-container');
    if (container.querySelector('.welcome-page') || !state.editor._textarea) {
        const textarea = createSimpleEditor();
        state.editor._textarea = textarea;
        textarea.value = tab.content;
    } else {
        state.editor._textarea.value = tab.content;
    }
    
    updateTabUI();
    updateStatus();
    
    // 高亮文件树中的项
    document.querySelectorAll('.tree-item').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tree-item').forEach(el => {
        if (el.getAttribute('onclick')?.includes(path)) {
            el.classList.add('active');
        }
    });
}

// 获取语言
function getLanguage(path) {
    const ext = path.split('.').pop().toLowerCase();
    const langs = {
        py: 'python',
        js: 'javascript',
        json: 'json',
        html: 'html',
        css: 'css',
        md: 'markdown',
        txt: 'plaintext',
    };
    return langs[ext] || 'plaintext';
}

// 更新标签UI
function updateTabUI() {
    const tabsContainer = document.getElementById('tabs');
    
    tabsContainer.innerHTML = state.openTabs.map(tab => `
        <div class="tab ${tab.path === state.activeTab ? 'active' : ''} ${tab.modified ? 'modified' : ''}"
             onclick="openFile('${tab.path}')">
            <span class="tab-name">${tab.name}</span>
            <span class="tab-close" onclick="event.stopPropagation(); closeTab('${tab.path}')">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </span>
        </div>
    `).join('');
}

// 关闭标签
function closeTab(path) {
    const tab = state.openTabs.find(t => t.path === path);
    
    if (tab && tab.modified) {
        if (!confirm(`${tab.name} 有未保存的更改，确定关闭吗？`)) {
            return;
        }
    }
    
    state.openTabs = state.openTabs.filter(t => t.path !== path);
    
    if (state.activeTab === path) {
        if (state.openTabs.length > 0) {
            openFile(state.openTabs[state.openTabs.length - 1].path);
        } else {
            state.activeTab = null;
            showWelcome();
        }
    }
    
    updateTabUI();
}

// 保存文件
async function saveFile() {
    if (!state.activeTab) return;
    
    const tab = state.openTabs.find(t => t.path === state.activeTab);
    if (!tab) return;
    
    try {
        const res = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                path: tab.path,
                content: tab.content,
            }),
        });
        
        const data = await res.json();
        
        if (data.success) {
            tab.modified = false;
            tab.originalContent = tab.content;
            updateTabUI();
            showOutput(`已保存: ${tab.name}`, 'success');
        } else {
            showOutput(`保存失败: ${data.error}`, 'error');
        }
    } catch (e) {
        showOutput(`保存失败: ${e.message}`, 'error');
    }
}

// 运行代码
async function runCode() {
    if (state.isRunning) return;
    
    let code = '';
    
    if (state.activeTab && state.editor) {
        code = state.editor.getValue();
    } else {
        showOutput('请先打开一个Python文件', 'error');
        return;
    }
    
    if (!code.trim()) {
        showOutput('代码为空', 'error');
        return;
    }
    
    state.isRunning = true;
    document.getElementById('btn-run').style.display = 'none';
    document.getElementById('btn-stop').style.display = 'flex';
    
    showOutput('>>> 运行中...\n', 'info');
    
    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code }),
        });
        
        const data = await res.json();
        
        if (data.output) {
            appendOutput(data.output);
        }
        
        if (data.error) {
            appendOutput(data.error, 'error');
        }
        
        if (data.returncode === 0) {
            appendOutput('\n>>> 运行完成', 'success');
        } else {
            appendOutput(`\n>>> 运行结束 (返回码: ${data.returncode})`, 'error');
        }
        
    } catch (e) {
        showOutput(`运行失败: ${e.message}`, 'error');
    } finally {
        state.isRunning = false;
        document.getElementById('btn-run').style.display = 'flex';
        document.getElementById('btn-stop').style.display = 'none';
    }
}

// 显示输出
function showOutput(text, type = '') {
    const terminal = document.getElementById('terminal-content');
    terminal.innerHTML = `<div class="output-line ${type}">${escapeHtml(text)}</div>`;
    terminal.scrollTop = terminal.scrollHeight;
}

// 追加输出
function appendOutput(text, type = '') {
    const terminal = document.getElementById('terminal-content');
    terminal.innerHTML += `<div class="output-line ${type}">${escapeHtml(text)}</div>`;
    terminal.scrollTop = terminal.scrollHeight;
}

// 清空终端
function clearTerminal() {
    document.getElementById('terminal-content').innerHTML = '';
}

// HTML转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 更新状态栏
function updateStatus() {
    const tab = state.openTabs.find(t => t.path === state.activeTab);
    if (tab) {
        document.getElementById('status-file').textContent = tab.name;
        document.getElementById('status-lang').textContent = 
            getLanguage(tab.path).charAt(0).toUpperCase() + getLanguage(tab.path).slice(1);
    }
}

// 显示/隐藏模态框
function showModal(id) {
    document.getElementById(id).style.display = 'flex';
    const input = document.querySelector(`#${id} input`);
    if (input) {
        input.focus();
        input.select();
    }
}

function closeModal(id) {
    document.getElementById(id).style.display = 'none';
}

// 打开文件夹 - 优先使用Qt原生对话框
async function openFolder() {
    // 尝试使用Qt原生对话框
    if (typeof QWebChannel !== 'undefined' && window.pybridge) {
        window.pybridge.openFolderDialog();
        closeModal('modal-open-folder');
        return;
    }
    
    // 回退到手动输入路径
    const path = document.getElementById('input-folder-path').value.trim();
    
    if (!path) {
        alert('请输入文件夹路径');
        return;
    }
    
    await setWorkspace(path);
}

// 设置工作目录
async function setWorkspace(path) {
    try {
        const res = await fetch('/api/workspace', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        
        const data = await res.json();
        
        if (data.success) {
            closeModal('modal-open-folder');
            refreshFiles();
            showOutput(`已打开: ${path}`, 'success');
        } else {
            alert(`无法打开文件夹: ${data.error}`);
        }
    } catch (e) {
        alert(`错误: ${e.message}`);
    }
}

// Qt调用此函数设置工作目录
function setWorkspaceFromQt(path) {
    setWorkspace(path);
}

// 新建文件/文件夹
let createType = 'file';
let createParent = '';

function createNewFile(parent = '') {
    createType = 'file';
    createParent = parent;
    document.getElementById('modal-new-title').textContent = '新建文件';
    document.getElementById('input-new-name').placeholder = 'example.py';
    document.getElementById('input-new-name').value = '';
    showModal('modal-new-file');
}

function createNewFolder(parent = '') {
    createType = 'folder';
    createParent = parent;
    document.getElementById('modal-new-title').textContent = '新建文件夹';
    document.getElementById('input-new-name').placeholder = 'folder_name';
    document.getElementById('input-new-name').value = '';
    showModal('modal-new-file');
}

async function createNew() {
    const name = document.getElementById('input-new-name').value.trim();
    
    if (!name) {
        alert('请输入名称');
        return;
    }
    
    try {
        const res = await fetch('/api/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                parent: createParent,
                isDirectory: createType === 'folder',
            }),
        });
        
        const data = await res.json();
        
        if (data.success) {
            closeModal('modal-new-file');
            refreshFiles();
            
            if (createType === 'file') {
                const path = createParent ? `${createParent}/${name}` : name;
                setTimeout(() => openFile(path), 100);
            }
        } else {
            alert(`创建失败: ${data.error}`);
        }
    } catch (e) {
        alert(`错误: ${e.message}`);
    }
}

// 右键菜单
let contextPath = '';
let contextIsDir = false;

function showContextMenu(event, path, isDir) {
    event.preventDefault();
    event.stopPropagation();
    
    contextPath = path;
    contextIsDir = isDir;
    
    const menu = document.getElementById('context-menu');
    menu.style.display = 'block';
    menu.style.left = `${event.clientX}px`;
    menu.style.top = `${event.clientY}px`;
}

function hideContextMenu() {
    document.getElementById('context-menu').style.display = 'none';
}

// 处理右键菜单操作
document.getElementById('context-menu').addEventListener('click', async (e) => {
    const action = e.target.dataset.action;
    hideContextMenu();
    
    if (action === 'new-file') {
        createNewFile(contextIsDir ? contextPath : '');
    } else if (action === 'new-folder') {
        createNewFolder(contextIsDir ? contextPath : '');
    } else if (action === 'rename') {
        const newName = prompt('新名称:', contextPath.split(/[/\\]/).pop());
        if (newName && newName !== contextPath.split(/[/\\]/).pop()) {
            try {
                const res = await fetch('/api/rename', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ oldPath: contextPath, newName }),
                });
                const data = await res.json();
                if (data.success) {
                    refreshFiles();
                } else {
                    alert(`重命名失败: ${data.error}`);
                }
            } catch (e) {
                alert(`错误: ${e.message}`);
            }
        }
    } else if (action === 'delete') {
        if (confirm(`确定删除 "${contextPath}" 吗？`)) {
            try {
                const res = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: contextPath }),
                });
                const data = await res.json();
                if (data.success) {
                    // 关闭已打开的标签
                    if (!contextIsDir) {
                        closeTab(contextPath);
                    }
                    refreshFiles();
                } else {
                    alert(`删除失败: ${data.error}`);
                }
            } catch (e) {
                alert(`错误: ${e.message}`);
            }
        }
    }
});

// 点击其他地方关闭菜单
document.addEventListener('click', hideContextMenu);

// 键盘快捷键
document.addEventListener('keydown', (e) => {
    // Ctrl+S: 保存
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        saveFile();
    }
    
    // Ctrl+N: 新建文件
    if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        createNewFile();
    }
    
    // F5: 运行
    if (e.key === 'F5') {
        e.preventDefault();
        runCode();
    }
    
    // Escape: 关闭模态框
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal').forEach(m => m.style.display = 'none');
        hideContextMenu();
    }
});

// 模态框输入框回车确认
document.getElementById('input-folder-path').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') openFolder();
});

document.getElementById('input-new-name').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') createNew();
});

// 侧边栏拖拽调整宽度
const sidebar = document.getElementById('sidebar');
const sidebarResizer = document.getElementById('sidebar-resizer');

sidebarResizer.addEventListener('mousedown', (e) => {
    e.preventDefault();
    sidebarResizer.classList.add('dragging');
    
    const startX = e.clientX;
    const startWidth = sidebar.offsetWidth;
    
    const onMouseMove = (e) => {
        const newWidth = startWidth + (e.clientX - startX);
        sidebar.style.width = `${Math.max(180, Math.min(400, newWidth))}px`;
    };
    
    const onMouseUp = () => {
        sidebarResizer.classList.remove('dragging');
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    };
    
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
});

// 终端面板拖拽调整高度
const terminalPanel = document.getElementById('terminal-panel');
const terminalResizer = document.getElementById('terminal-resizer');

terminalResizer.addEventListener('mousedown', (e) => {
    e.preventDefault();
    terminalResizer.classList.add('dragging');
    
    const startY = e.clientY;
    const startHeight = terminalPanel.offsetHeight;
    
    const onMouseMove = (e) => {
        const newHeight = startHeight - (e.clientY - startY);
        terminalPanel.style.height = `${Math.max(100, Math.min(window.innerHeight * 0.5, newHeight))}px`;
    };
    
    const onMouseUp = () => {
        terminalResizer.classList.remove('dragging');
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    };
    
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
});

// 工具栏按钮绑定
document.getElementById('btn-new-file').addEventListener('click', () => createNewFile());
document.getElementById('btn-new-folder').addEventListener('click', () => createNewFolder());
document.getElementById('btn-save').addEventListener('click', saveFile);
document.getElementById('btn-run').addEventListener('click', runCode);
document.getElementById('btn-open-folder').addEventListener('click', () => showModal('modal-open-folder'));
document.getElementById('btn-refresh').addEventListener('click', refreshFiles);
document.getElementById('btn-clear-terminal').addEventListener('click', clearTerminal);

// 初始化Qt WebChannel（如果在Qt环境中）
function initQtBridge() {
    if (typeof QWebChannel !== 'undefined') {
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.pybridge = channel.objects.pybridge;
            console.log('Qt WebChannel initialized');
        });
    }
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    initQtBridge();
});

// 初始化
refreshFiles();
