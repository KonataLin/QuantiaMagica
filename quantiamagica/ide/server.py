"""
QuantiaMagica IDE - 桌面应用

使用PyQt6 WebEngineView内嵌HTML实现VS Code风格IDE
"""

import os
import sys
import json
import subprocess
import threading
import shutil
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# 获取IDE资源目录
IDE_DIR = Path(__file__).parent
STATIC_DIR = IDE_DIR / 'static'

# 全局状态
_workspace_path = None
_browse_requested = False


class APIHandler(SimpleHTTPRequestHandler):
    """API请求处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)
    
    def do_GET(self):
        """处理GET请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/' or path == '/index.html':
            self._serve_file('index.html', 'text/html')
        elif path == '/style.css':
            self._serve_file('style.css', 'text/css')
        elif path == '/app.js':
            self._serve_file('app.js', 'application/javascript')
        elif path == '/api/files':
            self._handle_list_files()
        elif path == '/api/file':
            query = parse_qs(parsed.query)
            filepath = query.get('path', [''])[0]
            self._handle_read_file(filepath)
        elif path == '/api/completions':
            self._handle_completions()
        else:
            super().do_GET()
    
    def do_POST(self):
        """处理POST请求"""
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        
        path = urlparse(self.path).path
        
        if path == '/api/save':
            self._handle_save_file(data)
        elif path == '/api/workspace':
            self._handle_set_workspace(data)
        elif path == '/api/browse':
            self._handle_browse_folder(data)
        elif path == '/api/run':
            self._handle_run_code(data)
        elif path == '/api/delete':
            self._handle_delete(data)
        elif path == '/api/rename':
            self._handle_rename(data)
        elif path == '/api/terminal':
            self._handle_terminal(data)
        else:
            self._send_json({'error': 'Unknown endpoint'}, 404)
    
    def _serve_file(self, filename, content_type):
        """提供静态文件"""
        filepath = STATIC_DIR / filename
        if filepath.exists():
            self.send_response(200)
            self.send_header('Content-type', f'{content_type}; charset=utf-8')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)
    
    def _send_json(self, data, status=200):
        """发送JSON响应"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def _handle_list_files(self):
        """列出工作目录文件"""
        global _workspace_path
        if not _workspace_path or not os.path.exists(_workspace_path):
            self._send_json({'files': [], 'workspace': None})
            return
        
        def scan_dir(path, prefix=''):
            items = []
            try:
                for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())):
                    if entry.name.startswith('.') or entry.name == '__pycache__':
                        continue
                    
                    rel_path = os.path.join(prefix, entry.name) if prefix else entry.name
                    
                    if entry.is_dir():
                        children = scan_dir(entry.path, rel_path)
                        items.append({
                            'name': entry.name,
                            'path': rel_path,
                            'type': 'directory',
                            'children': children
                        })
                    else:
                        items.append({
                            'name': entry.name,
                            'path': rel_path,
                            'type': 'file'
                        })
            except PermissionError:
                pass
            return items
        
        files = scan_dir(_workspace_path)
        self._send_json({'files': files, 'workspace': _workspace_path})
    
    def _handle_read_file(self, filepath):
        """读取文件内容"""
        global _workspace_path
        if not filepath or not _workspace_path:
            self._send_json({'error': 'No file specified'}, 400)
            return
        
        full_path = os.path.join(_workspace_path, filepath)
        
        if not os.path.exists(full_path):
            self._send_json({'error': 'File not found'}, 404)
            return
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._send_json({'content': content, 'path': filepath})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_save_file(self, data):
        """保存文件"""
        global _workspace_path
        filepath = data.get('path', '')
        content = data.get('content', '')
        
        if not filepath or not _workspace_path:
            self._send_json({'error': 'No file specified'}, 400)
            return
        
        full_path = os.path.join(_workspace_path, filepath)
        
        try:
            os.makedirs(os.path.dirname(full_path) or '.', exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self._send_json({'success': True, 'path': filepath})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_create(self, data):
        """创建文件或文件夹"""
        global _workspace_path
        name = data.get('name', '')
        parent = data.get('parent', '')
        is_dir = data.get('isDirectory', False)
        
        if not name or not _workspace_path:
            self._send_json({'error': 'No name specified'}, 400)
            return
        
        full_path = os.path.join(_workspace_path, parent, name) if parent else os.path.join(_workspace_path, name)
        
        try:
            if is_dir:
                os.makedirs(full_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(full_path) or '.', exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('')
            self._send_json({'success': True})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_delete(self, data):
        """删除文件或文件夹"""
        global _workspace_path
        filepath = data.get('path', '')
        
        if not filepath or not _workspace_path:
            self._send_json({'error': 'No path specified'}, 400)
            return
        
        full_path = os.path.join(_workspace_path, filepath)
        
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
            self._send_json({'success': True})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_rename(self, data):
        """重命名"""
        global _workspace_path
        old_path = data.get('oldPath', '')
        new_name = data.get('newName', '')
        
        if not old_path or not new_name or not _workspace_path:
            self._send_json({'error': 'Invalid parameters'}, 400)
            return
        
        full_old = os.path.join(_workspace_path, old_path)
        full_new = os.path.join(os.path.dirname(full_old), new_name)
        
        try:
            os.rename(full_old, full_new)
            self._send_json({'success': True})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_set_workspace(self, data):
        """设置工作目录"""
        global _workspace_path
        path = data.get('path', '')
        
        if path and os.path.isdir(path):
            _workspace_path = path
            self._send_json({'success': True, 'workspace': path})
        else:
            self._send_json({'error': 'Invalid directory'}, 400)
    
    def _handle_browse_folder(self, data):
        """浏览文件夹（触发Qt对话框）"""
        global _browse_requested
        _browse_requested = True
        self._send_json({'browse': True})
    
    def _handle_delete(self, data):
        """删除文件或文件夹"""
        global _workspace_path
        filepath = data.get('path', '')
        
        if not filepath or not _workspace_path:
            self._send_json({'error': 'No path specified'}, 400)
            return
        
        full_path = os.path.join(_workspace_path, filepath)
        
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
            self._send_json({'success': True})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_rename(self, data):
        """重命名文件或文件夹"""
        global _workspace_path
        old_path = data.get('oldPath', '')
        new_name = data.get('newName', '')
        
        if not old_path or not new_name or not _workspace_path:
            self._send_json({'error': 'Invalid parameters'}, 400)
            return
        
        full_old = os.path.join(_workspace_path, old_path)
        full_new = os.path.join(os.path.dirname(full_old), new_name)
        
        try:
            os.rename(full_old, full_new)
            self._send_json({'success': True, 'newPath': os.path.relpath(full_new, _workspace_path)})
        except Exception as e:
            self._send_json({'error': str(e)}, 500)
    
    def _handle_terminal(self, data):
        """执行终端命令"""
        global _workspace_path
        command = data.get('command', '')
        
        if not command:
            self._send_json({'error': 'No command'}, 400)
            return
        
        try:
            cwd = _workspace_path or os.getcwd()
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd
            )
            self._send_json({
                'output': result.stdout + result.stderr,
                'returncode': result.returncode
            })
        except subprocess.TimeoutExpired:
            self._send_json({'output': '命令超时(30s)', 'returncode': -1})
        except Exception as e:
            self._send_json({'output': str(e), 'returncode': -1})
    
    def _handle_run_code(self, data):
        """运行Python代码"""
        global _workspace_path
        code = data.get('code', '')
        file_path = data.get('file', '')
        
        if not code:
            self._send_json({'error': 'No code provided'}, 400)
            return
        
        try:
            import tempfile
            project_root = Path(__file__).parent.parent.parent
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')
            
            # 标准化路径
            if file_path:
                file_path = os.path.normpath(file_path)
            
            # 判断是否使用原文件
            use_original = file_path and os.path.isfile(file_path)
            
            if use_original:
                # 保存并运行原文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                run_file = file_path
                cwd = os.path.dirname(file_path)
            else:
                # 创建临时文件
                cwd = _workspace_path or str(project_root)
                fd, run_file = tempfile.mkstemp(suffix='.py', dir=cwd, text=True)
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(code)
            
            # 运行
            result = subprocess.run(
                [sys.executable, run_file],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd,
                env=env
            )
            
            # 清理临时文件
            if not use_original:
                try:
                    os.unlink(run_file)
                except:
                    pass
            
            self._send_json({
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode
            })
            
        except subprocess.TimeoutExpired:
            self._send_json({'error': '运行超时(120s)', 'output': '', 'returncode': -1})
        except Exception as e:
            self._send_json({'error': f'运行错误: {e}', 'output': '', 'returncode': -1})
    
    def _handle_completions(self):
        """获取API补全"""
        completions = get_quantiamagica_completions()
        self._send_json({'completions': completions})
    
    def log_message(self, format, *args):
        """静默日志"""
        pass


def get_quantiamagica_completions():
    """获取quantiamagica模块的补全信息"""
    completions = []
    
    # 主要类
    classes = [
        {'label': 'SARADC', 'kind': 'Class', 'detail': '逐次逼近型ADC', 
         'insertText': 'SARADC(bits=${1:12}, vref=${2:1.0})'},
        {'label': 'PipelineADC', 'kind': 'Class', 'detail': '流水线ADC',
         'insertText': 'PipelineADC(bits=${1:12}, stages=${2:4})'},
        {'label': 'SigmaDeltaADC', 'kind': 'Class', 'detail': 'Sigma-Delta ADC',
         'insertText': 'SigmaDeltaADC(order=${1:2}, bits=${2:1}, osr=${3:64})'},
        {'label': 'Signal', 'kind': 'Class', 'detail': '信号生成器',
         'insertText': 'Signal'},
        {'label': 'GeneticOptimizer', 'kind': 'Class', 'detail': '遗传算法优化器',
         'insertText': 'GeneticOptimizer(genes, fitness_fn, maximize=${1:True})'},
        {'label': 'Gene', 'kind': 'Class', 'detail': '优化基因定义',
         'insertText': "Gene('${1:name}', ${2:min}, ${3:max}, '${4:float}')"},
    ]
    
    # 事件类
    events = [
        {'label': 'SamplingEvent', 'kind': 'Class', 'detail': '采样事件'},
        {'label': 'ComparatorEvent', 'kind': 'Class', 'detail': '比较器事件'},
        {'label': 'QuantizerEvent', 'kind': 'Class', 'detail': '量化器事件'},
        {'label': 'CapacitorSwitchEvent', 'kind': 'Class', 'detail': '电容切换事件'},
        {'label': 'StageEvent', 'kind': 'Class', 'detail': '流水线级事件'},
    ]
    
    # 方法
    methods = [
        {'label': 'sim', 'kind': 'Method', 'detail': '运行仿真',
         'insertText': 'sim(n_samples=${1:4096}, fs=${2:1e6}, fin=${3:10e3})'},
        {'label': 'enob', 'kind': 'Method', 'detail': '计算有效位数'},
        {'label': 'snr', 'kind': 'Method', 'detail': '计算信噪比'},
        {'label': 'sfdr', 'kind': 'Method', 'detail': '计算无杂散动态范围'},
        {'label': 'plot', 'kind': 'Method', 'detail': '绘制分析图'},
        {'label': 'spectrum', 'kind': 'Method', 'detail': '绘制频谱图'},
        {'label': 'report', 'kind': 'Method', 'detail': '生成分析报告',
         'insertText': "report('${1|all,spectrum,time,static,metrics|}')"},
    ]
    
    # 导入语句
    imports = [
        {'label': 'from quantiamagica import SARADC', 'kind': 'Snippet', 'detail': '导入SAR ADC'},
        {'label': 'from quantiamagica import PipelineADC', 'kind': 'Snippet', 'detail': '导入Pipeline ADC'},
        {'label': 'from quantiamagica import SigmaDeltaADC, QuantizerEvent', 'kind': 'Snippet', 'detail': '导入Sigma-Delta ADC'},
        {'label': 'from quantiamagica.optim import GeneticOptimizer, Gene', 'kind': 'Snippet', 'detail': '导入遗传优化器'},
        {'label': 'from quantiamagica import Signal', 'kind': 'Snippet', 'detail': '导入信号生成器'},
    ]
    
    # 代码模板
    templates = [
        {'label': 'adc_basic', 'kind': 'Snippet', 'detail': 'ADC基础仿真模板',
         'insertText': '''from quantiamagica import SARADC

# 创建ADC
adc = SARADC(bits=12, vref=1.0)

# 运行仿真
adc.sim(n_samples=4096, fs=1e6, fin=10e3)

# 查看结果
print(f"ENOB: {adc.enob():.2f} bits")
adc.report()
'''},
        {'label': 'sd_optimize', 'kind': 'Snippet', 'detail': 'Sigma-Delta优化模板',
         'insertText': '''from quantiamagica import SigmaDeltaADC, QuantizerEvent
from quantiamagica.optim import GeneticOptimizer, Gene
import numpy as np

# 定义基因
genes = [
    Gene('c1', 0.1, 0.5, 'float'),
    Gene('c2', 0.1, 0.5, 'float'),
]

# 适应度函数
def fitness(params):
    sd = SigmaDeltaADC(order=2, bits=1, osr=64)
    # ... 自定义逻辑
    return sd.enob()

# 运行优化
optimizer = GeneticOptimizer(genes, fitness, maximize=True)
result = optimizer.run(population_size=50)
print(f"最优ENOB: {result.best_fitness:.2f} bits")
'''},
    ]
    
    completions.extend(classes)
    completions.extend(events)
    completions.extend(methods)
    completions.extend(imports)
    completions.extend(templates)
    
    return completions


def kill_port(port):
    """杀掉占用指定端口的进程"""
    import socket
    try:
        # 先检查端口是否被占用
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:  # 端口被占用
            if sys.platform == 'win32':
                # Windows: 找到并杀掉进程
                import subprocess
                result = subprocess.run(
                    f'netstat -ano | findstr :{port}',
                    shell=True, capture_output=True, text=True
                )
                for line in result.stdout.strip().split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if parts:
                            pid = parts[-1]
                            subprocess.run(f'taskkill /PID {pid} /F', shell=True, 
                                         capture_output=True)
                            print(f"已关闭旧进程 PID:{pid}")
            else:
                # Linux/Mac
                import subprocess
                subprocess.run(f'fuser -k {port}/tcp', shell=True, capture_output=True)
    except Exception as e:
        pass  # 忽略错误

def start_server(port=8765, workspace=None):
    """启动后端API服务器（在后台线程运行）"""
    import time
    global _workspace_path
    if workspace:
        _workspace_path = workspace
    
    # 先杀掉旧进程
    kill_port(port)
    time.sleep(0.3)
    
    server = HTTPServer(('127.0.0.1', port), APIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    # 等待服务器启动
    time.sleep(0.5)
    return server


def launch(port=8765, workspace=None):
    """
    启动QuantiaMagica IDE桌面应用
    
    使用PyQt6 WebEngineView内嵌HTML界面
    """
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtCore import QUrl, QObject, pyqtSlot, QTimer
    except ImportError:
        print("错误: 需要安装 PyQt6 和 PyQt6-WebEngine")
        print("请运行: pip install PyQt6 PyQt6-WebEngine")
        return
    
    # 启动后端服务器
    print(f"启动服务器: http://127.0.0.1:{port}")
    server = start_server(port, workspace)
    
    # 创建Qt应用
    app = QApplication(sys.argv)
    app.setApplicationName("QuantiaMagica IDE")
    
    # 自定义页面类用于调试
    class DebugPage(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, line, source):
            print(f"[JS] {message} (line {line})")
    
    # 主窗口
    class IDEWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("QuantiaMagica IDE")
            self.setGeometry(100, 100, 1400, 900)
            self.setMinimumSize(800, 600)
            
            # 设置深色背景
            self.setStyleSheet("QMainWindow { background-color: #1e1e1e; }")
            
            # WebView
            self.browser = QWebEngineView()
            self.page = DebugPage(self.browser)
            self.browser.setPage(self.page)
            
            # 启用必要的设置
            settings = self.page.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            
            # 设置WebChannel
            self.channel = QWebChannel()
            self.bridge = BridgeObject(self)
            self.channel.registerObject('pybridge', self.bridge)
            self.page.setWebChannel(self.channel)
            
            # 加载完成/失败信号
            self.browser.loadFinished.connect(self.on_load_finished)
            
            # 先显示一个加载页面
            self.browser.setHtml("""
                <html><body style="background:#1e1e1e;color:#ccc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;font-family:sans-serif;">
                <div style="text-align:center;">
                    <div style="font-size:48px;font-weight:bold;color:#0078d4;">QM</div>
                    <div style="margin-top:16px;">正在初始化...</div>
                </div>
                </body></html>
            """)
            
            self.setCentralWidget(self.browser)
        
        def on_load_finished(self, ok):
            if ok:
                print("页面加载成功!")
                # 注入调试脚本
                self.browser.page().runJavaScript("console.log('页面已加载')")
            else:
                print("页面加载失败!")
                # 显示错误页面
                self.browser.setHtml("""
                    <html><body style="background:#1e1e1e;color:#fff;padding:50px;font-family:sans-serif;">
                    <h1>加载失败</h1>
                    <p>无法加载IDE界面，请检查:</p>
                    <ul>
                        <li>服务器是否正常运行</li>
                        <li>端口8765是否被占用</li>
                    </ul>
                    <p>尝试在浏览器中访问: <a href="http://127.0.0.1:8765" style="color:#0078d4">http://127.0.0.1:8765</a></p>
                    </body></html>
                """)
        
        def open_folder_dialog(self):
            folder = QFileDialog.getExistingDirectory(self, "选择工作目录")
            if folder:
                global _workspace_path
                _workspace_path = folder
                self.browser.page().runJavaScript(
                    f"if(typeof setWorkspaceFromQt==='function')setWorkspaceFromQt('{folder.replace(chr(92), '/')}')"
                )
    
    class BridgeObject(QObject):
        def __init__(self, window):
            super().__init__()
            self.window = window
        
        @pyqtSlot()
        def openFolderDialog(self):
            self.window.open_folder_dialog()
    
    window = IDEWindow()
    window.show()
    
    # 延迟加载完整IDE
    def delayed_load():
        print("加载IDE界面...")
        # 完整HTML，带语法高亮
        html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>QuantiaMagica IDE</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1e1e1e; color: #ccc; font-family: 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; }}
.toolbar {{ height: 40px; background: #2d2d2d; border-bottom: 1px solid #3c3c3c; display: flex; align-items: center; padding: 0 12px; gap: 8px; }}
.logo {{ background: linear-gradient(135deg, #0078d4, #00b4d8); color: white; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
.title {{ color: #9d9d9d; font-size: 13px; }}
.btn {{ background: transparent; border: none; color: #9d9d9d; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; display: flex; align-items: center; gap: 4px; }}
.btn:hover {{ background: #37373d; color: #fff; }}
.run-btn {{ background: #4caf50 !important; color: white !important; }}
.run-btn:hover {{ background: #45a049 !important; }}
.main {{ flex: 1; display: flex; overflow: hidden; }}
.sidebar {{ width: 260px; background: #252526; border-right: 1px solid #3c3c3c; display: flex; flex-direction: column; }}
.sidebar-header {{ padding: 10px 12px; border-bottom: 1px solid #3c3c3c; font-size: 11px; text-transform: uppercase; color: #9d9d9d; display: flex; justify-content: space-between; align-items: center; }}
.file-tree {{ flex: 1; overflow: auto; padding: 4px 0; }}
.file-item {{ padding: 5px 12px; cursor: pointer; font-size: 13px; display: flex; align-items: center; gap: 6px; }}
.file-item:hover {{ background: #37373d; }}
.file-item.active {{ background: #094771; }}
.editor-area {{ flex: 1; display: flex; flex-direction: column; }}
.tabs {{ height: 35px; background: #2d2d2d; border-bottom: 1px solid #3c3c3c; display: flex; align-items: flex-end; }}
.tab {{ padding: 8px 16px; background: #252526; border-right: 1px solid #3c3c3c; cursor: pointer; font-size: 12px; color: #9d9d9d; }}
.tab.active {{ background: #1e1e1e; color: #fff; border-top: 2px solid #0078d4; }}
.editor-wrapper {{ flex: 1; position: relative; overflow: hidden; }}
.line-numbers {{ position: absolute; left: 0; top: 0; bottom: 0; width: 50px; background: #1e1e1e; border-right: 1px solid #3c3c3c; padding: 16px 8px; font-family: Consolas, monospace; font-size: 14px; line-height: 1.5; color: #858585; text-align: right; overflow: hidden; user-select: none; }}
.code-editor {{ position: absolute; left: 50px; top: 0; right: 0; bottom: 0; }}
.code-editor textarea {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: transparent; color: transparent; caret-color: #fff; border: none; outline: none; resize: none; font-family: Consolas, monospace; font-size: 14px; padding: 16px; line-height: 1.5; z-index: 2; white-space: pre; overflow: auto; }}
.code-highlight {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #1e1e1e; font-family: Consolas, monospace; font-size: 14px; padding: 16px; line-height: 1.5; white-space: pre; overflow: auto; pointer-events: none; z-index: 1; }}
.terminal {{ height: 180px; background: #1e1e1e; border-top: 1px solid #3c3c3c; }}
.terminal-header {{ padding: 8px 12px; border-bottom: 1px solid #3c3c3c; font-size: 12px; background: #2d2d2d; display: flex; justify-content: space-between; }}
.terminal-content {{ padding: 12px; font-family: Consolas, monospace; font-size: 13px; overflow: auto; height: calc(100% - 35px); white-space: pre-wrap; color: #ccc; }}
.statusbar {{ height: 24px; background: #0078d4; color: white; display: flex; align-items: center; justify-content: space-between; padding: 0 12px; font-size: 12px; }}
.error {{ color: #f48771; }}
.success {{ color: #89d185; }}
/* 语法高亮颜色 - VS Code风格 */
.hl-keyword {{ color: #c586c0; font-weight: 500; }}
.hl-control {{ color: #c586c0; }}
.hl-builtin {{ color: #4ec9b0; }}
.hl-string {{ color: #ce9178; }}
.hl-docstring {{ color: #6a9955; }}
.hl-comment {{ color: #6a9955; font-style: italic; }}
.hl-number {{ color: #b5cea8; }}
.hl-function {{ color: #dcdcaa; }}
.hl-funcdef {{ color: #dcdcaa; }}
.hl-class {{ color: #4ec9b0; font-weight: 500; }}
.hl-classdef {{ color: #4ec9b0; font-weight: 500; }}
.hl-decorator {{ color: #d7ba7d; }}
.hl-self {{ color: #9cdcfe; font-style: italic; }}
.hl-param {{ color: #9cdcfe; }}
.hl-operator {{ color: #d4d4d4; }}
.hl-import {{ color: #c586c0; }}
.hl-module {{ color: #4ec9b0; }}
.hl-const {{ color: #4fc1ff; }}
.hl-variable {{ color: #9cdcfe; }}
.hl-error {{ text-decoration: wavy underline #f44336; }}
.hl-warning {{ text-decoration: wavy underline #ff9800; }}
/* 自动补全 */
.autocomplete {{ position: absolute; background: #252526; border: 1px solid #3c3c3c; border-radius: 4px; max-height: 200px; overflow: auto; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.4); display: none; }}
.autocomplete-item {{ padding: 6px 12px; cursor: pointer; font-size: 13px; display: flex; align-items: center; gap: 8px; }}
.autocomplete-item:hover, .autocomplete-item.selected {{ background: #094771; }}
.autocomplete-item .kind {{ color: #0078d4; font-size: 11px; }}
.autocomplete-item .label {{ color: #fff; }}
.autocomplete-item .detail {{ color: #888; font-size: 11px; margin-left: auto; }}
/* 右键菜单 */
.context-menu {{ position: fixed; background: #252526; border: 1px solid #3c3c3c; border-radius: 4px; padding: 4px 0; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: none; min-width: 150px; }}
.context-menu-item {{ padding: 6px 16px; cursor: pointer; font-size: 13px; color: #ccc; }}
.context-menu-item:hover {{ background: #094771; }}
.context-menu-divider {{ height: 1px; background: #3c3c3c; margin: 4px 0; }}
/* 终端输入 */
.terminal-input-line {{ display: flex; align-items: center; padding: 4px 12px; background: #1a1a1a; border-top: 1px solid #3c3c3c; }}
.terminal-prompt {{ color: #4ec9b0; margin-right: 8px; font-family: Consolas, monospace; font-size: 13px; }}
.terminal-input {{ flex: 1; background: transparent; border: none; outline: none; color: #ccc; font-family: Consolas, monospace; font-size: 13px; }}
</style>
</head>
<body>
<div class="toolbar">
    <span class="logo">QM</span>
    <span class="title">QuantiaMagica IDE</span>
    <div style="flex:1"></div>
    <button class="btn" onclick="newFile()">📄 新建</button>
    <button class="btn" onclick="saveFile()">💾 保存</button>
    <button class="btn run-btn" onclick="runCode()">▶ 运行</button>
    <button class="btn" onclick="qtOpenFolder()">📁 打开文件夹</button>
</div>
<div class="main">
    <div class="sidebar">
        <div class="sidebar-header">
            <span>资源管理器</span>
            <button class="btn" onclick="loadFiles()" style="padding:2px 6px;">🔄</button>
        </div>
        <div class="file-tree" id="file-tree">
            <div style="padding:20px;color:#6d6d6d;font-size:12px;text-align:center;">点击"打开文件夹"开始</div>
        </div>
    </div>
    <div class="editor-area">
        <div class="tabs" id="tabs"><div class="tab active" id="current-tab">untitled.py</div></div>
        <div class="editor-wrapper">
            <div class="line-numbers" id="line-numbers">1</div>
            <div class="code-editor">
                <div class="code-highlight" id="highlight"></div>
                <textarea id="editor" spellcheck="false"></textarea>
            </div>
            <div class="autocomplete" id="autocomplete"></div>
        </div>
        <div class="terminal">
            <div class="terminal-header">
                <span>终端</span>
                <button class="btn" onclick="clearTerminal()" style="padding:2px 6px;">清空</button>
            </div>
            <div class="terminal-content" id="terminal-output"></div>
            <div class="terminal-input-line">
                <span class="terminal-prompt">$</span>
                <input type="text" class="terminal-input" id="terminal-input" placeholder="输入命令..." onkeydown="handleTerminalKey(event)">
            </div>
        </div>
    </div>
</div>
<!-- 右键菜单 -->
<div class="context-menu" id="context-menu">
    <div class="context-menu-item" onclick="contextAction('open')">📄 打开</div>
    <div class="context-menu-item" onclick="contextAction('rename')">✏️ 重命名</div>
    <div class="context-menu-divider"></div>
    <div class="context-menu-item" onclick="contextAction('delete')" style="color:#f48771;">🗑️ 删除</div>
</div>
<div class="statusbar">
    <span id="status-msg">就绪</span>
    <span><span id="cursor-pos">行 1, 列 1</span> | UTF-8 | Python</span>
</div>

<script>
const API = 'http://127.0.0.1:{port}';
let workspace = null;
let currentFile = null;
let contextTarget = null;  // 右键菜单目标文件
let terminalHistory = [];
let historyIndex = -1;

// Python关键字和内置函数
const KEYWORDS = ['False','None','True','and','as','assert','async','await','break','class','continue','def','del','elif','else','except','finally','for','from','global','if','import','in','is','lambda','nonlocal','not','or','pass','raise','return','try','while','with','yield'];
const BUILTINS = ['print','len','range','str','int','float','list','dict','set','tuple','open','input','type','isinstance','hasattr','getattr','setattr','abs','all','any','bin','bool','bytes','callable','chr','dir','divmod','enumerate','eval','exec','filter','format','frozenset','globals','hash','help','hex','id','iter','locals','map','max','min','next','object','oct','ord','pow','repr','reversed','round','slice','sorted','staticmethod','sum','super','vars','zip'];
const QM_CLASSES = ['SARADC','PipelineADC','SigmaDeltaADC','Signal','GeneticOptimizer','Gene','SamplingEvent','ComparatorEvent','QuantizerEvent'];
const QM_METHODS = ['sim','enob','snr','sfdr','thd','inl','dnl','plot','spectrum','report','on','run'];

// Qt桥接 - 打开文件夹
function qtOpenFolder() {{
    // 通过API请求Python打开文件对话框
    fetch(API + '/api/browse', {{method: 'POST'}}).then(r => r.json()).then(d => {{
        if (d.path) {{ workspace = d.path; loadFiles(); }}
    }});
}}

// 被Python调用设置工作目录
function setWorkspaceFromQt(path) {{
    workspace = path;
    loadFiles();
    log('已打开: ' + path, 'success');
}}

async function loadFiles() {{
    const res = await fetch(API + '/api/files');
    const data = await res.json();
    const tree = document.getElementById('file-tree');
    workspace = data.workspace;
    if (data.files && data.files.length > 0) {{
        tree.innerHTML = renderFiles(data.files);
    }} else {{
        tree.innerHTML = '<div style="padding:20px;color:#6d6d6d;text-align:center;">' + (workspace ? '空文件夹' : '未打开文件夹') + '</div>';
    }}
}}

function renderFiles(files, indent = 0) {{
    return files.map(f => {{
        const style = 'padding-left:' + (12 + indent * 16) + 'px';
        const path = f.path.replace(/\\\\/g, '/');
        if (f.type === 'directory') {{
            return '<div class="file-item" style="' + style + '" data-path="' + path + '" oncontextmenu="showContextMenu(event, \\'' + path + '\\', true)">📁 ' + f.name + '</div>' + (f.children ? renderFiles(f.children, indent + 1) : '');
        }} else {{
            const icon = f.name.endsWith('.py') ? '🐍' : '📄';
            return '<div class="file-item" style="' + style + '" data-path="' + path + '" onclick="openFile(\\'' + path + '\\')" oncontextmenu="showContextMenu(event, \\'' + path + '\\', false)">' + icon + ' ' + f.name + '</div>';
        }}
    }}).join('');
}}

async function openFile(path) {{
    const res = await fetch(API + '/api/file?path=' + encodeURIComponent(path));
    const data = await res.json();
    if (data.content !== undefined) {{
        document.getElementById('editor').value = data.content;
        document.getElementById('current-tab').textContent = path.split(/[\\\\/]/).pop();
        // 保存完整路径（workspace + 相对路径）
        currentFile = workspace ? (workspace + '/' + path).replace(/\\\\/g, '/') : path;
        updateHighlight();
        updateLineNumbers();
    }}
}}

async function saveFile() {{
    const content = document.getElementById('editor').value;
    const name = currentFile || document.getElementById('current-tab').textContent;
    if (!workspace) {{ log('请先打开文件夹', 'error'); return; }}
    const res = await fetch(API + '/api/save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{path: name, content}})
    }});
    const data = await res.json();
    if (data.success) {{ log('已保存: ' + name, 'success'); }} 
    else {{ log('保存失败: ' + data.error, 'error'); }}
}}

async function runCode() {{
    const code = document.getElementById('editor').value;
    if (!code.trim()) {{ log('代码为空', 'error'); return; }}
    
    document.getElementById('status-msg').textContent = '运行中...';
    document.getElementById('status-msg').style.color = '#fff';
    log('>>> 运行代码...', '');
    try {{
        const res = await fetch(API + '/api/run', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{code, file: currentFile || ''}})
        }});
        const data = await res.json();
        if (data.output) log(data.output, '');
        if (data.error) log(data.error, 'error');
        log('>>> ' + (data.returncode === 0 ? '运行成功' : '运行失败'), data.returncode === 0 ? 'success' : 'error');
    }} catch (e) {{ 
        log('错误: ' + e.message, 'error'); 
    }}
    document.getElementById('status-msg').textContent = '就绪';
    runLint();
}}

function newFile() {{
    document.getElementById('editor').value = '';
    document.getElementById('current-tab').textContent = 'untitled.py';
    currentFile = null;
    updateHighlight();
    updateLineNumbers();
}}

function log(msg, type) {{
    const out = document.getElementById('terminal-output');
    const div = document.createElement('div');
    div.className = type;
    div.textContent = msg;
    out.appendChild(div);
    out.scrollTop = out.scrollHeight;
}}

function clearTerminal() {{ document.getElementById('terminal-output').innerHTML = ''; }}

// 终端命令处理
async function handleTerminalKey(e) {{
    if (e.key === 'Enter') {{
        const input = document.getElementById('terminal-input');
        const cmd = input.value.trim();
        if (!cmd) return;
        
        // 添加到历史
        terminalHistory.push(cmd);
        historyIndex = terminalHistory.length;
        
        // 显示命令
        log('$ ' + cmd, '');
        input.value = '';
        
        // 执行命令
        try {{
            const res = await fetch(API + '/api/terminal', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{command: cmd}})
            }});
            const data = await res.json();
            if (data.output) log(data.output, data.returncode === 0 ? '' : 'error');
        }} catch (e) {{
            log('错误: ' + e.message, 'error');
        }}
    }} else if (e.key === 'ArrowUp') {{
        e.preventDefault();
        if (historyIndex > 0) {{
            historyIndex--;
            document.getElementById('terminal-input').value = terminalHistory[historyIndex] || '';
        }}
    }} else if (e.key === 'ArrowDown') {{
        e.preventDefault();
        if (historyIndex < terminalHistory.length - 1) {{
            historyIndex++;
            document.getElementById('terminal-input').value = terminalHistory[historyIndex] || '';
        }} else {{
            historyIndex = terminalHistory.length;
            document.getElementById('terminal-input').value = '';
        }}
    }}
}}

// 右键菜单
function showContextMenu(e, path, isDir) {{
    e.preventDefault();
    contextTarget = {{path, isDir}};
    const menu = document.getElementById('context-menu');
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.style.display = 'block';
    // 隐藏"打开"选项如果是目录
    menu.children[0].style.display = isDir ? 'none' : 'block';
}}

function hideContextMenu() {{
    document.getElementById('context-menu').style.display = 'none';
}}

async function contextAction(action) {{
    hideContextMenu();
    if (!contextTarget) return;
    
    const path = contextTarget.path;
    
    if (action === 'open') {{
        openFile(path);
    }} else if (action === 'rename') {{
        const oldName = path.split('/').pop();
        const newName = prompt('输入新名称:', oldName);
        if (newName && newName !== oldName) {{
            const res = await fetch(API + '/api/rename', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{oldPath: path, newName}})
            }});
            const data = await res.json();
            if (data.success) {{
                log('已重命名: ' + oldName + ' -> ' + newName, 'success');
                loadFiles();
            }} else {{
                log('重命名失败: ' + data.error, 'error');
            }}
        }}
    }} else if (action === 'delete') {{
        if (confirm('确定删除 ' + path + ' ?')) {{
            const res = await fetch(API + '/api/delete', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{path}})
            }});
            const data = await res.json();
            if (data.success) {{
                log('已删除: ' + path, 'success');
                loadFiles();
                // 如果删除的是当前打开的文件
                if (currentFile && currentFile.endsWith(path)) {{
                    newFile();
                }}
            }} else {{
                log('删除失败: ' + data.error, 'error');
            }}
        }}
    }}
}}

// 点击其他地方隐藏菜单
document.addEventListener('click', hideContextMenu);

// 语法高亮 - 改进版
function highlightCode(code) {{
    // 先转义HTML
    let escaped = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // 用占位符保护字符串和注释，避免被其他规则破坏
    const tokens = [];
    let tokenId = 0;
    
    // 1. 三引号字符串(docstring) - 必须先处理
    escaped = escaped.replace(/(\'\'\'[\\s\\S]*?\'\'\'|\"\"\"[\\s\\S]*?\"\"\")/g, (m) => {{
        tokens[tokenId] = '<span class="hl-docstring">' + m + '</span>';
        return '___TOKEN' + (tokenId++) + '___';
    }});
    
    // 2. 单行注释
    escaped = escaped.replace(/(#.*?)$/gm, (m) => {{
        tokens[tokenId] = '<span class="hl-comment">' + m + '</span>';
        return '___TOKEN' + (tokenId++) + '___';
    }});
    
    // 3. 普通字符串
    escaped = escaped.replace(/(["'])(?:(?!\\1)[^\\\\\\n]|\\\\.)*?\\1/g, (m) => {{
        tokens[tokenId] = '<span class="hl-string">' + m + '</span>';
        return '___TOKEN' + (tokenId++) + '___';
    }});
    
    // 4. 装饰器
    escaped = escaped.replace(/(@\\w+)/g, '<span class="hl-decorator">$1</span>');
    
    // 5. 数字
    escaped = escaped.replace(/\\b(\\d+\\.?\\d*(?:e[+-]?\\d+)?)\\b/gi, '<span class="hl-number">$1</span>');
    
    // 6. def/class定义
    escaped = escaped.replace(/\\b(def)\\s+(\\w+)/g, '<span class="hl-keyword">$1</span> <span class="hl-funcdef">$2</span>');
    escaped = escaped.replace(/\\b(class)\\s+(\\w+)/g, '<span class="hl-keyword">$1</span> <span class="hl-classdef">$2</span>');
    
    // 7. import语句
    escaped = escaped.replace(/\\b(from)\\s+(\\S+)\\s+(import)/g, '<span class="hl-import">$1</span> <span class="hl-module">$2</span> <span class="hl-import">$3</span>');
    escaped = escaped.replace(/\\b(import)\\s+(\\S+)/g, '<span class="hl-import">$1</span> <span class="hl-module">$2</span>');
    
    // 8. self参数
    escaped = escaped.replace(/\\b(self)\\b/g, '<span class="hl-self">$1</span>');
    
    // 9. 关键字（控制流）
    const CONTROL = ['if','elif','else','for','while','try','except','finally','with','return','yield','break','continue','pass','raise','assert'];
    CONTROL.forEach(kw => {{
        escaped = escaped.replace(new RegExp('\\\\b(' + kw + ')\\\\b', 'g'), '<span class="hl-control">$1</span>');
    }});
    
    // 10. 其他关键字
    const OTHER_KW = ['and','or','not','in','is','None','True','False','lambda','global','nonlocal','del','async','await'];
    OTHER_KW.forEach(kw => {{
        escaped = escaped.replace(new RegExp('\\\\b(' + kw + ')\\\\b', 'g'), '<span class="hl-keyword">$1</span>');
    }});
    
    // 11. 内置函数
    BUILTINS.forEach(fn => {{
        escaped = escaped.replace(new RegExp('\\\\b(' + fn + ')\\\\s*\\\\(', 'g'), '<span class="hl-builtin">$1</span>(');
    }});
    
    // 12. QM类
    QM_CLASSES.forEach(c => {{
        escaped = escaped.replace(new RegExp('\\\\b(' + c + ')\\\\b', 'g'), '<span class="hl-class">$1</span>');
    }});
    
    // 13. 方法调用
    escaped = escaped.replace(/\\.(\\w+)\\s*\\(/g, '.<span class="hl-function">$1</span>(');
    
    // 14. 常量（全大写）
    escaped = escaped.replace(/\\b([A-Z][A-Z0-9_]+)\\b/g, '<span class="hl-const">$1</span>');
    
    // 15. 变量赋值 (name = value 中的name)
    escaped = escaped.replace(/^(\\s*)(\\w+)(\\s*=\\s*[^=])/gm, '$1<span class="hl-variable">$2</span>$3');
    
    // 16. 函数参数
    escaped = escaped.replace(/\\((\\w+)=/g, '(<span class="hl-param">$1</span>=');
    
    // 恢复保护的token
    for (let i = 0; i < tokenId; i++) {{
        escaped = escaped.replace('___TOKEN' + i + '___', tokens[i]);
    }}
    
    return escaped;
}}

function updateHighlight() {{
    const code = document.getElementById('editor').value;
    document.getElementById('highlight').innerHTML = highlightCode(code) + '\\n';
}}

function updateLineNumbers() {{
    const lines = document.getElementById('editor').value.split('\\n').length;
    document.getElementById('line-numbers').innerHTML = Array.from({{length: lines}}, (_, i) => i + 1).join('<br>');
}}

function updateCursorPos() {{
    const ta = document.getElementById('editor');
    const pos = ta.selectionStart;
    const lines = ta.value.substring(0, pos).split('\\n');
    document.getElementById('cursor-pos').textContent = '行 ' + lines.length + ', 列 ' + (lines[lines.length - 1].length + 1);
}}

// 自动补全 - IDEA风格实时补全
const completions = [
    // QuantiaMagica类
    ...QM_CLASSES.map(c => ({{label: c, kind: '类', detail: 'QuantiaMagica', insert: c}})),
    // 方法
    {{label: 'sim', kind: '方法', detail: '运行ADC仿真', insert: 'sim(n_samples=4096, fs=1e6)'}},
    {{label: 'enob', kind: '方法', detail: '计算有效位数', insert: 'enob()'}},
    {{label: 'snr', kind: '方法', detail: '计算信噪比', insert: 'snr()'}},
    {{label: 'sfdr', kind: '方法', detail: '无杂散动态范围', insert: 'sfdr()'}},
    {{label: 'plot', kind: '方法', detail: '绘图', insert: 'plot()'}},
    {{label: 'report', kind: '方法', detail: '生成报告', insert: "report('all')"}},
    // 代码片段
    {{label: 'from quantiamagica import', kind: '导入', detail: '导入模块', insert: 'from quantiamagica import '}},
    {{label: 'import numpy as np', kind: '导入', detail: 'NumPy', insert: 'import numpy as np'}},
    {{label: 'import matplotlib.pyplot as plt', kind: '导入', detail: 'Matplotlib', insert: 'import matplotlib.pyplot as plt'}},
    // 常用内置
    {{label: 'print', kind: '函数', detail: '打印输出', insert: 'print()'}},
    {{label: 'range', kind: '函数', detail: '生成范围', insert: 'range()'}},
    {{label: 'len', kind: '函数', detail: '获取长度', insert: 'len()'}},
    {{label: 'def', kind: '关键字', detail: '定义函数', insert: 'def ():\\n    '}},
    {{label: 'class', kind: '关键字', detail: '定义类', insert: 'class :\\n    '}},
    {{label: 'for', kind: '关键字', detail: 'for循环', insert: 'for  in :\\n    '}},
    {{label: 'if', kind: '关键字', detail: '条件判断', insert: 'if :\\n    '}},
    {{label: 'while', kind: '关键字', detail: 'while循环', insert: 'while :\\n    '}},
    {{label: 'try', kind: '关键字', detail: '异常处理', insert: 'try:\\n    \\nexcept Exception as e:\\n    '}},
    {{label: 'with', kind: '关键字', detail: '上下文管理', insert: 'with  as :\\n    '}},
    {{label: 'return', kind: '关键字', detail: '返回值', insert: 'return '}},
    // numpy常用
    {{label: 'np.array', kind: 'numpy', detail: '创建数组', insert: 'np.array([])'}},
    {{label: 'np.zeros', kind: 'numpy', detail: '零数组', insert: 'np.zeros()'}},
    {{label: 'np.ones', kind: 'numpy', detail: '全1数组', insert: 'np.ones()'}},
    {{label: 'np.arange', kind: 'numpy', detail: '范围数组', insert: 'np.arange()'}},
    {{label: 'np.linspace', kind: 'numpy', detail: '线性空间', insert: 'np.linspace(0, 1, 100)'}},
    {{label: 'np.sin', kind: 'numpy', detail: '正弦函数', insert: 'np.sin()'}},
    {{label: 'np.pi', kind: 'numpy', detail: '圆周率', insert: 'np.pi'}},
];

let acIndex = 0;
function showAutocomplete(word, x, y) {{
    const ac = document.getElementById('autocomplete');
    // 匹配：前缀匹配或包含匹配
    const matches = completions.filter(c => 
        c.label.toLowerCase().startsWith(word.toLowerCase()) ||
        c.label.toLowerCase().includes(word.toLowerCase())
    ).slice(0, 10);  // 最多显示10个
    
    if (matches.length === 0 || word.length < 1) {{ ac.style.display = 'none'; return; }}
    acIndex = 0;
    ac.innerHTML = matches.map((m, i) => 
        '<div class="autocomplete-item' + (i === 0 ? ' selected' : '') + '" data-insert="' + m.insert.replace(/"/g, '&quot;') + '">' +
        '<span class="kind">' + m.kind + '</span><span class="label">' + m.label + '</span><span class="detail">' + m.detail + '</span></div>'
    ).join('');
    ac.style.left = Math.min(x, window.innerWidth - 320) + 'px';
    ac.style.top = y + 'px';
    ac.style.display = 'block';
}}

function hideAutocomplete() {{ document.getElementById('autocomplete').style.display = 'none'; }}

function insertCompletion(insert) {{
    const ta = document.getElementById('editor');
    const pos = ta.selectionStart;
    const text = ta.value;
    const before = text.substring(0, pos);
    const wordStart = before.search(/[\\w.]*$/);
    ta.value = text.substring(0, wordStart) + insert + text.substring(pos);
    ta.selectionStart = ta.selectionEnd = wordStart + insert.length;
    hideAutocomplete();
    updateHighlight();
    updateLineNumbers();
    runLint();
}}

// 简单Lint检查
function runLint() {{
    const code = document.getElementById('editor').value;
    const lines = code.split('\\n');
    const errors = [];
    
    lines.forEach((line, i) => {{
        const lineNum = i + 1;
        const trimmed = line.trim();
        
        // 检查未闭合的括号
        const opens = (line.match(/[\\(\\[\\{{]/g) || []).length;
        const closes = (line.match(/[\\)\\]\\}}]/g) || []).length;
        if (opens > closes && !trimmed.endsWith(':') && !trimmed.endsWith(',') && !trimmed.endsWith('\\\\')) {{
            // 可能有问题，但不一定是错误
        }}
        
        // 检查缩进问题
        if (line.length > 0 && !line.startsWith(' ') && !line.startsWith('\\t') && !line.startsWith('#')) {{
            const indent = line.search(/\\S/);
            if (indent > 0 && indent % 4 !== 0) {{
                errors.push({{line: lineNum, msg: '缩进应为4的倍数', type: 'warning'}});
            }}
        }}
        
        // 检查常见错误
        if (/print\\s+[^(]/.test(line) && !/print\\s*=/.test(line)) {{
            errors.push({{line: lineNum, msg: 'print需要括号: print()', type: 'error'}});
        }}
        
        // 检查 = vs == 
        if (/if\\s+.*[^=!<>]=[^=]/.test(line)) {{
            errors.push({{line: lineNum, msg: '条件判断应使用 == 而非 =', type: 'warning'}});
        }}
    }});
    
    // 显示lint结果
    const statusMsg = document.getElementById('status-msg');
    if (errors.length > 0) {{
        statusMsg.textContent = '⚠ ' + errors.length + '个问题';
        statusMsg.style.color = '#ff9800';
    }} else {{
        statusMsg.textContent = '✓ 无问题';
        statusMsg.style.color = '#89d185';
    }}
    
    return errors;
}}

// 编辑器事件
const editor = document.getElementById('editor');
const highlight = document.getElementById('highlight');

editor.addEventListener('input', function() {{
    updateHighlight();
    updateLineNumbers();
    runLint();
    
    // 自动补全触发 - 只需1个字符就开始
    const pos = this.selectionStart;
    const before = this.value.substring(0, pos);
    const word = before.match(/[\\w.]*$/)[0];
    if (word.length >= 1) {{
        const lines = before.split('\\n');
        const lineHeight = 21;
        const y = Math.min((lines.length) * lineHeight + 50, window.innerHeight - 250);
        const x = Math.min(lines[lines.length - 1].length * 8.4 + 70, window.innerWidth - 320);
        showAutocomplete(word, x, y);
    }} else {{ hideAutocomplete(); }}
}});

editor.addEventListener('scroll', function() {{
    highlight.scrollTop = this.scrollTop;
    highlight.scrollLeft = this.scrollLeft;
    document.getElementById('line-numbers').scrollTop = this.scrollTop;
}});

editor.addEventListener('keydown', function(e) {{
    const ac = document.getElementById('autocomplete');
    if (ac.style.display === 'block') {{
        const items = ac.querySelectorAll('.autocomplete-item');
        if (e.key === 'ArrowDown') {{ e.preventDefault(); acIndex = Math.min(acIndex + 1, items.length - 1); items.forEach((it, i) => it.classList.toggle('selected', i === acIndex)); }}
        else if (e.key === 'ArrowUp') {{ e.preventDefault(); acIndex = Math.max(acIndex - 1, 0); items.forEach((it, i) => it.classList.toggle('selected', i === acIndex)); }}
        else if (e.key === 'Enter' || e.key === 'Tab') {{ e.preventDefault(); if (items[acIndex]) insertCompletion(items[acIndex].dataset.insert); }}
        else if (e.key === 'Escape') {{ hideAutocomplete(); }}
        return;
    }}
    if (e.key === 'Tab') {{
        e.preventDefault();
        const s = this.selectionStart;
        this.value = this.value.substring(0, s) + '    ' + this.value.substring(this.selectionEnd);
        this.selectionStart = this.selectionEnd = s + 4;
        updateHighlight();
    }}
    if (e.ctrlKey && e.key === 's') {{ e.preventDefault(); saveFile(); }}
    if (e.key === 'F5') {{ e.preventDefault(); runCode(); }}
}});

editor.addEventListener('click', function() {{ updateCursorPos(); hideAutocomplete(); }});
editor.addEventListener('keyup', updateCursorPos);

document.getElementById('autocomplete').addEventListener('click', function(e) {{
    const item = e.target.closest('.autocomplete-item');
    if (item) insertCompletion(item.dataset.insert);
}});

// 初始化
updateHighlight();
updateLineNumbers();
loadFiles();
console.log('IDE loaded');
</script>
</body>
</html>'''
        window.browser.setHtml(html, QUrl(f"http://127.0.0.1:{port}/"))
        print("HTML已加载")
    
    QTimer.singleShot(500, delayed_load)
    
    # 定时检查是否需要打开文件对话框
    def check_browse_request():
        global _browse_requested
        if _browse_requested:
            _browse_requested = False
            window.open_folder_dialog()
    
    browse_timer = QTimer()
    browse_timer.timeout.connect(check_browse_request)
    browse_timer.start(100)  # 每100ms检查一次
    
    print("IDE窗口已显示")
    sys.exit(app.exec())
