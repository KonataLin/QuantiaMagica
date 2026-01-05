# QuantiaMagica IDE 开发与维护文档

> VS Code风格的ADC仿真开发环境，使用PyQt6 WebEngineView内嵌HTML实现

**最后更新**: 2026-01-05

## 目录结构

```
quantiamagica/ide/
├── __init__.py          # 模块入口，导出launch函数
├── server.py            # 主程序（API服务器 + Qt桌面应用 + 内嵌HTML/CSS/JS）
└── static/              # 静态资源（备用，当前使用内嵌方式）
    ├── index.html
    ├── style.css
    └── app.js
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    PyQt6 QMainWindow                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              QWebEngineView (内嵌浏览器)                │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │                HTML/CSS/JS IDE界面               │  │  │
│  │  │  ┌─────────┐ ┌──────────────────────────────┐  │  │  │
│  │  │  │ 侧边栏   │ │         编辑器区域            │  │  │  │
│  │  │  │ 文件树   │ │  ┌────────────────────────┐ │  │  │  │
│  │  │  │         │ │  │   语法高亮代码编辑器    │ │  │  │  │
│  │  │  │         │ │  └────────────────────────┘ │  │  │  │
│  │  │  │         │ │  ┌────────────────────────┐ │  │  │  │
│  │  │  │         │ │  │      输出终端           │ │  │  │  │
│  │  │  │         │ │  └────────────────────────┘ │  │  │  │
│  │  │  └─────────┘ └──────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP API (127.0.0.1:8765)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python 后端服务器                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ 文件管理API  │ │ 代码运行API │ │ 补全/Lint API       │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 后端API服务器 (`APIHandler`)

基于Python `http.server` 实现的轻量级API服务器。

**API端点：**

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 返回静态HTML |
| `/api/files` | GET | 获取工作目录文件列表 |
| `/api/file?path=xxx` | GET | 读取文件内容 |
| `/api/save` | POST | 保存文件 |
| `/api/workspace` | POST | 设置工作目录 |
| `/api/browse` | POST | 触发Qt文件对话框 |
| `/api/run` | POST | 运行Python代码 |
| `/api/delete` | POST | 删除文件/文件夹 |
| `/api/rename` | POST | 重命名文件/文件夹 |
| `/api/terminal` | POST | 执行终端命令 |
| `/api/completions` | GET | 获取补全信息 |

### 2. Qt桌面应用 (`launch`函数)

使用PyQt6创建桌面窗口，内嵌WebEngineView显示HTML界面。

**关键特性：**
- `QWebEngineView`: 内嵌Chromium浏览器引擎
- `QWebChannel`: JS与Python双向通信
- `QFileDialog`: 原生文件对话框
- `QTimer`: 定时检查浏览请求

### 3. 前端界面（内嵌HTML/CSS/JS）

完全自包含的HTML字符串，无需外部CDN依赖。

**功能模块：**
- **语法高亮**: 基于正则表达式的Python语法着色
- **自动补全**: 实时补全提示，支持QuantiaMagica API
- **Lint检查**: 简单的语法错误检测
- **文件树**: 递归显示目录结构，支持右键菜单
- **交互式终端**: 底部可输入shell命令
- **文件管理**: 删除、重命名文件/文件夹

## 语法高亮规则

处理顺序（重要，避免冲突）：

1. **三引号字符串(docstring)** - 最先处理，保护内容
2. **单行注释(#)** - 保护内容
3. **普通字符串** - 保护内容
4. **装饰器(@)** - 金色
5. **数字** - 浅绿色
6. **def/class定义** - 关键字紫色 + 名称黄色/青色
7. **import语句** - 紫色 + 模块名青色
8. **self** - 浅蓝斜体
9. **控制流关键字** - 紫色
10. **其他关键字** - 紫色
11. **内置函数** - 青色
12. **QuantiaMagica类** - 青色
13. **方法调用** - 黄色
14. **常量(全大写)** - 亮蓝色
15. **变量赋值** - 浅蓝色
16. **函数参数** - 浅蓝色

**颜色方案（VS Code Dark+）：**

```css
.hl-keyword    { color: #c586c0; }  /* 紫色 - 关键字 */
.hl-control    { color: #c586c0; }  /* 紫色 - 控制流 */
.hl-builtin    { color: #4ec9b0; }  /* 青色 - 内置函数 */
.hl-string     { color: #ce9178; }  /* 橙色 - 字符串 */
.hl-docstring  { color: #6a9955; }  /* 绿色 - docstring */
.hl-comment    { color: #6a9955; }  /* 绿色 - 注释 */
.hl-number     { color: #b5cea8; }  /* 浅绿 - 数字 */
.hl-function   { color: #dcdcaa; }  /* 黄色 - 函数 */
.hl-class      { color: #4ec9b0; }  /* 青色 - 类 */
.hl-decorator  { color: #d7ba7d; }  /* 金色 - 装饰器 */
.hl-self       { color: #9cdcfe; }  /* 浅蓝 - self */
.hl-variable   { color: #9cdcfe; }  /* 浅蓝 - 变量 */
.hl-param      { color: #9cdcfe; }  /* 浅蓝 - 参数 */
.hl-const      { color: #4fc1ff; }  /* 亮蓝 - 常量 */
.hl-import     { color: #c586c0; }  /* 紫色 - import */
.hl-module     { color: #4ec9b0; }  /* 青色 - 模块名 */
```

## 自动补全

**触发条件：** 输入1个字符即开始

**补全项来源：**
- QuantiaMagica类: `SARADC`, `PipelineADC`, `SigmaDeltaADC`等
- ADC方法: `sim()`, `enob()`, `snr()`, `sfdr()`, `plot()`等
- Python内置: `print`, `range`, `len`, `def`, `class`等
- NumPy常用: `np.array`, `np.zeros`, `np.linspace`等
- 代码片段: import语句、循环结构等

**键盘操作：**
- `↑/↓`: 选择补全项
- `Tab/Enter`: 确认补全
- `Esc`: 关闭补全菜单

## Lint检查

**检查规则：**
1. `print`语法错误（Python 3需要括号）
2. 条件判断中 `=` 与 `==` 混淆
3. 缩进不是4的倍数（警告）

**显示位置：** 状态栏左侧

## 代码运行

**执行方式：**
1. 将代码写入临时文件（支持`__file__`）
2. 使用`subprocess`调用Python解释器
3. 设置`PYTHONPATH`包含项目根目录
4. 捕获stdout/stderr输出
5. 60秒超时保护
6. 运行后删除临时文件

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+S` | 保存文件 |
| `F5` | 运行代码 |
| `Tab` | 缩进4空格 / 确认补全 |
| `Esc` | 关闭补全菜单 |
| `Enter`(终端) | 执行命令 |
| `↑/↓`(终端) | 浏览历史命令 |
| `右键`(文件树) | 显示上下文菜单 |

## 依赖

```
PyQt6>=6.4.0
PyQt6-WebEngine>=6.4.0
```

安装：
```bash
pip install PyQt6 PyQt6-WebEngine
```

## 启动方式

```python
# 方式1: 命令行
python main.py

# 方式2: 代码调用
from quantiamagica.ide import launch
launch(port=8765, workspace="path/to/project")
```

## 维护指南

### 修改语法高亮

编辑 `server.py` 中的 `highlightCode` 函数：

```javascript
function highlightCode(code) {
    // 1. 先转义HTML字符
    // 2. 用token保护字符串和注释
    // 3. 按优先级应用正则替换
    // 4. 恢复保护的token
}
```

### 添加补全项

编辑 `server.py` 中的 `completions` 数组：

```javascript
const completions = [
    {label: '显示名', kind: '类型', detail: '描述', insert: '插入内容'},
    // ...
];
```

### 添加API端点

1. 在 `APIHandler` 类中添加处理方法
2. 在 `do_GET` 或 `do_POST` 中添加路由

### 修改样式

编辑 `server.py` 中 `delayed_load` 函数里的 `<style>` 部分。

## 已知限制

1. 语法高亮基于正则，复杂嵌套可能不准确
2. 自动补全是静态列表，无上下文感知
3. Lint检查较简单，不如专业工具
4. 不支持调试功能
5. 不支持多文件标签

## 交互式终端

底部终端支持输入shell命令：

```javascript
// 处理函数
async function handleTerminalKey(e) {
    if (e.key === 'Enter') {
        // 执行命令
        const res = await fetch(API + '/api/terminal', {
            method: 'POST',
            body: JSON.stringify({command: cmd})
        });
    }
}
```

**功能：**
- 支持任意shell命令 (`dir`, `python`, `pip`等)
- 命令历史记录（↑/↓浏览）
- 30秒超时保护

## 文件右键菜单

在文件树中右键点击显示上下文菜单：

| 选项 | 功能 |
|------|------|
| 打开 | 打开文件（仅文件） |
| 重命名 | 弹出输入框修改名称 |
| 删除 | 确认后删除文件/文件夹 |

---

## 版本更新日志

### v1.3.0 (2026-01-05)

**Pipeline完全重构：**
- ✅ 重构Pipeline为独立类（不继承ADConverter）
- ✅ 修复0.5 LSB舍入偏移，Pipeline ENOB = 23.99 bits（与24-bit SAR完全一致）
- ✅ 修复误差累积公式，正确计算增益误差/失调/噪声对ENOB的影响
  - 0.5mV失调 → 理论损失5.0 bits，仿真损失6.1 bits（吻合）
- ✅ 验证所有事件正确触发：StageEvent、ResidueEvent、InterstageGainEvent
- ✅ 修复IDE模块导入错误（IDEServer不存在）
- ✅ 更新docs/index.html Pipeline API文档

### v1.1.0 (2026-01-05)

**新功能：**
- ✅ 启动时自动kill占用端口的旧进程 (`kill_port`函数)
- ✅ 交互式终端，支持输入shell命令
- ✅ 文件右键菜单（删除、重命名）
- ✅ Qt原生文件对话框打开文件夹

**Bug修复：**
- 🐛 修复`__file__`未定义错误 - 改用临时文件/原文件运行
- 🐛 修复文件路径传递问题 - `currentFile`保存完整路径
- 🐛 修复旧进程占用端口导致新代码不生效
- 🐛 修复语法高亮docstring/注释不显示

**改进：**
- 改进自动补全，1个字符即触发
- 添加更多语法高亮颜色（变量、参数等）
- 添加简单Lint检查

### v1.0.0 (2026-01-04)

**初始版本：**
- PyQt6 WebEngineView内嵌HTML
- VS Code风格界面
- Python语法高亮
- 自动补全
- 代码运行

---

## 已知问题与解决方案

### 1. 白屏/界面不显示

**原因：** 旧进程占用端口，新代码未加载

**解决：**
```powershell
# 查找占用端口的进程
netstat -ano | findstr 8765
# 杀掉进程
taskkill /PID <进程号> /F
```

**注：** v1.1.0已自动处理此问题

### 2. `__file__`未定义错误

**原因：** 之前使用`python -c "code"`方式运行

**解决：** v1.1.0改为写入临时文件后运行

### 3. 打开文件后运行报错

**原因：** `currentFile`只保存相对路径

**解决：** v1.1.0修复，保存完整路径 `workspace + '/' + path`

---

## 未来改进

- [ ] 集成Monaco Editor CDN（需解决WebEngine加载问题）
- [ ] 添加上下文感知补全
- [ ] 集成Python AST进行更准确的语法分析
- [ ] 支持多标签编辑
- [ ] 添加断点调试功能
- [ ] 支持Git集成
- [ ] 文件拖拽支持
- [ ] 搜索/替换功能
