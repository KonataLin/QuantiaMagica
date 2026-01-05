"""
QuantiaMagica IDE - VS Code风格的ADC开发环境

专为ADC仿真设计的轻量级Python IDE，支持：
- Monaco编辑器（VS Code内核）
- 语法高亮和自动补全
- 文件/文件夹管理
- 一键运行Python代码
- quantiamagica API集成

Usage:
    from quantiamagica.ide import launch
    launch()  # 启动IDE
"""

from .server import launch

__all__ = ['launch']
