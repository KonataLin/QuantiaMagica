"""
Plotting Module - IEEE JSSC 风格绘图
====================================

提供符合 IEEE Journal of Solid-State Circuits (JSSC) 规范的专业绘图功能。

JSSC 规范要点:
- 字体: Times New Roman 或 Arial
- 字号: 8-10pt (figure), 标题稍大
- 线宽: 0.5-1pt
- 颜色: 黑白优先，彩色需可辨识
- 图像: 高分辨率 (300+ dpi)
- 尺寸: 单栏 3.5in, 双栏 7.16in

Example
-------
>>> from quantiamagica.plotting import jssc_style, plot_spectrum_jssc
>>> 
>>> with jssc_style():
...     adc.plot()
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
from contextlib import contextmanager
from typing import Optional, Tuple, Any, Dict, List
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# IEEE JSSC 样式配置
# =============================================================================

JSSC_STYLE = {
    # 字体设置
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    
    # 线条设置
    'lines.linewidth': 0.8,
    'lines.markersize': 4,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    
    # 网格
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    
    # 图例
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # 刻度
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    
    # 图形
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    
    # 颜色循环 (JSSC推荐的可区分颜色)
    'axes.prop_cycle': plt.cycler(color=[
        '#000000',  # 黑色 (主要)
        '#0072B2',  # 蓝色
        '#D55E00',  # 橙红色
        '#009E73',  # 绿色
        '#CC79A7',  # 粉色
        '#F0E442',  # 黄色
        '#56B4E9',  # 浅蓝
    ]),
    
    # 数学文本
    'mathtext.fontset': 'stix',
}

# 单栏和双栏尺寸 (英寸)
JSSC_SINGLE_COLUMN = (3.5, 2.625)  # 宽度 3.5in
JSSC_DOUBLE_COLUMN = (7.16, 4.0)   # 宽度 7.16in


@contextmanager
def jssc_style():
    """
    上下文管理器：临时应用 JSSC 样式。
    
    Example
    -------
    >>> with jssc_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot(x, y)
    ...     plt.savefig('figure.pdf')
    """
    old_params = {k: mpl.rcParams.get(k) for k in JSSC_STYLE.keys()}
    try:
        mpl.rcParams.update(JSSC_STYLE)
        yield
    finally:
        for k, v in old_params.items():
            if v is not None:
                mpl.rcParams[k] = v


def apply_jssc_style():
    """全局应用 JSSC 样式。"""
    mpl.rcParams.update(JSSC_STYLE)


def create_figure(
    columns: int = 1,
    aspect_ratio: float = 0.75,
    height: Optional[float] = None,
) -> Tuple[Any, Any]:
    """
    创建符合 JSSC 规范的图形。
    
    Parameters
    ----------
    columns : int
        1 = 单栏 (3.5in), 2 = 双栏 (7.16in)
    aspect_ratio : float
        高宽比。
    height : float, optional
        指定高度 (英寸)。
    
    Returns
    -------
    fig, ax
        Figure 和 Axes 对象。
    """
    if columns == 1:
        width = 3.5
    else:
        width = 7.16
    
    if height is None:
        height = width * aspect_ratio
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    return fig, ax


# =============================================================================
# JSSC 风格绘图函数
# =============================================================================

def plot_spectrum_jssc(
    freqs: NDArray,
    spectrum_db: NDArray,
    metrics: Dict[str, float],
    *,
    title: str = "",
    fs: float = 1e6,
    columns: int = 1,
    show: bool = True,
    save: Optional[str] = None,
    annotate_harmonics: bool = True,
    annotation_style: str = "box",
) -> Any:
    """
    绘制符合 JSSC 规范的频谱图。
    
    Parameters
    ----------
    freqs : NDArray
        频率 (Hz)。
    spectrum_db : NDArray
        功率谱 (dB)。
    metrics : Dict
        SNR, SFDR, ENOB 等指标。
    title : str
        图标题。
    fs : float
        采样率。
    columns : int
        1=单栏, 2=双栏。
    show : bool
        是否显示。
    save : str, optional
        保存路径。
    annotate_harmonics : bool
        是否标注谐波。
    annotation_style : str
        标注样式: 'box', 'arrow', 'simple'
    
    Returns
    -------
    fig
        Figure 对象。
    """
    with jssc_style():
        fig, ax = create_figure(columns=columns, aspect_ratio=0.6)
        
        # 绘制频谱
        ax.plot(freqs / 1e3, spectrum_db, 'k-', linewidth=0.6)
        
        # 设置坐标轴
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        if title:
            ax.set_title(title)
        
        # 设置范围
        ax.set_xlim([0, freqs[-1] / 1e3])
        y_min = max(-120, np.min(spectrum_db) - 10)
        ax.set_ylim([y_min, 10])
        
        # 网格
        ax.grid(True, which='major', linestyle='--', alpha=0.4, linewidth=0.4)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.3)
        
        # 标注基波
        signal_bin = np.argmax(spectrum_db[1:]) + 1
        signal_freq = freqs[signal_bin] / 1e3
        signal_power = spectrum_db[signal_bin]
        
        if annotate_harmonics:
            ax.annotate(
                'Fund.',
                xy=(signal_freq, signal_power),
                xytext=(signal_freq + freqs[-1]/1e3*0.05, signal_power + 5),
                fontsize=7,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='black'),
            )
        
        # 指标文本框
        if annotation_style == "box":
            textstr = '\n'.join([
                f'SNR = {metrics.get("snr", 0):.1f} dB',
                f'SFDR = {metrics.get("sfdr", 0):.1f} dB',
                f'THD = {metrics.get("thd", 0):.1f} dB',
                f'ENOB = {metrics.get("enob", 0):.2f} bits',
            ])
            props = dict(boxstyle='square,pad=0.3', facecolor='white', 
                        edgecolor='black', linewidth=0.5)
            ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=props, family='monospace')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
        if show:
            plt.show()
        
        return fig


def plot_inl_dnl_jssc(
    inl: NDArray,
    dnl: NDArray,
    bits: int,
    *,
    title: str = "",
    columns: int = 1,
    show: bool = True,
    save: Optional[str] = None,
) -> Any:
    """
    绘制符合 JSSC 规范的 INL/DNL 图。
    """
    with jssc_style():
        width = 3.5 if columns == 1 else 7.16
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, width*0.8), dpi=150)
        
        codes = np.arange(len(inl))
        
        # INL
        ax1.plot(codes, inl, 'k-', linewidth=0.5)
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
        ax1.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Code')
        ax1.set_ylabel('INL (LSB)')
        ax1.set_xlim([0, len(inl)-1])
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        
        # 标注最大值
        max_idx = np.argmax(np.abs(inl))
        ax1.annotate(f'Max: {inl[max_idx]:.2f}', 
                     xy=(max_idx, inl[max_idx]),
                     fontsize=7)
        
        # DNL
        ax2.plot(codes, dnl, 'k-', linewidth=0.5)
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
        ax2.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Code')
        ax2.set_ylabel('DNL (LSB)')
        ax2.set_xlim([0, len(dnl)-1])
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        
        if title:
            fig.suptitle(title, fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig


def plot_time_domain_jssc(
    timestamps: NDArray,
    input_signal: NDArray,
    reconstructed: NDArray,
    *,
    title: str = "",
    columns: int = 1,
    n_samples: Optional[int] = None,
    show: bool = True,
    save: Optional[str] = None,
) -> Any:
    """
    绘制符合 JSSC 规范的时域图。
    """
    with jssc_style():
        fig, ax = create_figure(columns=columns, aspect_ratio=0.5)
        
        if n_samples is not None:
            timestamps = timestamps[:n_samples]
            input_signal = input_signal[:n_samples]
            reconstructed = reconstructed[:n_samples]
        
        t_us = timestamps * 1e6
        
        ax.plot(t_us, input_signal, 'k-', linewidth=0.6, label='Input')
        ax.plot(t_us, reconstructed, 'k--', linewidth=0.5, label='Reconstructed')
        
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Voltage (V)')
        if title:
            ax.set_title(title)
        
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig


def plot_comparison_jssc(
    results: List[Dict],
    *,
    metric: str = "enob",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    columns: int = 1,
    style: str = "bar",
    show: bool = True,
    save: Optional[str] = None,
) -> Any:
    """
    绘制对比图（柱状图或折线图）。
    
    Parameters
    ----------
    results : List[Dict]
        每个字典包含 'name' 和对应指标。
    metric : str
        要对比的指标名。
    style : str
        'bar' 或 'line'。
    """
    with jssc_style():
        fig, ax = create_figure(columns=columns, aspect_ratio=0.6)
        
        names = [r.get('name', f'#{i}') for i, r in enumerate(results)]
        values = [r.get(metric, 0) for r in results]
        
        if style == "bar":
            bars = ax.bar(names, values, color='#404040', edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', fontsize=7)
        else:
            ax.plot(names, values, 'ko-', linewidth=0.8, markersize=5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or metric.upper())
        if title:
            ax.set_title(title)
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig


__all__ = [
    "JSSC_STYLE",
    "JSSC_SINGLE_COLUMN",
    "JSSC_DOUBLE_COLUMN",
    "jssc_style",
    "apply_jssc_style",
    "create_figure",
    "plot_spectrum_jssc",
    "plot_inl_dnl_jssc",
    "plot_time_domain_jssc",
    "plot_comparison_jssc",
]
