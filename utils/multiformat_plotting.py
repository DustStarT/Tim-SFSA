"""
多格式绘图工具 - 自动为所有matplotlib图表生成多种格式

使用方法：
    在main.py或run_experiment开始时调用一次：
    from utils.multiformat_plotting import enable_multiformat_plotting
    enable_multiformat_plotting(config)
    
    之后所有的 plt.savefig() 和 fig.savefig() 调用都会自动保存多种格式
"""

import os
import logging
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.figure

logger = logging.getLogger(__name__)

# 全局配置
_PLOT_FORMATS = ['png', 'pdf']
_PLOT_DPI = 200
_ENABLED = False
_ORIGINAL_SAVEFIG = None
_ORIGINAL_FIG_SAVEFIG = None


def enable_multiformat_plotting(config=None, formats=None, dpi=None):
    """
    启用多格式绘图功能
    
    这会monkey-patch matplotlib的savefig方法，使其自动保存多种格式
    
    参数:
        config: 配置对象，从中读取 visualization.plot_formats 和 visualization.plot_dpi
        formats: 强制指定的格式列表（覆盖配置），如 ['png', 'pdf', 'svg']
        dpi: 强制指定的DPI（覆盖配置）
    
    示例:
        # 从配置启用
        enable_multiformat_plotting(config)
        
        # 手动指定格式
        enable_multiformat_plotting(formats=['png', 'svg', 'pdf'], dpi=300)
    """
    global _PLOT_FORMATS, _PLOT_DPI, _ENABLED
    global _ORIGINAL_SAVEFIG, _ORIGINAL_FIG_SAVEFIG
    
    if _ENABLED:
        logger.warning("多格式绘图已经启用，跳过重复启用")
        return
    
    # 从配置读取格式和DPI
    if config is not None:
        try:
            if hasattr(config, 'visualization'):
                if hasattr(config.visualization, 'plot_formats') and formats is None:
                    formats = config.visualization.plot_formats
                if hasattr(config.visualization, 'plot_dpi') and dpi is None:
                    dpi = config.visualization.plot_dpi
        except Exception as e:
            logger.debug(f"从配置读取绘图参数失败: {e}")
    
    # 设置默认值
    if formats is not None:
        _PLOT_FORMATS = list(formats) if isinstance(formats, (list, tuple)) else [formats]
    if dpi is not None:
        _PLOT_DPI = int(dpi)
    
    logger.info(f"启用多格式绘图功能: 格式={_PLOT_FORMATS}, DPI={_PLOT_DPI}")
    
    # 保存原始方法
    _ORIGINAL_SAVEFIG = plt.savefig
    _ORIGINAL_FIG_SAVEFIG = matplotlib.figure.Figure.savefig
    
    # 创建包装方法
    def multiformat_savefig_wrapper(*args, **kwargs):
        """plt.savefig的包装器"""
        # 第一个参数通常是文件名
        if len(args) > 0:
            fname = args[0]
            args = args[1:]
        elif 'fname' in kwargs:
            fname = kwargs.pop('fname')
        else:
            logger.warning("savefig调用缺少文件名参数")
            return _ORIGINAL_SAVEFIG(*args, **kwargs)
        
        # 调用多格式保存
        return _save_multiformat(fname, False, *args, **kwargs)
    
    def multiformat_fig_savefig_wrapper(self, *args, **kwargs):
        """Figure.savefig的包装器"""
        # 第一个参数通常是文件名
        if len(args) > 0:
            fname = args[0]
            args = args[1:]
        elif 'fname' in kwargs:
            fname = kwargs.pop('fname')
        else:
            logger.warning("Figure.savefig调用缺少文件名参数")
            return _ORIGINAL_FIG_SAVEFIG(self, *args, **kwargs)
        
        # 调用多格式保存
        return _save_multiformat(fname, True, self, *args, **kwargs)
    
    # 替换方法
    plt.savefig = multiformat_savefig_wrapper
    matplotlib.figure.Figure.savefig = multiformat_fig_savefig_wrapper
    
    _ENABLED = True
    logger.info("✓ 多格式绘图功能已启用")


def disable_multiformat_plotting():
    """禁用多格式绘图功能，恢复原始的savefig方法"""
    global _ENABLED, _ORIGINAL_SAVEFIG, _ORIGINAL_FIG_SAVEFIG
    
    if not _ENABLED:
        logger.warning("多格式绘图未启用，无需禁用")
        return
    
    if _ORIGINAL_SAVEFIG is not None:
        plt.savefig = _ORIGINAL_SAVEFIG
    if _ORIGINAL_FIG_SAVEFIG is not None:
        matplotlib.figure.Figure.savefig = _ORIGINAL_FIG_SAVEFIG
    
    _ENABLED = False
    logger.info("多格式绘图功能已禁用")


def _save_multiformat(fname, is_figure_method, *args, **kwargs):
    """
    实际执行多格式保存的内部函数
    
    参数:
        fname: 文件名（可能包含或不包含扩展名）
        is_figure_method: 是否是Figure.savefig方法（True）还是plt.savefig（False）
        *args, **kwargs: 传递给原始savefig的其他参数
    """
    # 处理文件名：移除可能存在的扩展名
    base_path = str(fname)
    original_ext = None
    
    for ext in ['.png', '.pdf', '.svg', '.eps', '.jpg', '.jpeg']:
        if base_path.lower().endswith(ext):
            original_ext = ext
            base_path = base_path[:-len(ext)]
            break
    
    # 确保目录存在
    output_dir = os.path.dirname(base_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"创建输出目录失败: {output_dir}, 错误: {e}")
    
    # 决定要保存的格式
    formats_to_save = _PLOT_FORMATS.copy()
    
    # 如果原始调用指定了扩展名，确保该格式在列表中
    if original_ext:
        original_fmt = original_ext.strip('.')
        if original_fmt not in formats_to_save:
            formats_to_save.insert(0, original_fmt)
    
    # 设置默认DPI（如果未指定）
    if 'dpi' not in kwargs:
        kwargs['dpi'] = _PLOT_DPI
    
    # 保存每种格式
    saved_files = []
    errors = []
    
    for fmt in formats_to_save:
        fmt = fmt.lower().strip('.')
        output_path = f"{base_path}.{fmt}"
        
        try:
            if is_figure_method:
                # 是Figure.savefig调用
                fig = args[0]
                args_rest = args[1:]
                _ORIGINAL_FIG_SAVEFIG(fig, output_path, *args_rest, **kwargs)
            else:
                # 是plt.savefig调用
                _ORIGINAL_SAVEFIG(output_path, *args, **kwargs)
            
            saved_files.append(output_path)
        except Exception as e:
            error_msg = f"保存 {fmt} 格式失败: {output_path}, 错误: {e}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    # 记录结果
    if saved_files:
        if len(saved_files) == 1:
            logger.debug(f"图表已保存: {saved_files[0]}")
        else:
            logger.debug(f"图表已保存为 {len(saved_files)} 种格式: {base_path}.[{','.join(formats_to_save)}]")
    
    if errors and not saved_files:
        # 如果所有格式都失败了，抛出最后一个错误
        raise RuntimeError(f"所有格式保存均失败: {'; '.join(errors)}")


def set_plot_formats(formats: List[str]):
    """动态设置绘图格式"""
    global _PLOT_FORMATS
    _PLOT_FORMATS = list(formats)
    logger.info(f"已更新绘图格式: {_PLOT_FORMATS}")


def set_plot_dpi(dpi: int):
    """动态设置绘图DPI"""
    global _PLOT_DPI
    _PLOT_DPI = int(dpi)
    logger.info(f"已更新绘图DPI: {_PLOT_DPI}")


def get_current_settings():
    """获取当前设置"""
    return {
        'enabled': _ENABLED,
        'formats': _PLOT_FORMATS.copy(),
        'dpi': _PLOT_DPI
    }


# 兼容性：提供直接保存函数
def save_figure(fname, fig=None, formats=None, dpi=None, **kwargs):
    """
    直接保存图表为多种格式（不需要启用monkey-patch）
    
    参数:
        fname: 文件名（不含扩展名）
        fig: Figure对象（None表示当前图表）
        formats: 格式列表（None使用全局设置）
        dpi: DPI（None使用全局设置）
        **kwargs: 传递给savefig的其他参数
    """
    if formats is None:
        formats = _PLOT_FORMATS
    if dpi is None:
        dpi = _PLOT_DPI
    
    # 移除可能的扩展名
    base_path = str(fname)
    for ext in ['.png', '.pdf', '.svg', '.eps', '.jpg', '.jpeg']:
        if base_path.lower().endswith(ext):
            base_path = base_path[:-len(ext)]
            break
    
    # 确保目录存在
    output_dir = os.path.dirname(base_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        fmt = fmt.lower().strip('.')
        output_path = f"{base_path}.{fmt}"
        
        try:
            if fig is not None:
                fig.savefig(output_path, dpi=dpi, **kwargs)
            else:
                plt.savefig(output_path, dpi=dpi, **kwargs)
            saved_files.append(output_path)
        except Exception as e:
            logger.warning(f"保存 {fmt} 格式失败: {output_path}, 错误: {e}")
    
    return saved_files

