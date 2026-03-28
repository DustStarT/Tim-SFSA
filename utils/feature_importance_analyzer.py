"""
特征重要性分析器
用于分析输入特征对第三阶段模型性能的影响
"""
import torch
import numpy as np
import pandas as pd
import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class TimeHeadFeatureImportanceAnalyzer:
    """
    第三阶段模型特征重要性分析器
    通过多种方法（消融、置换、梯度）分析特征重要性
    """
    
    def __init__(self, model: torch.nn.Module, config: Any, device: str = 'cuda'):
        """
        初始化分析器
        
        Args:
            model: 第三阶段模型
            config: 配置对象
            device: 设备
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
    
    def ablation_analysis(self, feature_dict: Dict[str, torch.Tensor], 
                         target: torch.Tensor) -> Dict[str, float]:
        """
        消融分析：移除每个特征后计算性能下降
        
        Args:
            feature_dict: 输入特征字典
            target: 真实时间标签
            
        Returns:
            特征重要性分数字典（值越大表示越重要）
        """
        logger.info("开始消融分析...")
        
        # 计算基线性能
        with torch.no_grad():
            all_features = torch.cat(list(feature_dict.values()), dim=1)
            baseline_pred = self.model(all_features)
            if baseline_pred.ndim == 2 and baseline_pred.shape[1] >= 1:
                baseline_pred = baseline_pred[:, 0]
            baseline_loss = torch.nn.functional.l1_loss(baseline_pred, target).item()
        
        importance_scores = {}
        
        # 对每个特征进行消融
        for feat_name, feat_tensor in feature_dict.items():
            # 移除该特征
            other_features = [v for k, v in feature_dict.items() if k != feat_name]
            
            if len(other_features) == 0:
                # 如果只剩下一个特征，使用零向量
                masked_input = torch.zeros_like(all_features)
            else:
                masked_input = torch.cat(other_features, dim=1)
            
            # 计算移除该特征后的损失
            with torch.no_grad():
                masked_pred = self.model(masked_input)
                if masked_pred.ndim == 2 and masked_pred.shape[1] >= 1:
                    masked_pred = masked_pred[:, 0]
                masked_loss = torch.nn.functional.l1_loss(masked_pred, target).item()
            
            # 重要性 = 损失增加
            importance = masked_loss - baseline_loss
            importance_scores[feat_name] = importance
            
            logger.debug(f"  特征 {feat_name}: 基线损失={baseline_loss:.4f}, "
                        f"消融损失={masked_loss:.4f}, 重要性={importance:.4f}")
        
        # 归一化到 [0, 1]
        if len(importance_scores) > 0:
            max_importance = max(abs(v) for v in importance_scores.values())
            if max_importance > 0:
                importance_scores = {k: abs(v) / max_importance 
                                   for k, v in importance_scores.items()}
        
        logger.info(f"消融分析完成，基线损失: {baseline_loss:.4f}")
        return importance_scores
    
    def permutation_analysis(self, feature_dict: Dict[str, torch.Tensor], 
                           target: torch.Tensor, n_iterations: int = 10) -> Dict[str, float]:
        """
        置换分析：打乱每个特征值后计算性能下降
        
        Args:
            feature_dict: 输入特征字典
            target: 真实时间标签
            n_iterations: 迭代次数
            
        Returns:
            特征重要性分数字典
        """
        logger.info("开始置换分析...")
        
        # 计算基线性能
        with torch.no_grad():
            all_features = torch.cat(list(feature_dict.values()), dim=1)
            baseline_pred = self.model(all_features)
            if baseline_pred.ndim == 2 and baseline_pred.shape[1] >= 1:
                baseline_pred = baseline_pred[:, 0]
            baseline_loss = torch.nn.functional.l1_loss(baseline_pred, target).item()
        
        importance_scores = {}
        
        # 对每个特征进行置换
        for feat_name, feat_tensor in feature_dict.items():
            permuted_losses = []
            
            for _ in range(n_iterations):
                # 随机置换特征值
                perm_indices = torch.randperm(len(feat_tensor))
                permuted_feat = feat_tensor[perm_indices]
                
                # 构建置换后的输入
                permuted_dict = {k: (permuted_feat if k == feat_name else v) 
                                for k, v in feature_dict.items()}
                permuted_input = torch.cat(list(permuted_dict.values()), dim=1)
                
                # 计算损失
                with torch.no_grad():
                    permuted_pred = self.model(permuted_input)
                    if permuted_pred.ndim == 2 and permuted_pred.shape[1] >= 1:
                        permuted_pred = permuted_pred[:, 0]
                    permuted_loss = torch.nn.functional.l1_loss(permuted_pred, target).item()
                
                permuted_losses.append(permuted_loss)
            
            # 平均置换损失
            avg_permuted_loss = np.mean(permuted_losses)
            
            # 重要性 = 损失增加
            importance = avg_permuted_loss - baseline_loss
            importance_scores[feat_name] = importance
            
            logger.debug(f"  特征 {feat_name}: 基线损失={baseline_loss:.4f}, "
                        f"置换损失={avg_permuted_loss:.4f}, 重要性={importance:.4f}")
        
        # 归一化到 [0, 1]
        if len(importance_scores) > 0:
            max_importance = max(abs(v) for v in importance_scores.values())
            if max_importance > 0:
                importance_scores = {k: abs(v) / max_importance 
                                   for k, v in importance_scores.items()}
        
        logger.info("置换分析完成")
        return importance_scores
    
    def gradient_based_analysis(self, feature_dict: Dict[str, torch.Tensor], 
                               target: torch.Tensor) -> Dict[str, float]:
        """
        基于梯度的特征重要性分析
        
        Args:
            feature_dict: 输入特征字典
            target: 真实时间标签
            
        Returns:
            特征重要性分数字典
        """
        logger.info("开始梯度分析...")
        
        # 计算梯度
        all_features = torch.cat(list(feature_dict.values()), dim=1)
        all_features.requires_grad_(True)
        
        pred = self.model(all_features)
        if pred.ndim == 2 and pred.shape[1] >= 1:
            pred = pred[:, 0]
        
        loss = torch.nn.functional.l1_loss(pred, target)
        loss.backward()
        
        # 获取梯度
        gradients = all_features.grad.abs().mean(dim=0)
        
        # 将梯度分配到各个特征
        importance_scores = {}
        current_idx = 0
        
        for feat_name, feat_tensor in feature_dict.items():
            feat_size = feat_tensor.shape[1]
            feat_grad = gradients[current_idx:current_idx + feat_size]
            importance_scores[feat_name] = float(feat_grad.mean().item())
            current_idx += feat_size
        
        # 归一化到 [0, 1]
        if len(importance_scores) > 0:
            max_importance = max(importance_scores.values())
            if max_importance > 0:
                importance_scores = {k: v / max_importance 
                                   for k, v in importance_scores.items()}
        
        logger.info("梯度分析完成")
        return importance_scores
    
    def shap_analysis(self, feature_dict: Dict[str, torch.Tensor],
                      target: torch.Tensor,
                      max_background: int = 50) -> Dict[str, Any]:
        """
        基于 SHAP DeepExplainer 的特征重要性分析。

        将每个 feature_dict 的 tensor 块关联到对应名称，
        在整列（维度）维度上求 |shap_values| 均值，最终聚合成
        每个命名特征的一维重要性分数，并额外保留原始 shap_matrix
        用于细粒度可视化。

        Args:
            feature_dict: 命名特征张量字典，每个 tensor 形如 (N, d_i)
            target: 真实标签 (N,)
            max_background: 用于 DeepExplainer 的背景样本数

        Returns:
            dict，包含：
              'importance_scores'  -> Dict[str, float]  各特征聚合重要性
              'shap_values'        -> np.ndarray (N, D_total) 逐维 SHAP 值
              'feature_names'      -> List[str]          对应列名
        """
        if not SHAP_AVAILABLE:
            logger.warning("shap 未安装，跳过 SHAP 分析")
            return {}

        logger.info("开始 SHAP DeepExplainer 分析...")

        all_features = torch.cat(list(feature_dict.values()), dim=1)

        # 构建背景样本（随机子采样，加快速度）
        n_bg = min(max_background, all_features.shape[0])
        bg_idx = torch.randperm(all_features.shape[0])[:n_bg]
        background = all_features[bg_idx].detach().to(self.device)
        test_data = all_features.detach().to(self.device)

        # 定义一个 wrapper，确保输出为 (N, 1) 或 (N,)
        class ModelWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x)
                if out.ndim == 2 and out.shape[1] > 1:
                    out = out[:, :1]
                return out

        wrapped = ModelWrapper(self.model).to(self.device)
        wrapped.eval()

        try:
            explainer = shap.DeepExplainer(wrapped, background)
            shap_values = explainer.shap_values(test_data)  # list or ndarray
        except Exception as e:
            logger.warning(f"SHAP DeepExplainer 失败，尝试 GradientExplainer: {e}")
            try:
                explainer = shap.GradientExplainer(wrapped, background)
                shap_values = explainer.shap_values(test_data)
            except Exception as e2:
                logger.error(f"SHAP GradientExplainer 也失败: {e2}")
                return {}

        # shap_values 可能是 list（多输出）或 ndarray
        if isinstance(shap_values, list):
            shap_matrix = shap_values[0]  # 取第一个输出头
        else:
            shap_matrix = shap_values

        if hasattr(shap_matrix, 'cpu'):
            shap_matrix = shap_matrix.cpu().numpy()
        shap_matrix = np.array(shap_matrix)  # (N, D_total)

        # 构建逐列特征名 & 聚合到特征块
        feat_col_names = []
        importance_scores = {}
        col_idx = 0
        for feat_name, feat_tensor in feature_dict.items():
            d = feat_tensor.shape[1]
            for j in range(d):
                feat_col_names.append(f"{feat_name}[{j}]" if d > 1 else feat_name)
            block_shap = shap_matrix[:, col_idx: col_idx + d]
            importance_scores[feat_name] = float(np.abs(block_shap).mean())
            col_idx += d

        # 归一化到 [0, 1]
        max_imp = max(importance_scores.values()) if importance_scores else 1.0
        if max_imp > 0:
            importance_scores = {k: v / max_imp for k, v in importance_scores.items()}

        logger.info("SHAP 分析完成")
        return {
            'importance_scores': importance_scores,
            'shap_values': shap_matrix,
            'feature_names': feat_col_names,
            'feature_dict': feature_dict,
            'test_data': all_features.detach().cpu().numpy(),
        }

    def plot_shap_results(self, shap_result: Dict[str, Any], output_dir: str):
        """
        绘制三张专用 SHAP 可视化图表：
          1. Bar Summary Plot  — 全局平均 |SHAP| 特征重要性（柱状图）
          2. Dot Summary Plot  — Beeswarm/点图，展示每个样本每个特征的 SHAP 值分布
          3. Waterfall Plot    — 单样本（最高预测确信度）局部解释

        Args:
            shap_result: shap_analysis() 的返回值
            output_dir: 图表保存目录
        """
        if not shap_result or not SHAP_AVAILABLE:
            return

        shap_matrix = shap_result['shap_values']        # (N, D)
        feat_names = shap_result['feature_names']       # List[str]
        test_data = shap_result['test_data']            # (N, D)
        feature_dict = shap_result.get('feature_dict', {})
        feat_block_names = list(feature_dict.keys())    # 聚合块名称

        # 聚合到特征块维度（每块取均值），用于 Beeswarm & Waterfall
        block_shap_per_sample = []  # (N, n_blocks)
        block_data_per_sample = []  # (N, n_blocks)
        col_idx = 0
        for feat_name, feat_tensor in feature_dict.items():
            d = feat_tensor.shape[1]
            block_shap_per_sample.append(shap_matrix[:, col_idx: col_idx + d].mean(axis=1))
            block_data_per_sample.append(test_data[:, col_idx: col_idx + d].mean(axis=1))
            col_idx += d

        block_shap = np.stack(block_shap_per_sample, axis=1)  # (N, B)
        block_data = np.stack(block_data_per_sample, axis=1)  # (N, B)

        # ---- 图 1: Bar Summary Plot（全局平均绝对 SHAP 值）----
        try:
            plt.figure(figsize=(8, max(4, len(feat_block_names) * 0.5)))
            mean_abs = np.abs(block_shap).mean(axis=0)  # (B,)
            sorted_idx = np.argsort(mean_abs)
            sorted_names = [feat_block_names[i] for i in sorted_idx]
            sorted_vals = mean_abs[sorted_idx]
            colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(sorted_names)))
            bars = plt.barh(sorted_names, sorted_vals, color=colors, alpha=0.85)
            for bar in bars:
                w = bar.get_width()
                plt.text(w, bar.get_y() + bar.get_height() / 2,
                         f'{w:.4f}', ha='left', va='center', fontsize=9)
            plt.xlabel('平均 |SHAP 值|（特征对预测的贡献幅度）')
            plt.title('SHAP 特征重要性 — 全局 Bar Summary')
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_bar_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("SHAP Bar Summary 已保存")
        except Exception as e:
            logger.warning(f"SHAP Bar Summary 绘制失败: {e}")
            plt.close()

        # ---- 图 2: Dot / Beeswarm Summary Plot ----
        try:
            # 按平均绝对 SHAP 从高到低排列特征（SHAP 惯例反向显示）
            mean_abs = np.abs(block_shap).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]  # 最重要在上
            ordered_names = [feat_block_names[i] for i in order]
            ordered_shap = block_shap[:, order]
            ordered_data = block_data[:, order]

            n_feat = len(ordered_names)
            fig, ax = plt.subplots(figsize=(10, max(5, n_feat * 0.55)))

            # 颜色按特征值归一化（蓝=低，红=高）
            for j, fname in enumerate(ordered_names):
                sv = ordered_shap[:, j]
                fv = ordered_data[:, j]
                # 归一化特征值到 [0, 1]
                fv_min, fv_max = fv.min(), fv.max()
                if fv_max > fv_min:
                    fv_norm = (fv - fv_min) / (fv_max - fv_min)
                else:
                    fv_norm = np.full_like(fv, 0.5)

                # y 轴位置 = n_feat - 1 - j（最重要在顶部）
                y_base = n_feat - 1 - j
                # 加轻微 jitter 避免点重叠
                jitter = np.random.uniform(-0.2, 0.2, size=len(sv))
                scatter = ax.scatter(
                    sv, y_base + jitter,
                    c=fv_norm, cmap='coolwarm', alpha=0.6,
                    s=18, zorder=2
                )

            ax.set_yticks(range(n_feat))
            ax.set_yticklabels(reversed(ordered_names), fontsize=9)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
            ax.set_xlabel('SHAP 值（对预测的正/负影响）')
            ax.set_title('SHAP Dot Summary — 每样本特征贡献分布')
            ax.grid(axis='x', alpha=0.25, linestyle='--')

            # 颜色条
            sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                       norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('特征值（归一化）\n蓝=低  /  红=高', fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_dot_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("SHAP Dot Summary 已保存")
        except Exception as e:
            logger.warning(f"SHAP Dot Summary 绘制失败: {e}")
            plt.close()

        # ---- 图 3: Waterfall Plot（单样本，选预测均值最接近中位数的那个）----
        try:
            # 选取预测期望最代表性的样本（|SHAP 总和| 中位数样本）
            sample_total = np.abs(block_shap.sum(axis=1))
            median_idx = int(np.argmin(np.abs(sample_total - np.median(sample_total))))

            sv_sample = block_shap[median_idx]   # (B,)
            fv_sample = block_data[median_idx]   # (B,)

            # 按 SHAP 值从负到正排列
            order_wf = np.argsort(sv_sample)
            names_wf = [feat_block_names[i] for i in order_wf]
            sv_wf = sv_sample[order_wf]
            fv_wf = fv_sample[order_wf]

            fig, ax = plt.subplots(figsize=(9, max(5, len(names_wf) * 0.55)))
            colors_wf = ['#d73027' if v > 0 else '#4575b4' for v in sv_wf]
            bars = ax.barh(names_wf, sv_wf, color=colors_wf, alpha=0.85)
            ax.axvline(0, color='black', linewidth=0.9)

            for bar, fv in zip(bars, fv_wf):
                w = bar.get_width()
                ha = 'left' if w >= 0 else 'right'
                pad = 0.0005 if w >= 0 else -0.0005
                ax.text(w + pad, bar.get_y() + bar.get_height() / 2,
                        f'SHAP={w:+.4f}\n(feat={fv:.3g})',
                        ha=ha, va='center', fontsize=7.5)

            ax.set_xlabel('SHAP 值')
            ax.set_title(
                f'SHAP Waterfall — 单样本局部解释 (样本 #{median_idx})\n'
                f'红色=使预测增大 / 蓝色=使预测减小'
            )
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_waterfall.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("SHAP Waterfall 已保存")
        except Exception as e:
            logger.warning(f"SHAP Waterfall 绘制失败: {e}")
            plt.close()

    def comprehensive_analysis(self, feature_dict: Dict[str, torch.Tensor], 
                              target: torch.Tensor, 
                              methods: List[str] = ['ablation', 'permutation'],
                              output_dir: str = None) -> Dict[str, Any]:
        """
        综合特征重要性分析
        
        Args:
            feature_dict: 输入特征字典
            target: 真实时间标签
            methods: 分析方法列表，可包含 'ablation', 'permutation', 'gradient', 'shap'
            output_dir: 输出目录
            
        Returns:
            综合分析结果字典
        """
        logger.info("开始综合特征重要性分析...")
        
        results = {}
        shap_detail = None
        
        # 消融分析
        if 'ablation' in methods:
            results['ablation'] = self.ablation_analysis(feature_dict, target)
        
        # 置换分析
        if 'permutation' in methods:
            results['permutation'] = self.permutation_analysis(feature_dict, target)
        
        # 梯度分析
        if 'gradient' in methods:
            results['gradient'] = self.gradient_based_analysis(feature_dict, target)

        # SHAP 分析
        if 'shap' in methods:
            shap_detail = self.shap_analysis(feature_dict, target)
            if shap_detail and 'importance_scores' in shap_detail:
                results['shap'] = shap_detail['importance_scores']
        
        # 计算平均重要性（仅用标量 importance_scores 组成的 method 键）
        scalar_results = {k: v for k, v in results.items()}
        if len(scalar_results) > 0:
            all_features = set()
            for method_results in scalar_results.values():
                all_features.update(method_results.keys())
            
            avg_importance = {}
            for feat in all_features:
                importances = [r.get(feat, 0) for r in scalar_results.values()]
                avg_importance[feat] = np.mean(importances)
            
            results['average'] = avg_importance
        
        # 保存结果
        if output_dir:
            results_path = os.path.join(output_dir, 'feature_importance_analysis.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"特征重要性分析结果已保存到: {results_path}")
            
            # 绘制综合可视化（柱状图 / 热力图 / 雷达图）
            self.plot_importance_results(results, output_dir)

            # 绘制 SHAP 专项图（若已运行 SHAP）
            if shap_detail:
                self.plot_shap_results(shap_detail, output_dir)
        
        logger.info("综合特征重要性分析完成")
        return results
    
    def plot_importance_results(self, results: Dict[str, Dict[str, float]], 
                                output_dir: str):
        """
        绘制特征重要性结果
        
        Args:
            results: 分析结果字典
            output_dir: 输出目录
        """
        try:
            # 基础准备：获取所有特征并按重要性排序
            if 'average' in results:
                sorted_features = sorted(results['average'].keys(), 
                                      key=lambda k: results['average'][k])
            else:
                first_method = list(results.keys())[0]
                sorted_features = sorted(results[first_method].keys(), 
                                      key=lambda k: results[first_method][k])

            # --- 1. 改良的对齐多子图水平柱状图 ---
            n_methods = len([k for k in results.keys() if k != 'average'])
            fig, axes = plt.subplots(1, n_methods + (1 if 'average' in results else 0), 
                                   figsize=(5 * (n_methods + 1), max(6, len(sorted_features) * 0.3)))
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            plot_idx = 0
            for method_name, importance_scores in results.items():
                if method_name == 'average':
                    continue
                
                scores = [importance_scores.get(f, 0) for f in sorted_features]
                ax = axes[plot_idx]
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_features)))
                bars = ax.barh(sorted_features, scores, alpha=0.8, color=colors)
                ax.set_xlabel('重要性分数')
                ax.set_title(f'{method_name.title()} 分析')
                ax.grid(alpha=0.3, axis='x', linestyle='--')
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                            ha='left', va='center', fontsize=8, alpha=0.8)
                plot_idx += 1
            
            if 'average' in results:
                ax = axes[plot_idx]
                scores = [results['average'].get(f, 0) for f in sorted_features]
                colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(sorted_features)))
                bars = ax.barh(sorted_features, scores, alpha=0.8, color=colors)
                ax.set_xlabel('重要性分数')
                ax.set_title('平均重要性')
                ax.grid(alpha=0.3, axis='x', linestyle='--')
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                            ha='left', va='center', fontsize=8, alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance_comprehensive.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # --- 2. 特征重要性跨方法对比热力图 ---
            df_data = {}
            for method_name, importance_scores in results.items():
                df_data[method_name] = [importance_scores.get(f, 0) for f in reversed(sorted_features)]
            df_scores = pd.DataFrame(df_data, index=list(reversed(sorted_features)))
            
            plt.figure(figsize=(max(6, len(results) * 1.5), max(6, len(sorted_features) * 0.4)))
            sns.heatmap(df_scores, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': '重要性分数'})
            plt.title('特征重要性跨方法对比热力图')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

            # --- 3. Top-K 特征雷达图 ---
            top_k = min(8, len(sorted_features))
            if top_k >= 3:
                top_features = list(reversed(sorted_features))[:top_k]
                angles = np.linspace(0, 2 * np.pi, top_k, endpoint=False).tolist()
                angles += angles[:1]
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                
                for method_name, importance_scores in results.items():
                    values = [importance_scores.get(f, 0) for f in top_features]
                    values += values[:1]
                    
                    linestyle = '-' if method_name == 'average' else '--'
                    linewidth = 2.5 if method_name == 'average' else 1.5
                    alpha = 0.9 if method_name == 'average' else 0.6
                    
                    ax.plot(angles, values, label=method_name, 
                           linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                    if method_name == 'average':
                        ax.fill(angles, values, alpha=0.1)
                        
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(top_features, fontsize=10)
                ax.set_title(f'Top {top_k} 特征多视角重要性分布雷达图', pad=20)
                plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance_radar.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("所有特征重要性图表（柱状图、热力图、雷达图）已保存")
        except Exception as e:
            logger.warning(f"绘制特征重要性图表失败: {e}")
