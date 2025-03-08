#!/usr/bin/env python3
"""
enhanced_visualization.py

提供增强的可视化功能，用于PET数据的多切片和多角度可视化，以及完整环与不完整环数据的比较。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

def visualize_sinogram_multislice(sinogram, output_path, title="Sinogram Visualization", 
                                 slice_indices=None, num_slices=5, figsize=(15, 10)):
    """
    可视化正弦图的多个切片，放在同一个图上。
    
    Args:
        sinogram: 形状为 (N1, N2, N3) 的正弦图数组
        output_path: 输出图像的保存路径
        title: 图像标题
        slice_indices: 要显示的切片索引列表（如果为None，则自动选择）
        num_slices: 如果slice_indices为None，要显示的切片数量
        figsize: 图像大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取正弦图形状
    depth = sinogram.shape[2]
    
    # 如果未指定切片索引，则自动计算
    if slice_indices is None:
        # 均匀选择切片
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    # 创建图形和子图
    fig = plt.figure(figsize=figsize)
    
    # 使用GridSpec实现更灵活的布局
    n_rows = (len(slice_indices) + 2) // 3  # 每行3个切片
    gs = GridSpec(n_rows, 3, figure=fig)
    
    # 全局最大值和最小值（用于颜色范围）
    vmin, vmax = np.percentile(sinogram, [2, 98])
    
    # 创建子图并绘制每个切片
    for i, slice_idx in enumerate(slice_indices):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # 提取切片数据
        slice_data = sinogram[0, :, slice_idx] if sinogram.shape[0] == 1 else sinogram[:, :, slice_idx]
        
        # 绘制切片并保持宽高比适当
        im = ax.imshow(slice_data, cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f'Slice {slice_idx}')
        ax.set_xlabel('Radial Position')
        ax.set_ylabel('Angle')
    
    # 添加颜色条并调整布局
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    
    # 添加全局标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def visualize_sinogram_multi_perspective(sinogram, output_path, title="Multi-Perspective Sinogram", 
                                        figsize=(18, 12)):
    """
    从多个角度可视化正弦图：
    1. 角度-空间视图 (轴向切片)
    2. 角度-环差视图
    3. 空间-环差视图
    
    Args:
        sinogram: 形状为 (角度, 空间, 环差) 的正弦图数组
        output_path: 输出图像的保存路径
        title: 图像标题
        figsize: 图像大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取数据尺寸
    n_angles, n_radial, n_rings = sinogram.shape
    
    # 创建图形和子图
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # 获取合适的数据范围
    vmin, vmax = np.percentile(sinogram, [2, 98])
    
    # 1. 角度-径向视图 (中间环差切片)
    ax1 = fig.add_subplot(gs[0, 0])
    middle_ring = n_rings // 2
    im1 = ax1.imshow(sinogram[:, :, middle_ring], cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Angle-Radial View (Ring {middle_ring})')
    ax1.set_xlabel('Radial Position')
    ax1.set_ylabel('Angle')
    fig.colorbar(im1, ax=ax1)
    
    # 2. 角度-环差视图 (中间径向切片)
    ax2 = fig.add_subplot(gs[0, 1])
    middle_radial = n_radial // 2
    im2 = ax2.imshow(sinogram[:, middle_radial, :], cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Angle-Ring View (Radial {middle_radial})')
    ax2.set_xlabel('Ring Difference')
    ax2.set_ylabel('Angle')
    fig.colorbar(im2, ax=ax2)
    
    # 3. 径向-环差视图 (中间角度切片)
    ax3 = fig.add_subplot(gs[1, 0])
    middle_angle = n_angles // 2
    im3 = ax3.imshow(sinogram[middle_angle, :, :], cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Radial-Ring View (Angle {middle_angle})')
    ax3.set_xlabel('Ring Difference')
    ax3.set_ylabel('Radial Position')
    fig.colorbar(im3, ax=ax3)
    
    # 4. 叠加视图 (最大投影)
    ax4 = fig.add_subplot(gs[1, 1])
    # 计算沿环差方向的最大投影
    max_projection = np.max(sinogram, axis=2)
    im4 = ax4.imshow(max_projection, cmap='magma', aspect='auto', vmin=vmin, vmax=vmax)
    ax4.set_title('Maximum Intensity Projection')
    ax4.set_xlabel('Radial Position')
    ax4.set_ylabel('Angle')
    fig.colorbar(im4, ax=ax4)
    
    # 添加全局标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题留出空间
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def visualize_reconstruction_comparison(original, complete_recon, incomplete_recon, output_path, 
                                       slice_indices=None, num_slices=3, figsize=(18, 12)):
    """
    比较可视化：原始图像、完整环重建和不完整环重建。
    
    Args:
        original: 原始3D图像
        complete_recon: 使用完整环数据重建的3D图像
        incomplete_recon: 使用不完整环数据重建的3D图像
        output_path: 输出图像保存路径
        slice_indices: 要显示的切片索引，如果为None则自动选择
        num_slices: 要显示的切片数量
        figsize: 图像大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 确保所有输入图像具有相同的形状
    if not (original.shape == complete_recon.shape == incomplete_recon.shape):
        raise ValueError("所有输入图像必须具有相同的形状")
    
    # 获取图像形状
    depth = original.shape[0]
    
    # 如果未指定切片索引，则自动计算
    if slice_indices is None:
        # 均匀选择切片
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 为每个切片创建三列（原始、完整重建、不完整重建）
    gs = GridSpec(num_slices, 3, figure=fig)
    
    # 确定全局颜色范围
    all_data = np.concatenate([
        original.flatten(), 
        complete_recon.flatten(), 
        incomplete_recon.flatten()
    ])
    vmin, vmax = np.percentile(all_data, [2, 98])
    
    # 为每个切片创建可视化
    for i, slice_idx in enumerate(slice_indices):
        # 原始图像
        ax1 = fig.add_subplot(gs[i, 0])
        im1 = ax1.imshow(original[slice_idx], cmap='magma', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Original (Slice {slice_idx})')
        ax1.axis('off')
        
        # 完整环重建
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(complete_recon[slice_idx], cmap='magma', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Complete Ring Recon')
        ax2.axis('off')
        
        # 不完整环重建
        ax3 = fig.add_subplot(gs[i, 2])
        im3 = ax3.imshow(incomplete_recon[slice_idx], cmap='magma', vmin=vmin, vmax=vmax)
        ax3.set_title(f'Incomplete Ring Recon')
        ax3.axis('off')
    
    # 添加颜色条
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im3, cax=cbar_ax)
    
    # 添加全局标题
    fig.suptitle('Reconstruction Comparison', fontsize=16, y=0.98)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def visualize_multi_axial_views(volume, output_path, title="Multi-Axial Views", figsize=(15, 15)):
    """
    以三个主要轴向视图显示3D体积：轴向、冠状和矢状。
    
    Args:
        volume: 3D体积数据
        output_path: 输出图像保存路径
        title: 图像标题
        figsize: 图像大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取体积形状
    x_dim, y_dim, z_dim = volume.shape
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # 确定全局颜色范围
    vmin, vmax = np.percentile(volume, [2, 98])
    
    # 1. 轴向视图 (顶视图)
    ax1 = fig.add_subplot(gs[0, 0])
    axial_slice = volume[:, :, z_dim//2]
    im1 = ax1.imshow(axial_slice, cmap='magma', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Axial View (z={z_dim//2})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1)
    
    # 2. 冠状视图 (前视图)
    ax2 = fig.add_subplot(gs[0, 1])
    coronal_slice = volume[:, y_dim//2, :]
    im2 = ax2.imshow(coronal_slice, cmap='magma', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Coronal View (y={y_dim//2})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    fig.colorbar(im2, ax=ax2)
    
    # 3. 矢状视图 (侧视图)
    ax3 = fig.add_subplot(gs[1, 0])
    sagittal_slice = volume[x_dim//2, :, :]
    im3 = ax3.imshow(sagittal_slice, cmap='magma', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Sagittal View (x={x_dim//2})')
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')
    fig.colorbar(im3, ax=ax3)
    
    # 4. 三维体积渲染（最大强度投影）
    ax4 = fig.add_subplot(gs[1, 1])
    mip_z = np.max(volume, axis=2)
    im4 = ax4.imshow(mip_z, cmap='magma', vmin=vmin, vmax=vmax)
    ax4.set_title('Maximum Intensity Projection (Z)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    fig.colorbar(im4, ax=ax4)
    
    # 添加全局标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def compare_sinograms(complete_sinogram, incomplete_sinogram, output_path, 
                     title="Complete vs Incomplete Sinogram Comparison", 
                     slice_indices=None, num_slices=3, figsize=(18, 12)):
    """
    比较完整环和不完整环正弦图的差异。
    
    Args:
        complete_sinogram: 完整环正弦图数据
        incomplete_sinogram: 不完整环正弦图数据
        output_path: 输出图像保存路径
        title: 图像标题
        slice_indices: 要显示的切片索引，如果为None则自动选择
        num_slices: 要显示的切片数量
        figsize: 图像大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 检查两个正弦图的形状是否相同
    if complete_sinogram.shape != incomplete_sinogram.shape:
        print(f"Warning: Sinogram shapes don't match - Complete: {complete_sinogram.shape}, Incomplete: {incomplete_sinogram.shape}")
        # 如果形状不同，尝试裁剪到最小共同大小
        min_shape = [min(s1, s2) for s1, s2 in zip(complete_sinogram.shape, incomplete_sinogram.shape)]
        complete_sinogram = complete_sinogram[:min_shape[0], :min_shape[1], :min_shape[2]]
        incomplete_sinogram = incomplete_sinogram[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # 获取正弦图形状
    depth = complete_sinogram.shape[2]
    
    # 如果未指定切片索引，则自动计算
    if slice_indices is None:
        # 均匀选择切片
        slice_indices = np.linspace(0, min(41, depth-1), num_slices, dtype=int)
    
    # 创建差异正弦图(绝对差)
    difference = np.abs(complete_sinogram - incomplete_sinogram)
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 为每个切片创建三行（完整环、不完整环、差异）
    n_rows = num_slices
    n_cols = 3
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # 为完整和不完整正弦图创建共享的颜色范围
    vmin_all = min(np.percentile(complete_sinogram, 1), np.percentile(incomplete_sinogram, 1))
    vmax_all = max(np.percentile(complete_sinogram, 99), np.percentile(incomplete_sinogram, 99))
    
    # 为差异图创建单独的颜色范围
    vmin_diff = np.percentile(difference, 1)
    vmax_diff = np.percentile(difference, 99)
    
    # 绘制每个切片的比较
    for i, slice_idx in enumerate(slice_indices):
        # 完整环正弦图
        ax1 = fig.add_subplot(gs[i, 0])
        slice_data_complete = complete_sinogram[:, :, slice_idx]
        im1 = ax1.imshow(slice_data_complete, cmap='magma', aspect='auto', vmin=vmin_all, vmax=vmax_all)
        ax1.set_title(f'Complete Ring (Slice {slice_idx})')
        if i == n_rows - 1:  # 只在底行添加x轴标签
            ax1.set_xlabel('Radial Position')
        ax1.set_ylabel('Angle')
        
        # 不完整环正弦图
        ax2 = fig.add_subplot(gs[i, 1])
        slice_data_incomplete = incomplete_sinogram[:, :, slice_idx]
        im2 = ax2.imshow(slice_data_incomplete, cmap='magma', aspect='auto', vmin=vmin_all, vmax=vmax_all)
        ax2.set_title(f'Incomplete Ring (Slice {slice_idx})')
        if i == n_rows - 1:
            ax2.set_xlabel('Radial Position')
        ax2.set_ylabel('Angle')
        
        # 差异正弦图
        ax3 = fig.add_subplot(gs[i, 2])
        slice_data_diff = difference[:, :, slice_idx]
        im3 = ax3.imshow(slice_data_diff, cmap='hot', aspect='auto', vmin=vmin_diff, vmax=vmax_diff)
        ax3.set_title(f'Absolute Difference (Slice {slice_idx})')
        if i == n_rows - 1:
            ax3.set_xlabel('Radial Position')
        ax3.set_ylabel('Angle')
    
    # 添加颜色条
    plt.tight_layout()
    
    # 为正弦图添加颜色条
    cbar_ax1 = fig.add_axes([0.91, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(im1, cax=cbar_ax1)
    cbar1.set_label('Sinogram Intensity')
    
    # 为差异图添加颜色条
    cbar_ax2 = fig.add_axes([0.91, 0.15, 0.02, 0.3])
    cbar2 = fig.colorbar(im3, cax=cbar_ax2)
    cbar2.set_label('Absolute Difference')
    
    # 添加全局标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path