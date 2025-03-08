#!/usr/bin/env python3
"""
enhanced_visualization.py

提供增强的可视化功能，用于PET数据的多切片和多角度可视化。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

def create_enhanced_visualizations_thread(image, result_3d, sinogram, log_dir, image_filename):
    """在后台线程中创建增强可视化"""
    import threading
    
    def _create_visualizations():
        # 确保使用非交互式后端
        matplotlib.use('Agg')
        
        try:
            # 创建输出目录
            vis_dir = os.path.join(log_dir, "enhanced_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 基础文件名（不带扩展名）
            base_name = os.path.splitext(image_filename)[0]
            
            # 1. 多切片正弦图可视化
            sinogram_vis_path = os.path.join(vis_dir, f"{base_name}_sinogram_slices.png")
            visualize_sinogram_multislice(
                sinogram.numpy(),
                sinogram_vis_path,
                title=f"Sinogram Multiple Slices ({base_name})",
                num_slices=8
            )
            print(f"  -> Saved enhanced sinogram visualization to {sinogram_vis_path}")
            
            # 2. 体积多视角切片可视化
            volume_vis_path = os.path.join(vis_dir, f"{base_name}_volume_views.png")
            visualize_multi_axial_views(
                result_3d,
                volume_vis_path,
                title=f"Reconstructed Volume: Multi-Axial Views ({base_name})"
            )
            print(f"  -> Saved enhanced volume visualization to {volume_vis_path}")
            
            # 3. 正弦图的多角度视图
            sinogram_mp_path = os.path.join(vis_dir, f"{base_name}_sinogram_perspectives.png")
            visualize_sinogram_multi_perspective(
                sinogram.numpy(),
                sinogram_mp_path,
                title=f"Sinogram: Multi-Perspective View ({base_name})"
            )
            print(f"  -> Saved multi-perspective sinogram visualization to {sinogram_mp_path}")

            # 4. 三视图比较（原始图像、重建图像和正弦图）
            comparison_path = os.path.join(vis_dir, f"{base_name}_three_view_comparison.png")
            create_three_view_comparison(
                image=image,
                result_3d=result_3d,
                sinogram=sinogram.numpy(),
                output_path=comparison_path,
                title=f"Three-View Comparison ({base_name})"
            )
            print(f"  -> Saved three-view comparison to {comparison_path}")
            
        except Exception as e:
            print(f"Warning: Failed to create enhanced visualizations: {e}")
    
    # 创建并启动线程
    thread = threading.Thread(target=_create_visualizations)
    thread.daemon = False
    thread.start()
    return thread

def create_three_view_comparison(image, result_3d, sinogram, output_path, title="Three-View Comparison"):
    """创建三视图比较：原始图像、重建图像和正弦图"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取切片索引
    slice_index = result_3d.shape[2] // 2
    
    # 创建图形和子图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    im0 = axs[0].imshow(image[:, :, slice_index], cmap='magma', interpolation='nearest')
    axs[0].set_title(f'Original Image Slice (z = {slice_index})')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # 重建图像
    im1 = axs[1].imshow(result_3d[:, :, slice_index], cmap='magma', interpolation='nearest')
    axs[1].set_title(f'Reconstructed Slice')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    # 正弦图
    im2 = axs[2].imshow(sinogram[0, :, :42], cmap='magma', aspect='auto')
    axs[2].set_title(f'Sinogram First 42 Slices')
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    # 添加全局标题
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path