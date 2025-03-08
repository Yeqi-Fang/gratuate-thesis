#!/usr/bin/env python3
"""
test_listmode_to_incomplete.py

单元测试文件，用于测试listmode_to_incomplete.py中的功能。
"""

import os
import sys
import unittest
import numpy as np
import torch
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# 添加项目根目录到PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入要测试的模块
import listmode_to_incomplete as lti

class TestListModeToIncomplete(unittest.TestCase):
    """测试listmode_to_incomplete.py中的核心功能"""

    def setUp(self):
        """测试前准备工作"""
        # 创建临时目录用于测试输出
        self.test_dir = tempfile.mkdtemp()
        
        # 定义测试用PET扫描仪配置
        self.test_info = {
            'NrCrystalsPerRing': 16,  # 使用较小的环简化测试
            'NrRings': 4,
            'crystalTransSpacing': 4.0,
            'crystalAxialSpacing': 5.0
        }
        
        # 创建模拟的事件数据
        # 格式: [det1_id, det2_id, ...]
        self.mock_events = np.array([
            [0, 8],   # 探测器0和8（相对）
            [1, 9],   # 探测器1和9（相对）
            [2, 10],  # 探测器2和10（相对）
            [3, 11],  # 探测器3和11（相对）
            [4, 12],  # 探测器4和12（相对）
            [5, 13],  # 探测器5和13（相对）
            [6, 14],  # 探测器6和14（相对）
            [7, 15],  # 探测器7和15（相对）
            [16, 24], # 环1的探测器和对应探测器
            [32, 40]  # 环2的探测器和对应探测器
        ], dtype=np.int32)
        
        # 创建一个模拟的正弦图
        self.mock_sinogram = np.random.rand(8, 16, 20).astype(np.float32)
        
        # 保存模拟正弦图到临时文件
        os.makedirs(os.path.join(self.test_dir, 'sinogram'), exist_ok=True)
        self.sinogram_path = os.path.join(self.test_dir, 'sinogram', 'reconstructed_index123_num1000.npy')
        np.save(self.sinogram_path, self.mock_sinogram)

    def tearDown(self):
        """测试后清理工作"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)

    def test_build_missing_detector_ids(self):
        """测试构建缺失探测器ID集合的功能"""
        # 定义缺失扇区（探测器0-3对应的角度范围）
        missing_sectors = [(0, 90)]  # 0-90度
        
        # 获取缺失的探测器ID
        missing_ids = lti.build_missing_detector_ids(
            crystals_per_ring=self.test_info['NrCrystalsPerRing'],
            num_rings=self.test_info['NrRings'],
            missing_sectors=missing_sectors
        )
        
        # 计算预期结果: 
        # 在16探测器环上，每个探测器占22.5度(360/16)
        # 所以0-90度对应探测器0, 1, 2, 3, 4
        expected_ids = set()
        for ring in range(self.test_info['NrRings']):
            base_id = ring * self.test_info['NrCrystalsPerRing']
            for crystal in range(5):  # 0, 1, 2, 3, 4 (0到90度)
                expected_ids.add(base_id + crystal)
        
        # 验证结果
        self.assertEqual(missing_ids, expected_ids)

    def test_filter_listmode_data(self):
        """测试事件过滤功能"""
        # 定义缺失探测器ID（前两个探测器位于缺失区域）
        missing_ids = {0, 1, 8, 9}  # 对应事件数组的前两行
        
        # 过滤事件
        filtered_events = lti.filter_listmode_data(self.mock_events, missing_ids)
        
        # 验证结果：应保留原始事件中除前两行外的所有行
        expected_events = self.mock_events[2:]
        np.testing.assert_array_equal(filtered_events, expected_events)
        
        # 验证事件数量是否正确
        self.assertEqual(len(filtered_events), len(self.mock_events) - 2)

    def test_load_complete_sinogram(self):
        """测试加载完整环正弦图功能"""
        # 尝试加载存在的正弦图
        sinogram = lti.load_complete_sinogram(
            sinogram_dir=os.path.join(self.test_dir, 'sinogram'),
            index=123,
            num_events=1000
        )
        
        # 验证正弦图是否成功加载
        self.assertIsNotNone(sinogram)
        np.testing.assert_array_equal(sinogram, self.mock_sinogram)
        
        # 尝试加载不存在的正弦图
        nonexistent_sinogram = lti.load_complete_sinogram(
            sinogram_dir=os.path.join(self.test_dir, 'sinogram'),
            index=999,  # 不存在的索引
            num_events=1000
        )
        
        # 验证返回值是否为None
        self.assertIsNone(nonexistent_sinogram)

    @patch('listmode_to_incomplete.gate.listmode_to_sinogram')
    @patch('numpy.savez_compressed')
    def test_process_listmode_file(self, mock_savez, mock_listmode_to_sinogram):
        """测试处理单个listmode文件的功能"""
        # 模拟gate.listmode_to_sinogram的返回值
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = self.mock_sinogram
        mock_listmode_to_sinogram.return_value = mock_tensor
        
        # 创建临时输入文件
        input_dir = os.path.join(self.test_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)
        input_file = os.path.join(input_dir, 'listmode_data_minimal_123_1000.npz')
        np.savez_compressed(input_file, listmode=self.mock_events)
        
        # 定义缺失探测器ID
        missing_ids = {0, 1, 8, 9}
        
        # 执行处理函数（使用vis_level=0避免可视化）
        lti.process_listmode_file(
            input_file=input_file,
            output_dir=os.path.join(self.test_dir, 'output'),
            complete_sinogram_dir=os.path.join(self.test_dir, 'sinogram'),
            log_dir=os.path.join(self.test_dir, 'log'),
            missing_ids=missing_ids,
            num_events=1000,
            vis_level=0
        )
        
        # 验证savez_compressed是否被正确调用
        mock_savez.assert_called()
        
        # 验证listmode_to_sinogram是否被正确调用
        mock_listmode_to_sinogram.assert_called()
        
        # 验证输出目录是否被创建
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'output', 'sinogram_incomplete')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'output', 'listmode_incomplete')))

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_detector_coverage(self, mock_savefig):
        """测试探测器覆盖可视化功能"""
        # 模拟一组探测器ID
        missing_ids = set(range(10))
        
        # 调用可视化函数
        lti.visualize_detector_coverage(
            crystals_per_ring=self.test_info['NrCrystalsPerRing'],
            num_rings=self.test_info['NrRings'],
            missing_ids=missing_ids,
            missing_sectors=[(0, 90)],
            output_dir=self.test_dir
        )
        
        # 验证savefig是否被正确调用
        mock_savefig.assert_called_once()

class TestMainFunction(unittest.TestCase):
    """测试命令行接口和主函数"""
    
    def setUp(self):
        """创建测试环境"""
        self.test_dir = tempfile.mkdtemp()
        
        # 创建输入目录和文件
        self.input_dir = os.path.join(self.test_dir, 'input')
        os.makedirs(self.input_dir, exist_ok=True)
        
        # 创建模拟的listmode文件
        mock_events = np.array([[0, 8], [1, 9]], dtype=np.int32)
        self.input_file = os.path.join(self.input_dir, 'listmode_data_minimal_123_1000.npz')
        np.savez_compressed(self.input_file, listmode=mock_events)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir)

    @patch('listmode_to_incomplete.process_listmode_file')
    def test_main_function(self, mock_process):
        """测试主函数"""
        # 构建命令行参数
        test_args = [
            '--input_dir', self.input_dir,
            '--output_dir', os.path.join(self.test_dir, 'output'),
            '--num_events', '1000',
            '--vis_level', '0'
        ]
        
        # 使用patch模拟命令行参数
        with patch('sys.argv', ['listmode_to_incomplete.py'] + test_args):
            # 执行主函数
            lti.main()
            
            # 验证process_listmode_file被调用
            mock_process.assert_called()

if __name__ == '__main__':
    unittest.main()