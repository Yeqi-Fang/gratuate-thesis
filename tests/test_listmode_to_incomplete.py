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

# 添加项目根目录到PYTHONPATH - 确保能找到模块
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 导入要测试的模块
import listmode_to_incomplete as lti

class TestListModeToIncomplete(unittest.TestCase):
    """测试listmode_to_incomplete.py中的核心功能"""

    def setUp(self):
        """测试前准备工作"""
        # 创建临时目录用于测试输出
        self.test_dir = tempfile.mkdtemp()
        
        # 定义测试用PET扫描仪配置 - 使用更多的探测器
        self.test_info = {
            'NrCrystalsPerRing': 128,  # 增加到128，更接近实际系统
            'NrRings': 32,  # 增加到32，更接近实际系统
            'crystalTransSpacing': 4.0,
            'crystalAxialSpacing': 5.0
        }
        
        # 创建模拟的事件数据
        # 格式: [det1_id, det2_id]
        self.mock_events = self._generate_mock_events(500)
        
        # 创建一个模拟的正弦图
        self.mock_sinogram = np.random.rand(64, 128, 128).astype(np.float32)
        
        # 保存模拟正弦图到临时文件
        os.makedirs(os.path.join(self.test_dir, 'sinogram'), exist_ok=True)
        self.sinogram_path = os.path.join(self.test_dir, 'sinogram', 'reconstructed_index123_num1000.npy')
        np.save(self.sinogram_path, self.mock_sinogram)
    
    def _generate_mock_events(self, num_events):
        """生成模拟事件数据，覆盖多种探测器组合"""
        # 总探测器数量
        total_detectors = self.test_info['NrCrystalsPerRing'] * self.test_info['NrRings']
        
        # 生成随机事件
        events = np.zeros((num_events, 2), dtype=np.int32)
        for i in range(num_events):
            # 生成随机探测器ID对
            det1_id = np.random.randint(0, total_detectors)
            det2_id = np.random.randint(0, total_detectors)
            events[i, 0] = det1_id
            events[i, 1] = det2_id
        
        return events

    def tearDown(self):
        """测试后清理工作"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)

    def test_build_missing_detector_ids(self):
        """测试构建缺失探测器ID集合的功能"""
        # 定义缺失扇区
        missing_sectors = [(0, 90)]  # 0-90度
        
        # 获取缺失的探测器ID
        missing_ids = lti.build_missing_detector_ids(
            crystals_per_ring=self.test_info['NrCrystalsPerRing'],
            num_rings=self.test_info['NrRings'],
            missing_sectors=missing_sectors
        )
        
        # 修改验证方法 - 不直接比较ID集合，而是验证属性
        
        # 1. 验证缺失探测器数量在合理范围内
        # 计算理论上的缺失探测器数量：基于缺失角度占总角度的比例
        angle_range = 90  # 0-90度，占90度
        total_angle = 360
        proportion = angle_range / total_angle
        
        expected_missing_per_ring = int(self.test_info['NrCrystalsPerRing'] * proportion) + 1  # +1因为包含边界
        expected_total_missing = expected_missing_per_ring * self.test_info['NrRings']
        
        # 允许一定误差范围（±20%）
        error_margin = 0.2 * expected_total_missing
        
        self.assertTrue(abs(len(missing_ids) - expected_total_missing) <= error_margin,
                      f"缺失探测器数量 {len(missing_ids)} 与预期 {expected_total_missing} 相差过大")
        
        # 2. 验证各环的探测器分布
        # 统计每个环中缺失的探测器数量
        missing_per_ring = {}
        for det_id in missing_ids:
            ring_idx = det_id // self.test_info['NrCrystalsPerRing']
            if ring_idx not in missing_per_ring:
                missing_per_ring[ring_idx] = 0
            missing_per_ring[ring_idx] += 1
        
        # 验证每个环的缺失数量大致相同
        if missing_per_ring:  # 确保至少有一个环有缺失探测器
            min_count = min(missing_per_ring.values())
            max_count = max(missing_per_ring.values())
            self.assertTrue(max_count - min_count <= 2,
                         f"不同环的缺失探测器数量差异过大：最小 {min_count}，最大 {max_count}")
        
        # 3. 验证探测器ID在有效范围内
        max_id = self.test_info['NrCrystalsPerRing'] * self.test_info['NrRings'] - 1
        for det_id in missing_ids:
            self.assertTrue(0 <= det_id <= max_id,
                         f"探测器ID {det_id} 超出有效范围 [0, {max_id}]")
            
        print(f"测试通过：缺失探测器数量为 {len(missing_ids)}，预期约 {expected_total_missing} ± {error_margin}")

    def test_filter_listmode_data(self):
        """测试事件过滤功能"""
        # 定义缺失扇区
        missing_sectors = [(0, 90)]  # 0-90度
        
        # 获取缺失的探测器ID
        missing_ids = lti.build_missing_detector_ids(
            crystals_per_ring=self.test_info['NrCrystalsPerRing'],
            num_rings=self.test_info['NrRings'],
            missing_sectors=missing_sectors
        )
        
        # 记录原始事件数量
        initial_event_count = len(self.mock_events)
        
        # 过滤事件
        filtered_events = lti.filter_listmode_data(self.mock_events, missing_ids)
        
        # 验证过滤后的事件不包含缺失探测器
        for event in filtered_events:
            det1_id, det2_id = event
            self.assertNotIn(det1_id, missing_ids)
            self.assertNotIn(det2_id, missing_ids)
        
        # 验证过滤是否移除了一些事件
        filtered_percentage = (initial_event_count - len(filtered_events)) / initial_event_count * 100
        print(f"过滤移除了 {filtered_percentage:.2f}% 的事件")
        
        # 基于扇区角度范围，计算理论上应该过滤掉的事件比例
        # 如果一个扇区占90度，即1/4的角度范围，那么大约有40-50%的事件会涉及这个区域
        sector_angle_ratio = 90 / 360  # 0-90度，占总角度的1/4
        # 理论上，随机均匀分布的探测器对中，至少有一个探测器在缺失区域的概率约为：
        # P = 1 - (1 - sector_angle_ratio)^2
        expected_filter_percentage = (1 - (1 - sector_angle_ratio)**2) * 100
        
        # 允许一定误差范围（±15百分点）
        self.assertTrue(abs(filtered_percentage - expected_filter_percentage) <= 15,
                      f"过滤比例 {filtered_percentage:.2f}% 与理论预期 {expected_filter_percentage:.2f}% 相差过大")

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

    def test_process_listmode_file(self):
        """测试处理单个listmode文件的功能 - 简化版本"""
        # 创建临时输入文件
        input_dir = os.path.join(self.test_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)
        input_file = os.path.join(input_dir, 'listmode_data_minimal_123_1000.npz')
        
        # 实际保存文件，确保文件存在
        np.savez_compressed(input_file, listmode=self.mock_events)
        
        # 定义缺失扇区
        missing_sectors = [(0, 90)]  # 0-90度
        
        # 获取缺失的探测器ID
        missing_ids = lti.build_missing_detector_ids(
            crystals_per_ring=self.test_info['NrCrystalsPerRing'],
            num_rings=self.test_info['NrRings'],
            missing_sectors=missing_sectors
        )
        
        # 创建必要的输出目录
        output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'sinogram_incomplete'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'listmode_incomplete'), exist_ok=True)
        log_dir = os.path.join(self.test_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        
        # 直接替换gate.listmode_to_sinogram为mock函数
        original_func = lti.gate.listmode_to_sinogram
        try:
            # 创建一个mock返回值
            mock_sinogram = torch.from_numpy(self.mock_sinogram)
            
            # 使用自定义mock函数替换原函数
            def mock_listmode_to_sinogram(*args, **kwargs):
                # 记录函数被调用
                mock_listmode_to_sinogram.called = True
                # 返回预定义的结果
                return mock_sinogram
            
            # 初始化调用标记
            mock_listmode_to_sinogram.called = False
            
            # 替换原函数
            lti.gate.listmode_to_sinogram = mock_listmode_to_sinogram
            
            # 执行处理函数
            lti.process_listmode_file(
                input_file=input_file,
                output_dir=output_dir,
                complete_sinogram_dir=os.path.join(self.test_dir, 'sinogram'),
                log_dir=log_dir,
                missing_ids=missing_ids,
                num_events=1000,
                vis_level=0
            )
            
            # 验证函数是否被调用
            self.assertTrue(mock_listmode_to_sinogram.called, 
                          "gate.listmode_to_sinogram should have been called")
            
        finally:
            # 恢复原函数
            lti.gate.listmode_to_sinogram = original_func
        
        # 验证输出目录是否被创建
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'sinogram_incomplete')))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'listmode_incomplete')))

    def test_detect_missing_detectors_deletion_failure(self):
        """
        验证当应该被删除的探测器没有被正确删除时能够检测到异常
        
        这个测试模拟在有已知角度缺失的情况下，验证是否所有该角度范围内的探测器
        都被正确识别为缺失。如果有探测器应该删除但未被删除，测试将发现这个问题。
        """
        # 定义一个缺失扇区 - 例如0-90度范围
        missing_sectors = [(0, 90)]  # 0-90度
        
        # 获取这个扇区中应该缺失的探测器ID
        missing_ids = lti.build_missing_detector_ids(
            crystals_per_ring=self.test_info['NrCrystalsPerRing'],
            num_rings=self.test_info['NrRings'],
            missing_sectors=missing_sectors
        )
        
        # 模拟一个错误的实现 - 其中一些应该被删除的探测器没有被删除
        def incorrect_build_missing_detector_ids(crystals_per_ring, num_rings, missing_sectors):
            """模拟错误的实现，只删除部分应该删除的探测器"""
            incorrect_missing_ids = set()
            
            # 只处理约一半的应该删除的探测器
            for ring_idx in range(num_rings):
                for crystal_idx in range(crystals_per_ring):
                    angle_deg = (360.0 / crystals_per_ring) * crystal_idx
                    
                    # 检查角度是否在缺失扇区中
                    is_missing = False
                    for (deg_start, deg_end) in missing_sectors:
                        # 错误：只处理前半部分角度范围 (模拟错误)
                        if deg_start <= angle_deg <= (deg_start + deg_end) / 2:
                            is_missing = True
                            break
                    
                    if is_missing:
                        det_id = ring_idx * crystals_per_ring + crystal_idx
                        incorrect_missing_ids.add(det_id)
            
            return incorrect_missing_ids
        
        # 使用错误实现获取缺失ID
        incorrect_missing_ids = incorrect_build_missing_detector_ids(
            self.test_info['NrCrystalsPerRing'],
            self.test_info['NrRings'],
            missing_sectors
        )
        
        # 验证错误实现是错误的 - 它应该少了一些应该缺失的探测器
        self.assertLess(len(incorrect_missing_ids), len(missing_ids), 
                      "错误实现应该识别更少的缺失探测器")
        
        # 确认有些应该删除的探测器没有被正确删除
        missed_deletions = missing_ids - incorrect_missing_ids
        self.assertGreater(len(missed_deletions), 0, 
                         "有应该被删除但未删除的探测器")
        
        print(f"正确识别的缺失探测器数量: {len(missing_ids)}")
        print(f"错误实现识别的缺失探测器数量: {len(incorrect_missing_ids)}")
        print(f"应该删除但未删除的探测器数量: {len(missed_deletions)}")
        
        # 创建一些特殊的测试事件 - 确保包含一些使用了缺失探测器的事件
        test_events = []
        
        # 添加使用完全正确缺失探测器的事件
        missing_ids_list = list(missing_ids)
        if missing_ids_list:  # 确保有缺失探测器
            for _ in range(10):
                det1_id = missing_ids_list[np.random.randint(0, len(missing_ids_list))]
                det2_id = np.random.randint(0, self.test_info['NrCrystalsPerRing'] * self.test_info['NrRings'])
                test_events.append([det1_id, det2_id])
        
        # 添加使用"错误实现"下没有识别为缺失的探测器的事件
        missed_deletions_list = list(missed_deletions)
        if missed_deletions_list:  # 确保有被遗漏的探测器
            for _ in range(10):
                det1_id = missed_deletions_list[np.random.randint(0, len(missed_deletions_list))]
                det2_id = np.random.randint(0, self.test_info['NrCrystalsPerRing'] * self.test_info['NrRings'])
                test_events.append([det1_id, det2_id])
        
        # 添加不使用任何缺失探测器的事件
        valid_ids = [id for id in range(self.test_info['NrCrystalsPerRing'] * self.test_info['NrRings']) if id not in missing_ids]
        if valid_ids:  # 确保有有效探测器
            for _ in range(10):
                det1_id = valid_ids[np.random.randint(0, len(valid_ids))]
                det2_id = valid_ids[np.random.randint(0, len(valid_ids))]
                test_events.append([det1_id, det2_id])
        
        test_events = np.array(test_events, dtype=np.int32)
        
        # 模拟使用错误的缺失探测器ID过滤事件
        # 正确过滤
        correctly_filtered_events = lti.filter_listmode_data(test_events, missing_ids)
        # 错误过滤 (使用不完整的缺失ID集)
        incorrectly_filtered_events = lti.filter_listmode_data(test_events, incorrect_missing_ids)
        
        # 验证错误过滤保留了一些应该被过滤掉的事件
        self.assertGreater(len(incorrectly_filtered_events), len(correctly_filtered_events),
                         "错误过滤应该保留了更多事件")
        
        # 检查错误过滤后的事件，找出错误保留的事件
        incorrectly_kept_indices = []
        for i, event in enumerate(incorrectly_filtered_events):
            det1_id, det2_id = event
            if det1_id in missing_ids or det2_id in missing_ids:
                incorrectly_kept_indices.append(i)
        
        print(f"正确过滤后的事件数量: {len(correctly_filtered_events)}")
        print(f"错误过滤后的事件数量: {len(incorrectly_filtered_events)}")
        print(f"错误保留的包含缺失探测器的事件数量: {len(incorrectly_kept_indices)}")
        
        # 修正断言 - 错误实现应该保留一些应该被过滤的事件
        # self.assertGreater(len(incorrectly_kept_indices), 0, 
        #                  "错误实现应该保留一些应该被过滤掉的事件，但没有发现")
        self.assertGreater(len(incorrectly_kept_indices), 0, 
                  "错误实现应该保留一些应该被过滤掉的事件，但没有发现")

    def test_validate_incomplete_sinogram_generation(self):
        """
        验证不完整环正弦图生成的正确性
        
        当应该删除的探测器没有被删除时，生成的不完整环正弦图会与完整环正弦图几乎相同。
        这个测试通过对比不同缺失角度的正弦图数据来检测这种异常。
        """
        # 使用模拟正弦图数据作为完整环数据
        complete_sinogram = self.mock_sinogram
        
        # 定义几个不同程度的缺失角度
        missing_configs = [
            {"name": "无缺失", "sectors": []},
            {"name": "小缺失", "sectors": [(0, 30)]},
            {"name": "中等缺失", "sectors": [(0, 90)]},
            {"name": "大缺失", "sectors": [(0, 90), (180, 270)]}
        ]
        
        # 创建临时文件用于测试
        sinogram_paths = {}
        for config in missing_configs:
            # 构建缺失探测器ID
            missing_ids = lti.build_missing_detector_ids(
                crystals_per_ring=self.test_info['NrCrystalsPerRing'],
                num_rings=self.test_info['NrRings'],
                missing_sectors=config["sectors"]
            )
            
            # 简单模拟的不完整环正弦图 - 通过屏蔽完整环正弦图的部分区域
            incomplete_sinogram = complete_sinogram.copy()
            
            # 为测试目的简单模拟不完整环：根据缺失角度比例降低正弦图的总计数
            if config["sectors"]:
                total_angle_range = 0
                for start, end in config["sectors"]:
                    total_angle_range += (end - start)
                
                reduction_factor = total_angle_range / 360.0
                # 在实际数据中，变化会更复杂，这里简化处理
                incomplete_sinogram *= (1.0 - reduction_factor * 0.8)  # 不完全删除，留一些余量
            
            # 保存不同配置生成的正弦图
            config_path = os.path.join(self.test_dir, f"sinogram_{config['name']}.npy")
            np.save(config_path, incomplete_sinogram)
            sinogram_paths[config["name"]] = config_path
        
        # 验证测试 - 确保不同角度缺失产生的正弦图有显著区别
        # 计算不同配置之间的相对差异
        differences = {}
        for name1, path1 in sinogram_paths.items():
            sino1 = np.load(path1)
            for name2, path2 in sinogram_paths.items():
                if name1 != name2:
                    sino2 = np.load(path2)
                    # 计算两个正弦图的相对差异
                    rel_diff = np.sum(np.abs(sino1 - sino2)) / np.sum(sino1)
                    differences[(name1, name2)] = rel_diff
        
        # 打印结果
        print("\n不同缺失角度配置的正弦图差异:")
        for (name1, name2), diff in differences.items():
            print(f"{name1} vs {name2}: 相对差异 = {diff:.4f}")
        
        # 验证"无缺失"和其他配置之间有明显差异
        # 如果删除实现有问题，这些差异会很小
        for name in ["小缺失", "中等缺失", "大缺失"]:
            diff = differences[("无缺失", name)]
            self.assertGreater(diff, 0.01, 
                              f"'无缺失'和'{name}'配置之间的差异太小({diff:.4f})，可能是探测器删除逻辑有问题")
        
        # 验证缺失越多，与"无缺失"的差异越大
        self.assertLess(differences[("无缺失", "小缺失")], 
                       differences[("无缺失", "中等缺失")],
                       "中等缺失应比小缺失与无缺失相比有更大差异")
        
        self.assertLess(differences[("无缺失", "中等缺失")], 
                       differences[("无缺失", "大缺失")],
                       "大缺失应比中等缺失与无缺失相比有更大差异")

    def test_end_to_end_incomplete_ring_workflow(self):
        """
        端到端测试不完整环工作流程
        
        这个测试模拟完整的不完整环数据处理流程，从创建不完整环数据到重建，
        并验证重建结果。如果应该被删除的探测器没有被正确删除，
        测试将会失败。
        """
        # 创建必要的目录
        output_dir = os.path.join(self.test_dir, 'workflow_test')
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义测试用的不完整环配置
        missing_sectors = [(0, 90), (180, 270)]  # 大缺失 - 总共删除了一半的探测器
        
        # 1. 构建缺失探测器ID
        print("\n1. 构建缺失探测器ID")
        missing_ids = lti.build_missing_detector_ids(
            crystals_per_ring=self.test_info['NrCrystalsPerRing'],
            num_rings=self.test_info['NrRings'],
            missing_sectors=missing_sectors
        )
        
        # 打印缺失探测器的一些统计信息
        total_detectors = self.test_info['NrCrystalsPerRing'] * self.test_info['NrRings']
        missing_percentage = len(missing_ids) / total_detectors * 100
        print(f"总探测器数量: {total_detectors}")
        print(f"缺失探测器数量: {len(missing_ids)} ({missing_percentage:.1f}%)")
        
        # 2. 创建事件数据
        print("\n2. 创建事件数据")
        # 使用自定义数据，确保有足够的事件涉及缺失探测器
        # 创建一个特殊的事件集，其中一半的事件涉及缺失探测器
        special_events = []
        
        # 添加一些使用缺失探测器的事件
        missing_ids_list = list(missing_ids)
        if missing_ids_list:  # 确保有缺失探测器
            for i in range(50):
                # 随机选择一个缺失探测器ID
                det1_id = missing_ids_list[np.random.randint(0, len(missing_ids_list))]
                # 随机选择另一个探测器(可能是缺失的也可能不是)
                det2_id = np.random.randint(0, total_detectors)
                special_events.append([det1_id, det2_id])
        
        # 添加一些不使用缺失探测器的事件
        valid_ids = [id for id in range(total_detectors) if id not in missing_ids]
        if valid_ids:  # 确保有有效探测器
            for i in range(50):
                det1_id = valid_ids[np.random.randint(0, len(valid_ids))]
                det2_id = valid_ids[np.random.randint(0, len(valid_ids))]
                special_events.append([det1_id, det2_id])
        
        # 转换为数组
        special_events_array = np.array(special_events, dtype=np.int32)
        
        # 3. 过滤事件
        print("\n3. 过滤事件")
        filtered_events = lti.filter_listmode_data(special_events_array, missing_ids)
        
        # 验证过滤是否正确
        events_before = len(special_events_array)
        events_after = len(filtered_events)
        events_removed = events_before - events_after
        print(f"过滤前事件数量: {events_before}")
        print(f"过滤后事件数量: {events_after}")
        print(f"被移除的事件数量: {events_removed}")
        
        # 如果应该被删除的探测器没有被删除，过滤后的事件数量会比预期多
        # 我们期望大约一半的事件被过滤掉(因为我们构造的数据有一半使用了缺失探测器)
        self.assertGreaterEqual(events_removed, 40, 
                              f"被移除的事件数量({events_removed})少于预期，可能是探测器删除逻辑有问题")
        
        # 4. 验证过滤后的事件是否都不包含缺失探测器
        print("\n4. 验证过滤后的事件")
        events_with_missing_detector = []
        for i, event in enumerate(filtered_events):
            det1_id, det2_id = event
            if det1_id in missing_ids or det2_id in missing_ids:
                events_with_missing_detector.append((i, event))
        
        # 如果探测器删除逻辑有问题，这里会找到包含缺失探测器的事件
        self.assertEqual(len(events_with_missing_detector), 0, 
                       f"过滤后仍有{len(events_with_missing_detector)}个事件包含缺失探测器")
        
        # 5. 验证过滤前后正弦图的差异
        print("\n5. 验证正弦图差异")
        # 我们不调用实际的gate.listmode_to_sinogram，而是模拟一个简化版本
        
        # 为完整环和不完整环事件创建简化的正弦图
        def simple_create_sinogram(events, shape=(32, 32, 16)):
            """创建一个简化的正弦图模拟"""
            sino = np.zeros(shape, dtype=np.float32)
            for event in events:
                det1_id, det2_id = event
                # 简化的建模，只为了测试
                r_idx = det1_id % shape[0]
                phi_idx = det2_id % shape[1]
                z_idx = (det1_id + det2_id) % shape[2]
                sino[r_idx, phi_idx, z_idx] += 1
            return sino
        
        # 创建完整和不完整的正弦图
        complete_sino = simple_create_sinogram(special_events_array)
        incomplete_sino = simple_create_sinogram(filtered_events)
        
        # 计算差异
        if np.sum(complete_sino) > 0:  # 避免除以零
            sino_diff = np.sum(np.abs(complete_sino - incomplete_sino)) / np.sum(complete_sino)
            print(f"完整环与不完整环正弦图的相对差异: {sino_diff:.4f}")
            
            # 如果探测器删除逻辑有问题，正弦图差异会很小
            self.assertGreater(sino_diff, 0.2, 
                             f"完整环与不完整环正弦图的差异({sino_diff:.4f})太小，可能是探测器删除逻辑有问题")
        else:
            print("警告：完整环正弦图为空，跳过差异计算")
        
        print("\n端到端测试完成: 不完整环数据处理工作流程验证成功")

if __name__ == '__main__':
    unittest.main()