from opendataval.dataval.api import DataEvaluator, ModelMixin
import numpy as np
import torch
from torch.utils.data import Subset


class WeightedBipartiteEvaluator(DataEvaluator, ModelMixin):
    """基于加权二分图的数据估值方法
    
    通过多次随机采样建立训练样本和验证样本之间的连接。
    对于每次采样:
    1. 随机选择一定比例的训练样本
    2. 对于预测正确的验证样本,增加其与所有被采样的同类训练样本的连接权重
    3. 根据最优阈值过滤边的权重
    4. 使用贪心策略选择最大覆盖
    """

    def __init__(self, n_samples=50, sample_ratios=None, threshold_range=None, random_state=None):
        super().__init__(random_state=random_state)
        self.n_samples = n_samples
        
        # 生成n_samples个采样大小
        if sample_ratios is None:
            # 使用线性分布生成采样比例
            self.sample_ratios = np.linspace(0.01, 0.8, n_samples)
        else:
            self.sample_ratios = sample_ratios
            
        self.optimal_threshold = None
        self.validation_error = None
            
    def _get_class_labels(self, y):
        """获取类别标签"""
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y.ravel()

    def _build_weighted_edges(self, x_train, y_train, x_valid, y_valid):
        """构建加权二分图
        
        通过随机采样建立训练样本和验证样本之间的连接
        """
        n_train = len(x_train)
        n_valid = len(x_valid)
        
        # 初始化权重矩阵
        edge_weights = np.zeros((n_train, n_valid))
        
        # 获取类别标签
        train_labels = self._get_class_labels(y_train)
        valid_labels = self._get_class_labels(y_valid)
        
        # 对每个采样大小进行一次采样
        for ratio in self.sample_ratios:
            # 计算这次采样的大小
            size = max(1, int(ratio * n_train))
            
            # 随机采样训练点
            sampled_indices = self.random_state.choice(
                n_train, size=size, replace=False
            )
            
            # 训练模型
            subset_model = self.pred_model.clone()
            subset_model.fit(
                Subset(x_train, sampled_indices),
                Subset(y_train, sampled_indices)
            )
            
            # 预测验证集
            y_pred = subset_model.predict(x_valid)
            predictions = self._get_class_labels(y_pred)
            
            # 找到预测正确的验证点
            correct_valid = predictions == valid_labels
            
            # 对于每个预测正确的验证点
            for valid_idx in np.where(correct_valid)[0]:
                valid_class = valid_labels[valid_idx]
                
                # 找到所有被采样的同类训练点
                same_class_samples = sampled_indices[
                    train_labels[sampled_indices] == valid_class
                ]
                
                # 增加连接权重
                edge_weights[same_class_samples, valid_idx] += 1
                
        return edge_weights
    
    def _compute_valid_edges(self, edge_weights, threshold):
        """根据阈值过滤边,并返回二值化的邻接矩阵"""
        return (edge_weights >= threshold).astype(np.int32)


    def _compute_greedy_coverage(self, valid_edges):
        """使用贪心策略计算最大覆盖序列
        
        当没有任何训练点可以覆盖剩余验证点时,或者所有验证点都被覆盖时结束
        """
        n_train, n_valid = valid_edges.shape
        coverage_sequence = []
        remaining_edges = valid_edges.copy()
        remaining_valid = np.ones(n_valid, dtype=bool)
        max_iterations = n_train  # 防止死循环
        iterations = 0
        
        while iterations < max_iterations and remaining_valid.any():  # 只要还有未覆盖的验证点且未达到最大迭代次数
            # 计算每个训练点能覆盖多少还未覆盖的验证点
            coverage_counts = (remaining_edges & remaining_valid).sum(axis=1)
            
            # 如果没有新的覆盖或者达到最大迭代,退出循环
            if coverage_counts.max() == 0:
                break
                
            # 选择覆盖最多的点
            best_point = np.argmax(coverage_counts)
            coverage_sequence.append(best_point)
            
            # 更新剩余未覆盖的验证点
            newly_covered = remaining_edges[best_point] & remaining_valid
            remaining_valid[newly_covered] = False
            
            # 将这些验证点从其他训练点的可能覆盖中移除
            remaining_edges[:, newly_covered] = False
            
            iterations += 1
        
        # 添加剩余未选择的训练点
        remaining_points = set(range(n_train)) - set(coverage_sequence)
        remaining_list = list(remaining_points)
        self.random_state.shuffle(remaining_list)
        coverage_sequence.extend(remaining_list)
        
        return coverage_sequence
        
    
    def _find_optimal_threshold(self, edge_weights, x_train, y_train, x_valid, y_valid):
        """找到最优的边权重阈值"""
        max_weight = edge_weights.max()
        if max_weight == 0:
            return 0, float('inf')
            
        # 生成阈值范围
        thresholds = np.linspace(0, max_weight, 200)
        
        # 预先计算所有采样的验证准确率
        subset_accuracies = {}  # 存储每个采样比例的验证准确率
        for ratio in self.sample_ratios:
            # 随机采样训练点
            size = max(1, int(ratio * len(x_train)))
            indices = self.random_state.choice(len(x_train), size, replace=False)
            
            # 训练模型并计算验证准确率
            subset_model = self.pred_model.clone()
            subset_model.fit(
                Subset(x_train, indices),
                Subset(y_train, indices)
            )
            y_pred = subset_model.predict(x_valid)
            val_acc = self.evaluate(y_valid, y_pred)
            
            subset_accuracies[ratio] = {
                'indices': indices,
                'accuracy': val_acc
            }
        
        min_error = float('inf')
        best_threshold = None
        
        # 评估每个阈值
        for threshold in thresholds:
            mse = 0
            # 对每个采样比例
            for ratio in self.sample_ratios:
                # 获取预计算的准确率和索引
                val_acc = subset_accuracies[ratio]['accuracy']
                indices = subset_accuracies[ratio]['indices']
                
                # 计算有效边
                valid_edges = self._compute_valid_edges(edge_weights, threshold)
                # 只保留采样的训练点
                valid_edges_subset = valid_edges[indices]
                
                # 计算覆盖分数
                coverage = valid_edges_subset.any(axis=0).mean()
                
                mse += (coverage - val_acc) ** 2
                
            avg_mse = mse / len(self.sample_ratios)
            if avg_mse < min_error:
                min_error = avg_mse
                best_threshold = threshold
                
        return best_threshold, min_error
        

    def train_data_values(self, *args, **kwargs):
        """训练数据估值模型"""
        print(f"开始构建加权二分图 ({self.n_samples}次)...")
        
        # 构建加权二分图
        edge_weights = self._build_weighted_edges(
            self.x_train, self.y_train,
            self.x_valid, self.y_valid
        )
        
        print(f"开始寻找最优阈值 ...")
        self.optimal_threshold, self.validation_error = self._find_optimal_threshold(
            edge_weights,
            self.x_train, self.y_train,
            self.x_valid, self.y_valid
        )
        print(f"找到最优阈值: {self.optimal_threshold:.3f}, 验证MSE: {self.validation_error:.3f}")
        
        # 使用最优阈值获取有效边
        valid_edges = self._compute_valid_edges(edge_weights, self.optimal_threshold)
        
        # 使用贪心策略找到覆盖序列 
        coverage_sequence = self._compute_greedy_coverage(valid_edges)
        
        # 基于序列位置计算数据值
        n_train = len(self.x_train)
        self.data_values = np.zeros(n_train)
        for i, idx in enumerate(coverage_sequence):
            self.data_values[idx] = n_train - i
            
        return self
        
    def evaluate_data_values(self) -> np.ndarray:
        """返回归一化的数据值"""
        normalized_values = (self.data_values - self.data_values.min())
        if self.data_values.max() > self.data_values.min():
            normalized_values /= (self.data_values.max() - self.data_values.min())
        return normalized_values