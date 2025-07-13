import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActivationMonitor:
    """
    监控神经网络的激活状态，实现归一化和覆盖率计算
    """
    def __init__(self, model, threshold=0.2, eps=1e-6):
        """
        Args:
            model: 要监控的模型
            threshold: 激活阈值τ
            eps: 数值稳定性小量ε
        """
        self.model = model
        self.threshold = threshold
        self.eps = eps
        self.activation_stats = {}
        
        # 注册hook来捕获各层激活
        self._register_hooks()
    
    def _register_hooks(self):
        """为卷积层和全连接层注册前向hook"""
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.register_forward_hook(self._get_activation_hook(name))
    
    def _get_activation_hook(self, name):
        """返回用于捕获激活的hook函数"""
        def hook(module, input, output):
            # 卷积层处理: 全局平均池化空间维度
            if isinstance(module, nn.Conv2d):
                # output shape: [B, C, H, W]
                activated = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            else:  # 线性层
                activated = output  # [B, D]
            
            # 归一化激活值
            batch_max, _ = activated.max(dim=0, keepdim=True)  # 每神经元在批次中的最大值
            batch_min, _ = activated.min(dim=0, keepdim=True)  # 每神经元在批次中的最小值
            normalized = (activated - batch_min) / (batch_max -  batch_min + self.eps)  # 公式(1)
            
            # 计算当前批次的覆盖率
            coverage = (normalized > self.threshold).float().mean() # 公式(2)
            
            # 保存统计信息
            self.activation_stats[name] = {
                'normalized': normalized.detach(),
                'coverage': coverage
            }
        return hook
    
    def get_coverage(self, layer_name=None):
        """获取指定层或所有层的覆盖率"""
        if layer_name:
            return self.activation_stats.get(layer_name, {}).get('coverage', 0.0)
        return {name: stats['coverage'].item() for name, stats in self.activation_stats.items()}
    
    def get_normalized_activations(self, layer_name):
        """获取指定层的归一化激活值"""
        return self.activation_stats.get(layer_name, {}).get('normalized', None)

class PerturbationGenerator(nn.Module):
    """
    扰动生成器G，生成对抗扰动
    """
    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape
        
        # 简单的深度卷积网络结构
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, c, 3, padding=1),
            nn.Tanh()  # 输出在[-1,1]范围内
        )
        
        # 初始化小权重，使初始扰动较小
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 生成扰动，限制其幅度
        return self.net(x) * 0.1  # 限制扰动大小

def adversarial_loss(
    perturbed_images, 
    original_images, 
    target_model, 
    activation_monitor, 
    alpha=1.0, 
    beta=0.1,
    threshold=0.5  # 新增阈值参数，默认为0.5
):
    """
    计算对抗损失，修正后的低激活惩罚项基于 max(threshold - normalized_activations, 0)
    
    Args:
        perturbed_images: 扰动后的图像 [B, C, H, W]
        original_images: 原始图像 [B, C, H, W]
        target_model: 目标模型（被攻击的模型）
        activation_monitor: 激活监控器（ActivationMonitor 实例）
        alpha: 原始任务损失权重
        beta: 低激活惩罚项权重
        threshold: 激活阈值（默认为0.5）
    """
    # 1. 原始任务损失（分类误差）
    # logits = target_model(perturbed_images)
    # task_loss = F.cross_entropy(logits, target_model(original_images).argmax(dim=1))
    
    # 2. 修正后的低激活惩罚项
    penalty_loss = 0.0
    for layer_name, stats in activation_monitor.activation_stats.items():
        normalized_activations = stats['normalized']  # 形状 [B, N]，值范围 [0, 1]
        
        # 计算 max(threshold - normalized_activations, 0)
        penalty = torch.clamp(threshold - normalized_activations, min=0)  # [B, N]
        penalty = penalty.mean()  # 对所有神经元和批次取平均
        
        penalty_loss += penalty
    
    # 3. 总损失
    # total_loss = alpha * task_loss + beta * penalty_loss
    return penalty_loss
    
    
# 示例使用
if __name__ == "__main__":
    # 1. 创建目标模型（假设是一个分类CNN）
    class TargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32*8*8, 10)  # 假设输入是32x32，经过两次下采样为8x8
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    target_model = TargetModel()
    
    # 2. 创建激活监控器
    activation_monitor = ActivationMonitor(target_model, threshold=0.2)
    
    # 3. 创建扰动生成器
    perturb_generator = PerturbationGenerator(input_shape=(3, 32, 32))
    
    # 4. 模拟训练过程
    optimizer = torch.optim.Adam(perturb_generator.parameters(), lr=0.001)
    
    # 模拟输入数据 (batch_size=16, 3通道, 32x32)
    original_images = torch.rand(16, 3, 32, 32)
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # 生成扰动
        perturbations = perturb_generator(original_images)
        
        # 应用扰动 (限制在有效像素范围内)
        perturbed_images = torch.clamp(original_images + perturbations, 0, 1)
        
        # 前向传播目标模型以获取激活
        _ = target_model(perturbed_images)
        
        # 计算损失
        loss, loss_dict = adversarial_loss(
            perturbed_images, 
            original_images, 
            target_model, 
            activation_monitor
        )
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 打印训练信息
        print(f"Epoch {epoch+1}:")
        print(f"  Total Loss: {loss.item():.4f}")
        print(f"  Task Loss: {loss_dict['task_loss'].item():.4f}")
        print(f"  Penalty Loss: {loss_dict['penalty_loss'].item():.4f}")
        
        # 打印各层覆盖率
        for layer_name, coverage in activation_monitor.get_coverage().items():
            print(f"  {layer_name} Coverage: {coverage.item():.4f}")