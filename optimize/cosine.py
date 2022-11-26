""" Cosine Scheduler
Cosine schedule with warmup.
Copyright 2021 Ross Wightman
"""
import math
import torch

from timm.scheduler.scheduler import Scheduler
# 超参数的优化方案，动态调整学习率的算法，统称为 LRScheduler 学习率调度器


class CosineScheduler(Scheduler):
    """
    Cosine decay with warmup.
    This is described in the paper https://arxiv.org/abs/1608.03983.
    Modified from timm's implementation.
    """
    # 由于刚开始训练时，模型的权重(weights)
    # 是随机初始化的，此时若选择一个较
    # 大的学习率，可能带来模型的不稳定（振荡），选择Warmup预热学习率的方
    # 式，可以使得开始训练的几个epoch或者一些step内学习率较小，在预热的小学
    # 习率下，模型可以慢慢趋于稳定，等模型相对稳定后在选择预先设置的学习率进
    # 行训练，使得模型收敛速度变得更快，模型效果更佳。

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_name: str,
                 t_max: int,
                 value_min: float = 0.,
                 warmup_t=0,
                 const_t=0,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field=param_name, initialize=initialize)
        # globalstep 当前执行了多少步，一步表示做完了一个epoch的一个batch
        # 因此有totalstep = epoch * N / batchsize 也有warmupstep
        # warmup_learning_rate: 这是warm up阶段线性增长的初始值
        # warmup_steps: warm_up总的需要持续的步数
        # learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降
        # const_t 预热如果提前达到了lrbase，也不开始下降，必须走完要求的步数
        assert t_max > 0
        assert value_min >= 0
        assert warmup_t >= 0
        assert const_t >= 0

        self.cosine_t = t_max - warmup_t - const_t
        self.value_min = value_min
        self.warmup_t = warmup_t
        self.const_t = const_t

        if self.warmup_t:
            self.warmup_steps = [(v - value_min) / self.warmup_t for v in self.base_values]
            super().update_groups(self.value_min)
        else:
            self.warmup_steps = []

    def _get_value(self, t):
        if t < self.warmup_t:
            values = [self.value_min + t * s for s in self.warmup_steps]
        elif t < self.warmup_t + self.const_t:
            values = self.base_values
        else:
            t = t - self.warmup_t - self.const_t

            value_max_values = [v for v in self.base_values]

            values = [
                self.value_min + 0.5 * (value_max - self.value_min) * (1 + math.cos(math.pi * t / self.cosine_t))
                for value_max in value_max_values
            ]

        return values

    def get_epoch_values(self, epoch: int):
        return self._get_value(epoch)