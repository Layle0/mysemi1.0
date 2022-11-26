import torch
import torch.nn as nn


class CharbonnierL1Loss(nn.Module):
    # 改良l1损失
    def __init__(self):
        super(CharbonnierL1Loss, self).__init__()
        self.eps = 1e-6

    # 设置eps的作用是防止梯度消失

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        # diff是输出图像和label的误差
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
