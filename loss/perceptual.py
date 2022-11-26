import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16, VGG16_Weights

# 感知损失不拘泥于像素级差异，而是学习语义差异，风格差异，因此需要获得高级特征
# 高级特征恰恰是卷积神经网络提取出来的，我们选取卷积网络中的几个中间层，把他们提取出的特征图
# 放到某个损失函数中去计算
# --- Perceptual loss network  --- #
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # vgg_model = vgg16(pretrained=True).features[:16]

        # VGG16网络是固定参数的，也就是参数不更新，于是下面把该模型的所有parameters的requires_grad自动求导关掉了
        # 有时候也会使用vgg_model.eval()冻结权重
        # 为什么不更新参数？因为我们拿VGG网络去指导生成网络Fw修改参数，使得它生成的图案的高级特征与VGG中提取的风格特征
        # 内容特征等高级特征计算得到的损失最小，真正在做学习的是生成网络
        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        vgg_model = vgg_model.cuda()
        # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg_model
        # 我们只需要获取某几层输出作为高级特征来算损失，不是使用所有的特征，因为还有很多低级特征
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        # 将VGG16个模型组件按顺序执行，也就是做前向传播
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            # 如果执行到了想提取高级特征的块，这里用编号表示，想提取3 8 15号模型块的输出
            if name in self.layer_name_mapping:
                # 把这些层输出的高级特征保存下来并且返回为list
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        # 生成网络正向传播生成输出图
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            # 计算生成图案的高级特征，与VGG提取的高级特征做均方差当损失，均方差函数在functional里
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)
