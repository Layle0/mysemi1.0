import os
from tqdm import tqdm # 进度条库
from torch import no_grad
from kornia.metrics import ssim, psnr # 使用kornia库中的指标计算函数
from utils.util import *

from loss.perceptual import PerceptualLoss
from loss.ssim import SSIMLoss
from loss.charbonnier import CharbonnierL1Loss
# 使用自定义的损失函数
from torch.nn import L1Loss
# 使用库自带的损失函数

from PIL import Image
# 如果还要用无标签图片需要再引入unlabel_dataloader
def train_one_epoch(train_dataloader, unlabel_dataloader, model, optimizer):
    model.train()
    loss_recorder = MetricRecorder()
    # 设置用于存储loss的对象
    Perp_loss = PerceptualLoss().cuda()
    Ssim_loss = SSIMLoss().cuda()
    CbL1_loss = CharbonnierL1Loss().cuda()
    # 将本模型所需的自定义损失函数导入，并使用cuda加速
    unlabel_iter = iter(unlabel_dataloader)
    criterion = L1Loss().to('cuda')
    for batch in tqdm(train_dataloader, ncols=50):
        # tqdm需要传入一个可迭代对象，这里使用dataloader, ncols控制进度条的长度
        input_img, label_img = batch
        unlabel_img = next(unlabel_iter)
        input_img   = input_img.to('cuda')
        label_img   = label_img.to('cuda')
        unlabel_img = unlabel_img.to('cuda')
        # 图片数量不匹配是一个问题,目前设置unlabel 和label都是400张

        # with autocast()半精度加速,如果损失函数是自定义的就不适合用
        output_img = model(input_img)
        Total_loss = criterion(output_img, input_img)
        #
        optimizer.zero_grad()
        # 优化器每次使用前需要清空梯度
        Total_loss.backward()
        optimizer.step()
        # 反向传播计算梯度，优化器根据梯度更新参数
        loss_recorder.update(Total_loss.item())
    return {'train_loss': loss_recorder.avg,
            'lr': optimizer.param_groups[0]['lr']}
    # optimizer里面的参数分成'params' 'lr' 'weight_decay' 等几个大组

def train(opt, logger, train_dataloader, unlabel_dataloader, valid_dataloader, model, optimizer, scheduler):
    set_all_random_operate_seed()
    # 随机种子默认值为42
    make_all_dirs(opt)
    # 为该模型创建存储信息的文件夹，包括显示指标走势的张量板，记录当前参数的检查点，测试图片
    best_metric = {'ssim': {'value': .0, 'epoch': 0},
                   'psnr': {'value': .0, 'epoch': 0}}
    # 记录全局最佳指标
    if input('Whether to resume to training: ') == 'yes':
        print('------continue training------')
        # 如果继续训练，就需要从检查点拿到之前的参数
        start_epoch = load_parameters(opt, best_metric, model, optimizer, scheduler)
        # load_para是save的逆过程，读取latest.pth内的信息，更新模型、优化器参数、更新最优指标
    else:
        print('------start    training------')
        start_epoch = 1
    print('start form {}th epoch'.format(start_epoch))
    max_epoch = opt.train_epoch + 1
    for current_epoch in range(start_epoch, max_epoch):
        train_return = train_one_epoch(train_dataloader, unlabel_dataloader, model, optimizer)
        # return返回一个字典，存储最关键的loss 和 lr信息，记录到张量板上
        if not opt.resume_flag:
            opt.resume_flag = True
        logger.log_multi_scaler(train_return, current_epoch)
        scheduler.step(current_epoch)
        # 训练完一个epoch，要把结果存入张量板目录下，用优化器更新参数和学习率
        # 接着使用验证集查看当前指标如何
        valid_result = None
        if current_epoch % opt.valid_frequency == 0:
            # 每隔valid_frequency代验证一次指标
            valid_result = valid(opt, model, valid_dataloader)

            logger.log_multi_scaler(valid_result, current_epoch)
            # 在张量板中同时比较训练集、验证集loss可以查看过拟合情况
            if valid_result['ssim'] > best_metric['ssim']['value']:
                best_metric['ssim']['value'] = valid_result['ssim']
                best_metric['ssim']['epoch'] = current_epoch
                save_parameters(opt, 'best_ssim', current_epoch, best_metric, model, optimizer, scheduler)
            if valid_result['psnr'] > best_metric['psnr']['value']:
                best_metric['psnr']['value'] = valid_result['psnr']
                best_metric['psnr']['epoch'] = current_epoch
                save_parameters(opt, 'best_psnr', current_epoch, best_metric, model, optimizer, scheduler)
        save_parameters(opt, 'latest', current_epoch, best_metric, model, optimizer, scheduler)
        # 由于学习的过程会产生波动，因此存储了最优ssim、最优psnr、最近一个epoch三类信息
        print_epoch_result(train_return, valid_result, current_epoch)
        print('best ssim: ', best_metric['ssim']['value'], '  best epoch: ', best_metric['ssim']['epoch'])
        print('best psnr: ', best_metric['psnr']['value'], '  best epoch: ', best_metric['psnr']['epoch'])
        # 打印本epoch的所有关注的参数和指标


def valid(opt, model, valid_dataloader):
    model.eval() # 验证时模型不更新权重参数
    loss_recorder = MetricRecorder()
    ssim_recorder = MetricRecorder()
    psnr_recorder = MetricRecorder()
    #
    criterion = L1Loss().to('cuda')
    #
    for i, batch in enumerate(valid_dataloader):
        input_img, label_img = batch
        input_img = input_img.to('cuda')
        label_img = label_img.to('cuda')
        with torch.no_grad(): # 验证时模型不自动计算梯度
            output_img = model(input_img)

        # 验证集采用什么损失函数没有定论，简单的使用库中的L1loss效果一般就很不错
        output_img = output_img.clamp(0, 1)
        Total_loss = criterion(output_img, label_img)
        #

        loss_recorder.update(Total_loss.item())
        ssim_recorder.update(ssim(output_img, label_img, 5).mean().item())
        psnr_recorder.update(psnr(output_img, label_img, 1).item())
        # 使用kornia库中的指标函数，ssim中的5是高斯核的大小，也即窗口大小window_size
        # psnr中的1是max_value，也就是该指标的最大值，默认设置为1
        if i % 10 == 0:
            save_img(opt, input_img, label_img, output_img)
        # 将验证的图片和其结果存储
    return {'valid_loss': loss_recorder.avg, 'ssim': ssim_recorder.avg,
            'psnr': psnr_recorder.avg}