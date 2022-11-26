import os
from datasets.UIEB_dataset import Train_dataset, Valid_dataset, Unlabel_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW # 一种优化器采取的策略Adam
from optimize.cosine import CosineScheduler # 一种自定义的学习率衰减策略

from utils.util import *
from train import train


from models.shallow import UWnet
from options.Base_options import Base_options

def configurate_optimizer_and_scheduler(model, opt):
    optimizer = AdamW(
        params=model.parameters(),
        lr=opt.lr_init,
        weight_decay=opt.lr_decay_weight
    )
    # 对模型的所有可学习参数以及学习率初始化优化器
    scheduler = CosineScheduler(
        optimizer=optimizer,
        param_name='lr',
        t_max=opt.train_epoch + 10,
        value_min=opt.lr_min,
        warmup_t=opt.warmup_epochs,
        const_t=0
    )
    # 针对学习率设置衰减策略, t_max应该稍微比总epoch大一点
    return optimizer, scheduler

def configurate_dataloader(opt):
    train_dataset = Train_dataset(
        root_folder=opt.train_dataroot,
        size=opt.train_imgsize
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    ) # pin memory表示锁页内存，也就是该页面不会做交换页面操作，一定程度上提升速度
    valid_dataset = Valid_dataset(
        root_folder=opt.valid_dataroot,
        size=opt.valid_imgsize
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.valid_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    unlabel_dataset = Unlabel_dataset(
        root_folder=opt.unlabel_dataroot,
        size=opt.unlabel_imgsize
    )
    unlabel_dataloader = DataLoader(
        dataset=unlabel_dataset,
        batch_size=opt.unlabel_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    return train_dataloader, valid_dataloader, unlabel_dataloader

if __name__ == '__main__':
    opt = Base_options().parse()
    base_dir = os.path.join('./log', opt.model_name, 'base')
    model = UWnet().cuda()
    print(model)
    logger = Logger(os.path.join(base_dir, 'tensorboard'))
    optimizer, scheduler = configurate_optimizer_and_scheduler(model, opt)
    print("=========finish   initialize=========\n")

    train_dataloader, valid_dataloader, unlabel_dataloader = configurate_dataloader(opt)
    train(opt, logger, train_dataloader, unlabel_dataloader, valid_dataloader, model, optimizer, scheduler)