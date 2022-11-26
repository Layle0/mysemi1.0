import argparse
import os
from utils.util import *
import torch
# import models
# import data

class Base_options():
    # 深度学习网络的通用参数设置

    def __init__(self):
        self._init_flag = False
        # 设置了一个初始化标志位

    def initialize(self, parser):
        # 训练前的配置
        parser.add_argument('--train_dataroot', type=str,default='./data/train',help="训练集根目录")
        parser.add_argument('--valid_dataroot', type=str,default='./data/valid',help="验证集根目录")
        parser.add_argument('--unlabel_dataroot', type=str,default='./data',help="无标签根目录")
        parser.add_argument('--experiment_store_filename', type=str, default='experiment', help="某次实验存储模型参数的文件名，最终文件会叫experiment_train.txt等")
        parser.add_argument('--gpu_ids', type=str, default='0', help="gpu设置，单卡就是0，多卡是字符串'0,1,2'")
        parser.add_argument('--train_epoch', type=int, default=500, help="预期训练的最大迭代数")
        # parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help="预训练模型参数，以及训练到一半的模型的参数的存储目录")

        # 模型相关配置
        parser.add_argument('--model_name', type=str, default='semi_network', help="当前使用的网络模型类型")
        parser.add_argument('--model_input_ch',  type=int, default=3, help="该模型对应的输入通道数")
        parser.add_argument('--model_output_ch', type=int, default=3, help="该模型最终的输出通道数")

        parser.add_argument('--phase', type=str, default='train', help="当前模型")

        # 数据相关配置
        parser.add_argument('--train_batch_size', type=int, default=8, help="训练时一个批次有多少样本，主要取决于显卡、内存性能")
        parser.add_argument('--valid_batch_size', type=int, default=8, help="验证时一个批次有多少样本，主要取决于显卡、内存性能")
        parser.add_argument('--unlabel_batch_size', type=int, default=8, help="训练时一个批次有多少无标签样本，主要取决于显卡、内存性能")
        parser.add_argument('--train_imgsize', type=int, default=256, help="训练时输入网络的图像大小")
        parser.add_argument('--valid_imgsize', type=int, default=256, help="验证时输入网络的图像大小")
        parser.add_argument('--unlabel_imgsize', type=int, default=256, help="训练时输入网络的无标签图像大小")
        parser.add_argument('--num_workers', type=int, default=4, help="读取数据时采用多少个进程同时进行")

        # 训练时的配置
        # parser.add_argument('--resume_flag', type=bool, default=False, help="模型是继续训练还是从头开始训练")
        parser.add_argument('--valid_frequency', type=int, default=1, help="训练时验证指标的频率，一般来说每代都要验证")

        # 学习率相关,用于初始化优化器
        parser.add_argument('--lr_init', type=float, default=1.5e-4, help="训练模型的初始学习率")
        parser.add_argument('--lr_min', type=float, default=1e-6, help="训练时学习率衰减的最小值")
        parser.add_argument('--warmup_epochs', type=int, default=3, help="训练时学习率衰减前的预热阶段")
        parser.add_argument('--lr_decay_weight', type=float, default=1e-8, help="学习率衰减策略使用到的衰减参数")

        self._init_flag = True
        return parser

    def gather_options(self):

        if not self._init_flag:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # 如果没有初始化，做且只做一次初始化，先用ArgumentParser创建一个解析对象（且设置该对象的参数按照默认值填充）
        # 调用initalize方法，添加所有参数到对象里，这一步parser里的参数才正式填入

        opt, _ = parser.parse_known_args()
        # parse_args()方法可以解包解析对象，返回一个命名空间Namespace(xxxx:yyy)里面是若干键值对
        # 使用parse_known_args()会返回(Namespace, [])它会把该模型目前需要的参数放到Namespace中，暂时用不上的
        # 参数丢到空列表中备用，就不会存在参数不一致报错的问题

        model_name = opt.model_name
        # 这里可以根据具体模型的需要，再对parser添加一些参数



        # end

        self.parser = parser
        # 保存解析器到对象里，把需要的参数以Namespace形式返回
        return parser.parse_args()

    def print_options(self, opt):

        message = ''
        message += '----------------- Options ---------------\n'
        # 此时传入的opt其实就是parse_args()得到的Namespace，将这个类实例先用vars()函数转为一个字典
        # 变成可迭代对象，再用items()函数转为一个列表，该列表的每个元素都是一个二元元组(key, value)
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            # 利用parser自带的函数get_default查看设置这个key时的默认值是多少
            if v != default:
                comment = '\t[default: %s]' % str(default)
            # 如果参数设置的和默认值不一样就多打印一下提示信息
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            # 大小于号表示对齐方式，左对齐右对齐，数字表示所占宽度
        message += '----------------- End -------------------'
        print(message)

        # # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # # checkpoint是存放不同模型设置的参数的根目录
        # util.mkdirs(expr_dir)
        # # 为每个模型单独创建子目录
        # file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        # # phase参数在基类是没有的，需要派生类去实现，常用的阶段有 train val test等
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()

        self.print_options(opt)

        # set gpu id
        # 在默认参数中的gpu_ids是string类型，现在想设置多卡CUDA要转成int list比较好用
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:# 默认-1表示cpu，0以上表示gpu
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:# 设置CUDA的当前设备为gpu的第一张卡
            torch.cuda.set_device(opt.gpu_ids[0])
        # parse中的gpu_ids改变了，所以要更新对象中的opt
        self.opt = opt
        return self.opt

# options.py通过调用parse函数完成任务，首先调用gather_options函数设置好所需的所有参数，其中还嵌套调用了init函数
# 接着调用了print_options函数，该函数打印了所有参数的信息，并创建了子目录和指定文件名的txt文件保存参数信息
# 最后设置了gpu配置信息

