"""This module contains simple helper functions """
from __future__ import print_function
import torch
import random
import numpy as np
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


class MetricRecorder:
    # ???????????????????????????????????????
    def __init__(self):
        self.avg = .0
        self.count = 0
        self.value = .0
        self.total = .0

    def reset(self):
        self.avg = .0
        self.count = 0
        self.value = .0
        self.total = .0

    def update(self, value):
        value = round(value, 4)
        self.value = value
        self.total += value
        self.count += 1
        self.avg = round(self.total / self.count, 4)

def set_all_random_operate_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_all_dirs(opt):
    # ???????????????????????????????????????????????????
    # "./log/XXmodel/base/
    base_dir = os.path.join('./log', opt.model_name, 'base')
    if not os.path.exists(os.path.join(base_dir, 'tensorboard')):
        os.makedirs(os.path.join(base_dir, 'tensorboard'))
    if not os.path.exists(os.path.join(base_dir, 'checkpoint')):
        os.makedirs(os.path.join(base_dir, 'checkpoint'))
    if not os.path.exists(os.path.join(base_dir, 'sample_img')):
        os.makedirs(os.path.join(base_dir, 'sample_img'))

class Logger:
    # ?????????????????????????????????tensorboard??????
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    # summarywriter??????????????????????????????????????????????????????tensorboard??????
    def log_multi_scaler(self, scaler_dict, epoch):
        for key, value in scaler_dict.items():
            self.writer.add_scalar(key, value, epoch)
            # ??????????????????????????????epoch??????add_scalar????????????

def save_parameters(opt, save_name, epoch, best_metric, model, optimizer, scheduler):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        # ?????????????????????????????????????????????????????????????????????????????????????????????????????????
        'end_epoch': epoch,
        'best_ssim': best_metric['ssim']['value'],
        'ssim_epoch': best_metric['ssim']['epoch'],
        'best_psnr': best_metric['psnr']['value'],
        'psnr_epoch': best_metric['psnr']['epoch']
        # ?????????????????????????????????????????????????????????
    }, os.path.join('./log', opt.model_name, 'base', 'checkpoint', save_name + '.pth'))
    # ??????????????? ./log/XXmodel/base/checkpoint/best_ssim.pth???

def load_parameters(opt, best_metric, model, optimizer, scheduler):
    checkpoint_path = os.path.join('./log', opt.model_name, 'base/checkpoint/latest.pth')
    parameters = torch.load(checkpoint_path, map_location='cpu')
    best_metric['ssim']['value'] = parameters['best_ssim']
    best_metric['ssim']['epoch'] = parameters['ssim_epoch']
    best_metric['psnr']['value'] = parameters['best_psnr']
    best_metric['psnr']['epoch'] = parameters['psnr_epoch']
    model.load_state_dict(parameters['model'])
    optimizer.load_state_dict(parameters['optimizer'])
    scheduler.load_state_dict(parameters['scheduler'])
    return parameters['end_epoch'] + 1

def save_img(opt, input_img, label_img, output_img):
    base_dir = os.path.join('./log', opt.model_name, 'base', 'sample_img')
    input  = make_grid(input_img)
    label  = make_grid(label_img)
    output = make_grid(output_img)
    # make_grid???????????????N?????????????????????????????????????????????????????????????????????????????????????????????
    save_image(input,  os.path.join(base_dir, 'input.png'))
    save_image(label,  os.path.join(base_dir, 'label.png'))
    save_image(output, os.path.join(base_dir, 'output.png'))

def print_epoch_result(train_return, valid_result, current_epoch):
    print('Epoch[{}]:   '.format(current_epoch))
    for key, value in train_return.items():
        print(key, ':  ', value)
    if valid_result is not None:
        for key, value in valid_result.items():
            print(key, ':  ', value)