import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import  torchvision.transforms.functional as functional
from torchvision.transforms import (RandomCrop, Pad, RandomHorizontalFlip,
                                    RandomVerticalFlip, Resize, ToTensor)

class Train_dataset(Dataset):
    _INPUT_ = 'input'
    _LABEL_ = 'label'
    # 训练集 pair-image 在root_folder根目录下的两个子目录下

    def __init__(self, root_folder:str, size:int):
        # size参数是最终输入到模型中学习的图像的尺寸，数据集中的图像一般会做图像增广，而不是直接使用
        super(Train_dataset, self).__init__()
        self._size = size
        self._root = root_folder
        self._all_filenames = os.listdir(os.path.join(self._root, self._INPUT_))
        # 成对的input和label图像的名字应该保持一致，因此文件名也只需要存一份list

        # UIEB集的特点在于图像的分辨率相差非常大，整合为相同或相似尺寸更有利于输入给模型
        self._resize = Resize((self._size, self._size))

    def __len__(self):
        return len(self._all_filenames)

    def __getitem__(self, index):
        # 在图像增强、恢复任务中，dataloader每次取出一对图像
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._all_filenames[index]))
        label_img = Image.open(os.path.join(self._root, self._LABEL_, self._all_filenames[index]))
        # 对图像做数据增强1.扩充了数据集提高泛化性 2.可以更全面的学习到图像的不同区域特征
        input_img_tensor, label_img_tensor = self._aug_img(input_img, label_img)
        return input_img_tensor, label_img_tensor

    def _aug_img(self, input_img, label_img):
        # padding 如果数据集图像比需求的尺寸小就要做填充
        pad_w = self._size - input_img.width  if input_img.width  < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        label_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(label_img)
        # pair img的填充方案必须保持一致 padding=(左，上，右，下)
        # random crop
        i, j, h, w = RandomCrop.get_params(input_img, output_size=(self._size, self._size))
        # get_params函数可以记录下本次随机裁剪方案，因为pair img需要做相同的操作
        # (i, j)表示裁剪的起点也就是左上角,(h, w) = output_size(h, w)表示从起点开始沿着h、w维裁剪的长度
        input_img = functional.crop(input_img, i, j, h, w)
        label_img = functional.crop(label_img, i, j, h, w)
        # random flip
        vertical_filp_probability   = random.randint(0, 1)
        horizontal_filp_probability = random.randint(0, 1)
        input_img = RandomVerticalFlip(vertical_filp_probability)(input_img)
        input_img = RandomHorizontalFlip(horizontal_filp_probability)(input_img)
        label_img = RandomVerticalFlip(vertical_filp_probability)(label_img)
        label_img = RandomHorizontalFlip(horizontal_filp_probability)(label_img)
        # 设置相同的随机数种子决定是否采取水平、垂直翻转
        # random rotate
        rand_rotate = random.randint(0, 3)
        input_img = functional.rotate(input_img, 90 * rand_rotate)
        label_img = functional.rotate(label_img, 90 * rand_rotate)
        # 设置随机旋转的角度
        # to tensor
        input_img_tensor = ToTensor()(input_img)
        label_img_tensor = ToTensor()(label_img)

        # 还有更多可选的aug img方案，比如mix up、 norm
        return input_img_tensor, label_img_tensor


class Valid_dataset(Dataset):
    _INPUT_ = 'input'
    _LABEL_ = 'label'

    # 训练集 pair-image 在root_folder根目录下的两个子目录下

    def __init__(self, root_folder: str, size: int):
        # size参数是最终输入到模型中学习的图像的尺寸，数据集中的图像一般会做图像增广，而不是直接使用
        super(Valid_dataset, self).__init__()
        self._size = size
        self._root = root_folder
        self._all_filenames = os.listdir(os.path.join(self._root, self._INPUT_))
        # 成对的input和label图像的名字应该保持一致，因此文件名也只需要存一份list

        # UIEB集的特点在于图像的分辨率相差非常大，整合为相同或相似尺寸更有利于输入给模型
        self._resize = Resize((self._size, self._size))

    def __len__(self):
        return len(self._all_filenames)

    def __getitem__(self, index):
        # 在图像增强、恢复任务中，dataloader每次取出一对图像
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._all_filenames[index]))
        label_img = Image.open(os.path.join(self._root, self._LABEL_, self._all_filenames[index]))
        # 对图像做数据增强1.扩充了数据集提高泛化性 2.可以更全面的学习到图像的不同区域特征
        input_img_tensor, label_img_tensor = self._aug_img(input_img, label_img)
        return input_img_tensor, label_img_tensor

    def _aug_img(self, input_img, label_img):
        # padding 如果数据集图像比需求的尺寸小就要做填充
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        label_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(label_img)
        # pair img的填充方案必须保持一致 padding=(左，上，右，下)
        # resize 验证集不需要做训练，因此没必要做数据增强，只是调整图片大小满足网络需求即可
        input_img = self._resize(input_img)
        label_img = self._resize(label_img)
        # to tensor
        input_img_tensor = ToTensor()(input_img)
        label_img_tensor = ToTensor()(label_img)
        return input_img_tensor, label_img_tensor


class Unlabel_dataset(Dataset):
    _INPUT_ = 'unlabel'

    def __init__(self, root_folder: str, size: int):
        # size参数是最终输入到模型中学习的图像的尺寸，数据集中的图像一般会做图像增广，而不是直接使用
        super(Unlabel_dataset, self).__init__()
        self._size = size
        self._root = root_folder
        self._all_filenames = os.listdir(os.path.join(self._root, self._INPUT_))

        # UIEB集的特点在于图像的分辨率相差非常大，整合为相同或相似尺寸更有利于输入给模型
        self._resize = Resize((self._size, self._size))

    def __len__(self):
        return len(self._all_filenames)

    def __getitem__(self, index):
        # 在图像增强、恢复任务中，dataloader每次取出一张图像
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._all_filenames[index]))
        # 对图像做数据增强1.扩充了数据集提高泛化性 2.可以更全面的学习到图像的不同区域特征
        input_img_tensor = self._aug_img(input_img)
        return input_img_tensor

    def _aug_img(self, input_img):
        # 无标签真实图像的aug，除了常规的padding resize保证图片尺寸适应网络外
        # 一般会根据具体半监督网络特点做专门的设计
        # padding 如果数据集图像比需求的尺寸小就要做填充
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        # resize
        input_img = self._resize(input_img)
        # to tensor
        input_img_tensor = ToTensor()(input_img)
        return input_img_tensor
