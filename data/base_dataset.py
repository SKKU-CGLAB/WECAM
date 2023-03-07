import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


def get_params(opt, size=(0, 0)):
    w, h = size
    new_h = h
    new_w = w
    flip = random.random() > 0.5
    # b_crop = random.random() > 0.5

    if opt.preprocess_mode == 'resize_and_crop':
        new_h, new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size[1]
        new_h = opt.load_size[1] * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size[1] * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)
    elif opt.preprocess_mode == 'keep_aspect':
        crop_pos = (0, 0)
        opt.crop_size = 0

        # if b_crop and opt.data_aug:
        if opt.data_aug:
            opt.crop_size = (random.randint(2, new_w-2), random.randint(2, new_h-2))
            crop_pos = (new_w-opt.crop_size[0])//2, (new_h-opt.crop_size[1])//2
            opt.crop_size = new_w-(2*crop_pos[0]), new_h-(2*crop_pos[1])

        return {'crop_pos': crop_pos, 'flip': flip}

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'resize' in opt.preprocess_mode:
        transform_list.append(transforms.Resize(opt.load_size, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size[1], method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size[1], method)))
    elif 'keep_aspect' in opt.preprocess_mode:
        if params['crop_pos'] != (0, 0):
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
            transform_list.append(transforms.Resize(opt.img_size, interpolation=method))

        transform_list.append(transforms.Lambda(lambda img: __resize_image_with_aspect_ratio(img, [opt.load_size[1], opt.load_size[0]], method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [__normalize()]

    return transforms.Compose(transform_list)


def __normalize():
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    x1, y1 = pos
    if isinstance(size, tuple):
        tw, th = size
    else:
        tw = th = size

    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


###############################################################################
# Code from
# https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
# Resize image with padding to keep its aspect ratio
###############################################################################
def __resize_image_with_aspect_ratio(img, size, method=Image.BICUBIC):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(max(size)) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    img = img.resize(new_size, method)
    # create a new image and paste the resized on it
    n_ch = np.asarray(img).shape[2] if len(np.asarray(img).shape) > 2 else 1
    mode = "RGB" if n_ch == 3 else "L"
    new_im = Image.new(mode, (size[0], size[1]))
    new_im.paste(img, ((size[0] - new_size[0]) // 2,
                      (size[1] - new_size[1]) // 2))

    # new_im.show()
    return new_im
