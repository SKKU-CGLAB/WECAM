import numpy as np

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os


class DfovDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        image_paths, label_paths = self.get_paths(opt)

        util.natural_sort(image_paths)
        util.natural_sort(label_paths)

        image_paths = image_paths[:opt.max_dataset_size]
        label_paths = label_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(image_paths, label_paths):
                self.paths_match(path1, path2), \
                "The image-label pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/dfov_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.image_paths = image_paths
        self.label_paths = label_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        image_paths = []
        label_paths = []
        assert False, "A subclass of DfovDataset must override self.get_paths(self, opt)"
        return image_paths, label_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        assert False, "A subclass of DfovDataset must override self.__getitem__(self, opt)"
        return input_dict

    def transform_fov(self, img, label):
        w, h = img.size
        crop_w, crop_h = self.opt.crop_size

        fovx, fovy = label["fovx"] * np.pi/180, label["fovy"] * np.pi/180 # to radian
        new_fovx = np.arctan(crop_w / w * np.tan(fovx / 2)) * 2
        new_fovy = np.arctan(crop_h / h * np.tan(fovy / 2)) * 2
        label["fovx"] = round(new_fovx * 180 / np.pi) # to degree
        label["fovy"] = round(new_fovy * 180 / np.pi) # to degree

        if self.opt.debug:
            print(f"img_size: {img.size}=>{self.opt.crop_size}, fov: ({round(fovx * 180 / np.pi)}, {round(fovy * 180 / np.pi)})=>({label['fovx']}, {label['fovy']})")

        return label

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
