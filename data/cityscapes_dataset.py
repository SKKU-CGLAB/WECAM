import os
from data.dfov_dataset import DfovDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from glob import glob
import numpy as np


class CityscapesDataset(DfovDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = DfovDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--img_size', type=int, default=[1024, 2048], nargs=2, help='original image size')
        parser.set_defaults(aspect_ratio=2.0)
        parser.set_defaults(batchSize=32)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser


    def get_paths(self, opt):
        root = opt.root
        dataset_mode = opt.dataset_mode
        phase = opt.phase
        # phase = 'val' if opt.phase == 'test' else 'train'

        image_dir = os.path.join(root, dataset_mode, 'leftImg8bit_trainvaltest/leftImg8bit', phase)
        image_paths = make_dataset(image_dir, recursive=True)

        ##### for debugging #####
        if opt.debug:
            length = opt.batchSize
            if opt.phase == "train":
                length = opt.batchSize*2
            image_paths = image_paths[:length]
        ##########################

        labels_paths = glob(os.path.join(root, dataset_mode, 'labels/fov', phase, '**/*.npy'), recursive=True)

        ##### for debugging #####
        if opt.debug:
            labels_paths = labels_paths[:length]
        ##########################

        return image_paths, labels_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3])
            

    def __getitem__(self, index):
        params = get_params(self.opt, (self.opt.img_size[1], self.opt.img_size[0]))

        # Image
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # Label
        label_path = self.label_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        label = np.load(label_path, allow_pickle=True).item()
        if self.opt.crop_size != 0:
            label = self.transform_fov(image, label)

        input_dict = {
            'image': image_tensor,
            'path': image_path,
            'label': (label["fovx"], label["fovy"]),
            'label_path': label_path
        }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict