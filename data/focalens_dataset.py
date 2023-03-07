from collections import defaultdict
import os
from data.dfov_dataset import DfovDataset, get_params, get_transform
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms
import data.cityscapes_labels as cityscapes_labels
import torch

class FocalensDataset(DfovDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = DfovDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--img_size', type=int, default=[320, 480], nargs=2, help='original image size hxw')
        parser.add_argument('--subset', type=str, default="city", help="subset of focalens datset")
        parser.set_defaults(aspect_ratio=480/320)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser


    def initialize(self, opt):
        self.opt = opt

        image_paths, labels, edge_paths, semantic_paths  = self.get_paths(opt)

        image_paths = image_paths[:opt.max_dataset_size]
        labels = labels[:opt.max_dataset_size]
        edge_paths = edge_paths[:opt.max_dataset_size]
        semantic_paths = semantic_paths[:opt.max_dataset_size]

        self.image_paths = image_paths
        self.labels = labels
        self.edge_paths = edge_paths
        self.semantic_paths = semantic_paths

        size = len(self.labels)
        self.dataset_size = size
    

    def get_paths(self, opt):
        root = opt.root
        dataset_mode = opt.dataset_mode
        subset = opt.subset
        phase = opt.phase
        # phase = 'val' if opt.phase == 'test' else 'train'

        dataset_root = os.path.join(root, dataset_mode)
        index_txt = glob(os.path.join(dataset_root, "xy", f"{subset}_{phase}*"))
        image_paths = []
        labels = []
        edge_paths = []
        semantic_paths = []

        with open(index_txt[0], "r") as f:
            lines = f.readlines()

        for line in lines:
            img_name, fovx, fovy = line.split()
            image_paths.append(os.path.join(dataset_root, "images", f"{img_name}.jpg"))
            if opt.network in "edgeattention":
                edge_paths.append(os.path.join(dataset_root, "edge", f"{img_name}.jpg"))
                semantic_paths.append(os.path.join(dataset_root, "semantic", f"{img_name}.png"))

            labels.append({"fovx": fovx, "fovy": fovy})


        ##### for debugging #####
        if opt.debug:
            length = opt.batchSize
            if opt.phase == "train":
                length = opt.batchSize*10
            image_paths = image_paths[:length]
        ##########################

        ##### for debugging #####
        if opt.debug:
            labels = labels[:length]
        ##########################

        return image_paths, labels, edge_paths, semantic_paths


    def __getitem__(self, index):
        params = get_params(self.opt, (self.opt.img_size[1], self.opt.img_size[0]))

        # Image
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # Label
        label = self.labels[index]

        if self.opt.crop_size != 0:
            label = self.transform_fov(image, label)

        input_dict = {
            'image': image_tensor,
            'path': image_path,
            'label': (float(label["fovx"]), float(label["fovy"])),
        }
        
        # edgeattention network needs edge and semantic
        if self.opt.network in "edgeattention":
            transform_image = get_transform(self.opt, params, Image.NEAREST, normalize=False)
            edge_path = self.edge_paths[index]
            edge = Image.open(edge_path)
            edge = edge.convert('RGB')
            edge_tensor = transform_image(edge)
            input_dict["edge"] = edge_tensor

            semantic_path = self.semantic_paths[index]
            semantic = Image.open(semantic_path)
            semantic = semantic.convert('RGB')
            semantic_tensor = transform_image(semantic)
            input_dict["semantic"] = semantic_tensor
            
            weight_map = torch.zeros((1, semantic_tensor.shape[1], semantic_tensor.shape[2]), dtype=torch.float32)
            i_semantic_tensor = np.transpose((semantic_tensor*255).type(torch.uint8).numpy(), (1,2,0))
                                            
            for label in cityscapes_labels.labels:
                if label.category == "void":
                    continue
                indices = np.where(np.all(i_semantic_tensor == label.color, axis=2))
                coords = list(zip(indices[0], indices[1]))
                for coord in coords:
                   weight_map[0, coord[0], coord[1]] = label.weight
                   
            input_dict["weight_map"] = weight_map


        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
    

    def postprocess(self, input_dict):
        #radian to degree
        input_dict["label"] = ( input_dict["label"][0]*180/np.pi, input_dict["label"][1]*180/np.pi )

        return input_dict