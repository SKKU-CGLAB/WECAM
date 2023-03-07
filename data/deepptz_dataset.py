from collections import defaultdict
import os
from data.dfov_dataset import DfovDataset, get_params, get_transform
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms
import data.cityscapes_labels as cityscapes_labels
import torch
from scipy.stats import norm
import numpy as np
import math
import uuid

class DeepPTZDataset(DfovDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = DfovDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--img_size', type=int, default=[299, 299], nargs=2, help='original image size hxw')
        parser.add_argument('--subset', type=str, default="city", help="subset of focalens datset")
        parser.set_defaults(aspect_ratio=299/299)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser


    def initialize(self, opt):
        self.opt = opt

        image_paths, labels, edge_paths, semantic_paths, b_cached, cache_names = self.get_paths(opt)

        image_paths = image_paths[:opt.max_dataset_size]
        labels = labels[:opt.max_dataset_size]
        edge_paths = edge_paths[:opt.max_dataset_size]
        semantic_paths = semantic_paths[:opt.max_dataset_size]

        self.image_paths = image_paths
        self.labels = labels
        self.edge_paths = edge_paths
        self.semantic_paths = semantic_paths
        self.b_cached = b_cached
        self.cache_names = cache_names
        self.cache_name = str(uuid.uuid4())
        if self.opt.phase != 'test':
            os.makedirs(os.path.join(self.opt.cache_root, self.cache_name), exist_ok=True)

        size = len(self.labels)
        self.dataset_size = size
    

    def get_paths(self, opt):
        root = opt.root
        dataset_mode = opt.dataset_name if opt.dataset_name != 'none' else opt.dataset_mode
        subset = opt.subset
        phase = opt.phase
        #phase = 'train' if opt.phase == 'train' else 'val'

        dataset_root = os.path.join(root, dataset_mode)
        index_txt = os.path.join(dataset_root, phase, f"{phase}_sample.txt") if opt.sample else os.path.join(dataset_root, phase, f"{phase}.txt")
        image_paths = []
        labels = []
        edge_paths = []
        semantic_paths = []
        b_cached = []
        cache_names = []
        net_h, net_w = opt.load_size
        img_h, img_w = opt.img_size
        ratio = net_h/img_h
        
        
        if "perceptual" in opt.network:
            slope_rv = norm(loc = 0, scale = .5)
            offset_rv = norm(loc = 0, scale = 1.)

        with open(index_txt, "r") as f:
            lines = f.readlines()

        for line in lines:
            # yaw=>roll, roll=>pitch, pitch=>yaw
            img_path, pitch, yaw, roll, focal, distor, offset, fov_deg = line.split()
            image_paths.append(os.path.join(dataset_root, phase, img_path))
            b_cached.append(False)
            if opt.phase != 'test':
                cache_names.append(str(uuid.uuid4()))
            if float(fov_deg) > 143.:
                fov_deg = 143.
            elif float(fov_deg) < 33.:
                fov_deg = 33.
            
            if "edgeattention" in opt.network:
                edge_paths.append(os.path.join(dataset_root, phase, "edge", f"{img_path}"))
                if not opt.only_gan:
                    semantic_paths.append(os.path.join(dataset_root, phase, "semantic", img_path.replace("/", "/semantic_map/")))

            focal = float(focal) * ratio
        
            label = {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw), "focal": float(focal), "distor": float(distor), "offset": float(offset), "fov_deg": float(fov_deg)}
            
            if "perceptual" in opt.network:
                step = 1.0 / 256.
                slope_bin_idx = math.floor(slope_rv.cdf(np.deg2rad(float(roll))) / step)
                if slope_bin_idx < 0:
                    slope_bin_idx = 0
                elif slope_bin_idx > 255:
                    slope_bin_idx = 255
                label["slope_label"] = np.asarray([0] * 256)
                label["slope_label"][slope_bin_idx] = 1
                
                offset_bin_idx = math.floor(offset_rv.cdf(float(offset)) / step)
                if offset_bin_idx < 0:
                    offset_bin_idx = 0
                elif offset_bin_idx > 255:
                    offset_bin_idx = 255
                label["offset_label"] = np.asarray([0] * 256)
                label["offset_label"][offset_bin_idx] = 1.
                
                step = (143.-33.) / 256.
                fov_bin_idx = math.floor((float(fov_deg)-33.) / step)
                if fov_bin_idx < 0:
                    fov_bin_idx = 0
                elif fov_bin_idx > 255:
                    fov_bin_idx = 255
                label["fov_label"] = np.asarray([0] * 256)
                label["fov_label"][fov_bin_idx] = 1     
            elif "deepcalib" in opt.network:
                focal_start = 50. * ratio
                focal_end = 500. * ratio
                classes_focal = torch.linspace(focal_start, focal_end, 46)
                label["focal_label"] = torch.argmin(torch.abs(classes_focal - label["focal"]))
                classes_distortion = torch.linspace(0., 1.2, 61)
                label["distor_label"] = torch.argmin(torch.abs(classes_distortion - label["distor"]))
            
            labels.append(label)
                


        ##### for debugging #####
        if opt.debug:
            length = opt.batchSize
            if opt.phase == "train":
                length = opt.batchSize*20
            image_paths = image_paths[:length]
        ##########################

        ##### for debugging #####
        if opt.debug:
            labels = labels[:length]
        ##########################

        return image_paths, labels, edge_paths, semantic_paths, b_cached, cache_names


    def __getitem__(self, index):
        params = get_params(self.opt, (self.opt.img_size[1], self.opt.img_size[0]))
        input_dict = {
            'path': self.image_paths[index],
            'label': self.labels[index],
        }

        if self.opt.phase != 'test':
            cache_path = os.path.join(self.opt.cache_root, self.cache_name, self.cache_names[index])
        cache_tensor = {} if self.b_cached[index] == False else torch.load(cache_path)

        if self.b_cached[index] == False:
            # Image
            image = Image.open(input_dict['path']).convert('RGB')

            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image)
            input_dict['image'] = image_tensor
            cache_tensor['image'] = input_dict['image']

            if "edgeattention" in self.opt.network:
                exclude_idx = [] if self.opt.indoor else [0, 1, 8, 9, 10]
                transform_image = get_transform(self.opt, params, Image.NEAREST, normalize=False)
                edge = Image.open(self.edge_paths[index]).convert('RGB')
                edge_tensor = transform_image(edge)
                input_dict["edge"] = edge_tensor
                cache_tensor['edge'] = edge_tensor
                if not self.opt.only_gan:
                    semantic_path = self.semantic_paths[index]

                    ## for 299x299
                    # semantic = np.load(f"{semantic_path}.npy")
                    # semantic_tensor = torch.tensor(semantic).float()
                    # input_dict["semantic"] = semantic_tensor
                    # weight_map = torch.tensor(np.load(f"{semantic_path}.weight.npy")).float()
                    ## for 299x299
                    
                    ## for 256x256
                    semantic = np.asarray(np.load(semantic_path.replace(".png", ".npz"), allow_pickle=True)['arr_0'] * 255., dtype=np.uint8)
                    semantic = np.delete(semantic, exclude_idx, 0)

                    semantic_tensor = torch.tensor([])
                    for i in range(semantic.shape[0]):
                        temp_semantic = semantic[i]
                        temp_semantic = transform_image(Image.fromarray(temp_semantic))
                        semantic_tensor = torch.cat((semantic_tensor, temp_semantic), 0)

                    input_dict["semantic"] = semantic_tensor
                    cache_tensor['semantic'] = semantic_tensor
                    
                    if 'multi' in self.opt.w_type or 'mean' in self.opt.w_type:
                        rel_w_path = semantic_path.replace('.png', '.weight.npz')
                        if not 'size_weighted' in self.opt.w_type:
                            rel_weight_map = np.asarray(np.load(rel_w_path)['modi_rel'])
                        else:
                            rel_weight_map = np.asarray(np.load(rel_w_path)['modi_rel_size_weighted'])

                        inshade_w_path = semantic_path.replace('.png', '.non_smoothed_inshade_weight.npz') if self.opt.non_smoothed else semantic_path.replace('.png', '.inshade_weight.npz')
                        inshade_weight_map = np.asarray(np.load(inshade_w_path)['inshade_weight'])
                            
                        weight_map = (rel_weight_map * inshade_weight_map) if 'multi' in self.opt.w_type else (rel_weight_map+inshade_weight_map)/2

                        weight_map = transform_image(Image.fromarray(np.asarray(weight_map*255., dtype=np.uint8)))
                    elif self.opt.w_type.lower() != 'none':
                        if self.opt.w_type == 'inshade_weight':
                            weight_path = semantic_path.replace('.png', '.non_smoothed_inshade_weight.npz') if self.opt.non_smoothed else semantic_path.replace('.png', '.inshade_weight.npz')
                        else:
                            weight_path = semantic_path.replace('.png', '.weight.npz')
                        weight_map = np.asarray(np.load(weight_path)[self.opt.w_type] * 255., dtype=np.uint8)
                        weight_map = transform_image(Image.fromarray(weight_map))
                    ## for 256x256 
                    input_dict["weight_map"] = weight_map[0] if self.opt.w_type.lower() != 'none' else torch.as_tensor(-1)
                    cache_tensor['weight_map'] = input_dict["weight_map"]
            if self.opt.phase != 'test':            
                torch.save(cache_tensor, cache_path)
            self.b_cached[index] = True
        else:
            input_dict['image'] = cache_tensor['image']
            if "edgeattention" in self.opt.network:
                input_dict["edge"] = cache_tensor['edge']
                if not self.opt.only_gan:
                    input_dict["semantic"] = cache_tensor['semantic']
                    input_dict["weight_map"] = cache_tensor['weight_map']

        # if self.opt.crop_size != 0:
        #     label = self.transform_fov(image, label)
            
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
    
    def postprocess(self, input_dict):
        return input_dict
