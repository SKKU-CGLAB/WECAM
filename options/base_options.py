import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='fov_regression_with_aug', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='0, 1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='/data/checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='dfov', help='which model to extract features')
        parser.add_argument('--network', type=str, default='basic', help='which model to extract features')
        parser.add_argument('--norm', type=str, default='batchinstance', help='instance normalization or batch normalization [instance | batch | batchinstance | none]')
        # parser.add_argument('--semantic',  action='store_true', help='input with semantic')
        parser.add_argument('--use_pretrained', action='store_true', help='Does ft_network use pretrained model?')
        parser.add_argument('--only_gan', action='store_true', help='for training only GAN')
        parser.add_argument('--pretrained_gan', action='store_true', help='training with pretrained Edge-GAN')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='keep_aspect', help='scaling and cropping of images at load time.', choices=("keep_aspect","resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, nargs=2, default=[1024, 1024], help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=0, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=181, help='# of gt label (0~180 : fovx (0-180 deg), 181~361: fovy (0-180 deg), if #of output channel is 1, it is regression model')
        parser.add_argument('--output_nc', type=int, default=181, help='# of output channels(0~180 : fovx (0-180 deg), 181~361: fovy (0-180 deg), if # of output channel is 1, it is regression model')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--ce_w', type=float, default=1.0, help='SORD_CE weight for loss function')
        parser.add_argument('--mse_w', type=float, default=1.0, help='MSE weight for loss function')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--w_type', type=str, default='modi_rel', help='edge weight type [modi_rel | modi_rel_size_weighted]')
        parser.add_argument('--non_smoothed', action='store_true')
        parser.add_argument('--non_SORD', action='store_true')


        # for setting inputs
        parser.add_argument('--dataset_mode', type=str, default='cityscapes')
        parser.add_argument('--root', type=str, default=r"/data/dataset", help='content dataroot')
        parser.add_argument('--dataset_name', type=str, default="none", help='content dataroot')
        parser.add_argument('--indoor', action='store_true')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')
        parser.add_argument('--data_aug', action='store_true',
                            help='if specified, train or test with fov augmentation using crop')
        parser.add_argument('--sample', action='store_true', help='50000 datasets for training')
        parser.add_argument('--cache_root', type=str, default="/dfov/.cache", help='content dataroot')

        # for displays
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.num_upsampling_layers = 'more'
        opt.isTrain = self.isTrain   # train or test

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
