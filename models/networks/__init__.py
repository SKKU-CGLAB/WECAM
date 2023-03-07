import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
import util.util as util
from torch.optim import lr_scheduler


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    net_cls = find_network_using_name(opt.network, 'net')
    parser = net_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt, *args):

    net = cls(opt, *args)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)

    return net


def define_Net(opt):
    net_cls = find_network_using_name(opt.network, 'net')
    net = create_network(net_cls, opt)

    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.first_epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
