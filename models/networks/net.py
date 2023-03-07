from re import S
from dominate.tags import output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
import torchvision
import functools
from models.networks import base_network
from models.networks.base_network import BaseNetwork
from models.networks.layers import BatchInstanceNorm2d


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'batchinstance':
        norm_layer = functools.partial(BatchInstanceNorm2d, affine=True)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dir=0):
        super().__init__()
        # vertical, height
        if dir == 0:
            kernel_size1, kernel_size2 = (3, 1), (3, 1)
            padding1, padding2 = (1, 0), (1, 0)
        # horizontal, width
        elif dir == 1:
            kernel_size1, kernel_size2 = (1, 3), (1, 3)
            padding1, padding2 = (0, 1), (0, 1)
        elif dir == -1:
            kernel_size1, kernel_size2 = (3, 3), (3, 3)
            padding1, padding2 = (1, 1), (1, 1)
        
        if stride == 2:
            kernel_size1 = 3
            padding1 = 1

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size1, stride=stride, padding=padding1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size2, stride=1, padding=padding2, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel , kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x, edge=None):
        if edge == None:
            x = self.conv_block(x) + self.shortcut(x)
        else:
            x = self.conv_block(x) + self.shortcut(edge)
        x = self.relu(x)
        return x
    
   
class BasicNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[299, 299])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc

        self.estimator = nn.Sequential(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, opt.ngf, kernel_size=7, padding=0, bias=False),
                nn.BatchNorm2d(opt.ngf),
                nn.ReLU(True)
            ),
            nn.Sequential(
                BasicConvBlock(opt.ngf, opt.ngf * 2, stride=2, dir=-1),
                BasicConvBlock(opt.ngf * 2, opt.ngf * 2, stride=1, dir=-1)
            ), # 64->128
            nn.Sequential(
                BasicConvBlock(opt.ngf * 2, opt.ngf * 4, stride=2, dir=-1),
                BasicConvBlock(opt.ngf * 4, opt.ngf * 4, stride=1, dir=-1),
            ), # 128->256
            nn.Sequential(
                BasicConvBlock(opt.ngf * 4, opt.ngf * 8, stride=2, dir=-1),
                BasicConvBlock(opt.ngf * 8, opt.ngf * 8, stride=1, dir=-1),
            ), # 256->512
            nn.Sequential(
                BasicConvBlock(opt.ngf * 8, opt.ngf * 16, stride=2, dir=-1),
                BasicConvBlock(opt.ngf * 16, opt.ngf * 16, stride=1, dir=-1),
            ) # 512->1024
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.roll_fc = nn.Sequential(
           nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.pitch_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        # self.yaw_fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 512),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, output_nc),
        # )
        self.focal_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.distor_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )

    def forward(self, input):
        output = self.estimator(input)
        output = self.avg_pool(output)
        output = torch.flatten(output, 1)

        # roll, pitch, yaw, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.yaw_fc(output), self.focal_fc(output), self.distor_fc(output)
        # return (roll, pitch, yaw, focal, distor)

        roll, pitch, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.focal_fc(output), self.distor_fc(output)
        return (roll, pitch, focal, distor)


class DeepFocalNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.backbone = torchvision.models.alexnet(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        top_in_ftrs = self.backbone.classifier[6].in_features
        
        self.fov_fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(top_in_ftrs, 1)
        )


    def forward(self, input):
        out = self.backbone.features(input)
        out = self.backbone.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fov_fc(out)

        return out


class DeepCalibNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[299, 299])
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.backbone= torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        self.fc_focal = nn.Linear(2048, 46)
        self.fc_distor = nn.Linear(2048, 61)
        

    def forward(self, input):
        ##### Focal length regression ####
        # N x 3 x 299 x 299
        x = self.backbone.Conv2d_1a_3x3(input)
        # N x 32 x 149 x 149
        x = self.backbone.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.backbone.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.backbone.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.backbone.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.backbone.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.backbone.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.backbone.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.backbone.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.backbone.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.backbone.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.backbone.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.backbone.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.backbone.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.backbone.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        out_focal = self.fc_focal(x)
        out_distor = self.fc_distor(x)
        ##### Focal length regression ####

        return (out_focal, out_distor)
    
    
class PerceptualNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc

        self.backbone = torchvision.models.densenet121(pretrained=True)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        self.roll_fc = nn.Linear(1024, 256) # [− π / 2 , π / 2 ], µ = 0, σ = 0.5, slope, 256 bins
        self.offset_fc = nn.Linear(1024, 256) # [−2.0,2.0], µ = 0, σ = 1, offset, 256 bins
        self.v_fov_fc = nn.Linear(1024, 256) # uniform 256 bins, vertical fov  33~143 degree
        
    def forward(self, input):
        features = self.backbone.features(input)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out_roll = self.roll_fc(out)
        out_offset_fc = self.offset_fc(out)
        out_v_fov = self.v_fov_fc(out)
        
        return out_roll, out_offset_fc, out_v_fov


class ResNet(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.output_nc = self.opt.output_nc

        self.model_ft = torchvision.models.resnet101(pretrained=True)
        for param in self.model_ft.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)

        return x


class AsymmResNet(ResNet):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.x_conv1 = BasicConvBlock(2048, 2048, stride=1, dir=1)
        self.y_conv1 = BasicConvBlock(2048, 2048, stride=1, dir=0)
        
        self.x_fc = nn.Sequential(
            nn.Linear(2048, self.output_nc),
        )
        self.y_fc = nn.Sequential(
            nn.Linear(2048, self.output_nc),
        )


    def forward(self, x):
        feat = super().forward(x)

        out_x = self.x_conv1(feat)
        out_x = self.model_ft.avgpool(out_x)
        out_x = torch.flatten(out_x, 1)
        out_x = self.x_fc(out_x)

        out_y = self.y_conv1(feat)
        out_y = self.model_ft.avgpool(out_y)
        out_y = torch.flatten(out_y, 1)
        out_y = self.x_fc(out_y)
        
        return out_x, out_y


class SymmResNet(ResNet):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.x_conv1 = BasicConvBlock(2048, 2048, stride=1, dir=-1)
        self.y_conv1 = BasicConvBlock(2048, 2048, stride=1, dir=-1)
        
        self.x_fc = nn.Sequential(
            nn.Linear(2048, self.output_nc),
        )
        self.y_fc = nn.Sequential(
            nn.Linear(2048, self.output_nc),
        )


    def forward(self, x):
        feat = super().forward(x)

        out_x = self.x_conv1(feat)
        out_x = self.model_ft.avgpool(out_x)
        out_x = torch.flatten(out_x, 1)
        out_x = self.x_fc(out_x)

        out_y = self.y_conv1(feat)
        out_y = self.model_ft.avgpool(out_y)
        out_y = torch.flatten(out_y, 1)
        out_y = self.x_fc(out_y)
        
        return out_x, out_y

        
class EdgeAttentionCalibNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[299, 299])
        # parser.set_defaults(load_size=[256, 256])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_semantic = 20 if self.opt.indoor else 17
        output_nc = self.opt.output_nc
        norm_layer = get_norm_layer(norm_type=opt.norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.netG = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
        self.netD = NLayerDiscriminator(3+3, opt.ndf, n_layers=3, norm_layer=norm_layer)

        if self.opt.pretrained_gan:
            for param in self.netG.parameters():
                param.requires_grad = False
            for param in self.netD.parameters():
                param.requires_grad = False
        

        # edge attention module
        self.edge_maps = [] # for edge attention
        self.edge_attention_module = nn.ModuleList([])
        self.edge_attention_module.append(nn.Sequential(
            BasicConvBlock(in_channel=3, out_channel=self.n_semantic, stride=1, dir=-1),
            nn.Sigmoid()
        ))
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = mult = 2 ** i
            self.edge_attention_module.append(nn.Sequential(
                BasicConvBlock(opt.ngf * mult, opt.ngf * mult, stride=1, dir=-1),
                nn.Sigmoid()
            ))
        
        #FoV estimator
        self.estimator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(self.n_semantic, opt.ngf, kernel_size=7, padding=0, bias=use_bias),
                nn.BatchNorm2d(opt.ngf),
                nn.ReLU(True)
            ),
            nn.Sequential(
                BasicConvBlock(opt.ngf, opt.ngf * 2, stride=2, dir=-1),
            ), # 64->128
            nn.Sequential(
                BasicConvBlock(opt.ngf * 2, opt.ngf * 4, stride=2, dir=-1),
            ), # 128->256
            nn.Sequential(
                BasicConvBlock(opt.ngf * 4, opt.ngf * 8, stride=2, dir=-1),
            ), # 256->512
            nn.Sequential(
                BasicConvBlock(opt.ngf * 8, opt.ngf * 16, stride=2, dir=-1),
            ) # 512->1024
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.roll_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.pitch_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        # self.yaw_fc = nn.Sequential(
        #     # nn.Dropout(0.5),
        #     # nn.Linear(1024, 512),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, output_nc),
        # )
        self.focal_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.distor_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )

    def forward(self, input, edge_weight_map=None, mode="generator"):
        #generate edge
        if mode == "generator":
            # release
            del self.edge_maps
            self.edge_maps = [] # for edge attention
            for i, layer in enumerate(self.netG.model):
                output = layer(input)
                input = output
                # layer 13 ~ 16
                if i > 12:
                    self.edge_maps.append(output.detach())
                    
                    # [128, 128, 128], [64, 256, 256], [3, 256, 256]
            fake_B = output
            return fake_B

        if mode == "discriminator":
            pred = self.netD(input)
            return pred

        if mode == "estimation":
            for i, layer in enumerate(self.estimator):
                # edge attention
                if i in range(4):
                    edge_feat = self.edge_attention_module[i](self.edge_maps[-1-i])
                    weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                    weighted_edge_feat = edge_feat * weight_map
                    output = layer(weighted_edge_feat*input+input)
                else:
                    output = layer(input)

                input = output
                
            output = F.relu(output, True)
            output = self.avg_pool(output)
            output = torch.flatten(output, 1)
        
            # roll, pitch, yaw, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.yaw_fc(output), self.focal_fc(output), self.distor_fc(output)
            roll, pitch, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.focal_fc(output), self.distor_fc(output)
            # return (roll, pitch, yaw, focal, distor)
            return (roll, pitch, focal, distor)
        
class EdgeAttentionCalibPerceptualNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        
        norm_layer = get_norm_layer(norm_type=opt.norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.netG = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
        self.netD = NLayerDiscriminator(3+3, opt.ndf, n_layers=3, norm_layer=norm_layer)
        self.n_semantic = 20 if self.opt.indoor else 17

        if self.opt.pretrained_gan:
            for param in self.netG.parameters():
                param.requires_grad = False
            for param in self.netD.parameters():
                param.requires_grad = False
        
        self.backbone = torchvision.models.densenet121(pretrained=True)
        # edge attention module
        self.edge_attention_module = nn.ModuleList([
            nn.Sequential(
                BasicConvBlock(in_channel=3, out_channel=self.n_semantic, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=self.n_semantic, out_channel=64, stride=2, dir=-1),
                BasicConvBlock(in_channel=64, out_channel=64, stride=2, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=64, out_channel=128, stride=2, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=128, out_channel=256, stride=2, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=256, out_channel=512, stride=2, dir=-1),
                nn.Sigmoid()
            )
        ])
        
        #FoV estimator
        self.estimator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_semantic, opt.ngf, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ), # 64, 56, 56
            nn.Sequential(
                self.backbone.features.denseblock1,
                self.backbone.features.transition1,
            ), # 128, 28, 28
            nn.Sequential(
                self.backbone.features.denseblock2,
                self.backbone.features.transition2    
            ), # 256, 14, 14
            nn.Sequential(
                self.backbone.features.denseblock3,
                self.backbone.features.transition3  
            ), # 512, 7, 7
            self.backbone.features.denseblock4,
            nn.BatchNorm2d(1024)
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.roll_fc = nn.Sequential(
            nn.Linear(1024, 256),
        )
        self.pitch_fc = nn.Sequential(
            nn.Linear(1024, 256),
        )
        # self.yaw_fc = nn.Sequential(
        #     # nn.Dropout(0.5),
        #     # nn.Linear(1024, 512),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, output_nc),
        # )
        self.focal_fc = nn.Sequential(
            nn.Linear(1024, 256),
        )
        self.distor_fc = nn.Sequential(
            nn.Linear(1024, 256),
        )

    def forward(self, input, edge_weight_map=None, mode="generator"):
        #generate edge
        if mode == "generator":
            self.fake_B = self.netG(input)
            return self.fake_B

        if mode == "discriminator":
            pred = self.netD(input)
            return pred

        if mode == "estimation":
            n_attention = len(self.edge_attention_module)
            edge_feat = self.fake_B.detach()
            for i, layer in enumerate(self.estimator):
                if i < n_attention:
                    # edge attention
                    edge_feat = self.edge_attention_module[i](edge_feat)
                    weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                    weighted_edge_feat = edge_feat * weight_map
                    output = layer(weighted_edge_feat*input+input)
                else:
                    output = layer(input)

                input = output

            output = F.relu(output, inplace=True)
            output = self.avg_pool(output)
            output = torch.flatten(output, 1)
        
            # roll, pitch, yaw, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.yaw_fc(output), self.focal_fc(output), self.distor_fc(output)
            roll, pitch, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.focal_fc(output), self.distor_fc(output)
            # return (roll, pitch, yaw, focal, distor)
            return (roll, pitch, focal, distor)


class SingleEdgeAttentionCalibNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        self.n_semantic = 20 if self.opt.indoor else 17
           
        self.backbone = torchvision.models.densenet121(pretrained=True)
        
        #FoV estimator
        self.estimator = nn.ModuleList([
          self.backbone.features[0:4], 
          # 64, 56, 56
            nn.Sequential(
                self.backbone.features.denseblock1,
                self.backbone.features.transition1,
            ), # 128, 28, 28
            nn.Sequential(
                self.backbone.features.denseblock2,
                self.backbone.features.transition2    
            ), # 256, 14, 14
            nn.Sequential(
                self.backbone.features.denseblock3,
                self.backbone.features.transition3  
            ), # 512, 7, 
            nn.Sequential(
                self.backbone.features.denseblock4,
                nn.BatchNorm2d(1024)
            )
        ])
        
        self.roll_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.pitch_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.focal_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.distor_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.fcs = nn.ModuleList([
            self.roll_fc,
            self.pitch_fc,
            self.focal_fc,
            self.distor_fc
        ])

        setattr(self.fcs, 'init_weights', True)

        # edge attention module
        self.edge_attention_module = nn.ModuleList([
            #BasicConvBlock(3, self.n_semantic, stride=1, dir=-1), # n_semantic, 224, 224
            nn.Sequential(
                BasicConvBlock(in_channel=3, out_channel=64, stride=2, dir=-1), # 64, 112, 112
                BasicConvBlock(in_channel=64, out_channel=64, stride=2, dir=-1), # 64, 56, 56
            ),
            BasicConvBlock(in_channel=64, out_channel=128, stride=2, dir=-1), # 128, 28, 28
            BasicConvBlock(in_channel=128, out_channel=256, stride=2, dir=-1), # 256, 14, 14
            BasicConvBlock(in_channel=256, out_channel=512, stride=2, dir=-1), # 512, 7, 7
            BasicConvBlock(in_channel=512, out_channel=1024, stride=1, dir=-1) # 1024, 7, 7
        ])

        setattr(self.edge_attention_module, 'init_weights', True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, edge, edge_weight_map=None):
        edge_feat = edge
        b_weighted = True if edge_weight_map.max() != -1 else False

        for i, est_layer in enumerate(self.estimator):
            # edge attention
            if b_weighted:
                weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                weighted_edge_feat = self.sigmoid(edge_feat * weight_map * input)
            else:
                weighted_edge_feat = self.sigmoid(edge_feat * input)

            output = est_layer(weighted_edge_feat + input)
            edge_feat = self.edge_attention_module[i](edge_feat)
        
            input = output
        
        input = F.relu(input, inplace=True)
        if b_weighted:
            weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
            weighted_edge_feat = self.sigmoid(edge_feat * weight_map * input)
        else:
            weighted_edge_feat = self.sigmoid(edge_feat * input)
        output = self.avg_pool(weighted_edge_feat + input)
        output = torch.flatten(output, 1)
    
        roll, pitch, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.focal_fc(output), self.distor_fc(output)

        return (roll, pitch, focal, distor)


class SimpleNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        self.backbone = torchvision.models.densenet121(pretrained=True)
        
        self.roll_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.pitch_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.focal_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.distor_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, output_nc),
            #nn.Dropout(0.5),
            #nn.Linear(512, output_nc),
        )
        self.fcs = nn.ModuleList([
            self.roll_fc,
            self.pitch_fc,
            self.focal_fc,
            self.distor_fc
        ])
        setattr(self.fcs, 'init_weights', True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        features = self.backbone.features(input)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
    
        roll, pitch, focal, distor = self.roll_fc(out), self.pitch_fc(out), self.focal_fc(out), self.distor_fc(out)

        return (roll, pitch, focal, distor)
    

class OnlySemanticNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        self.n_semantic = 20 if self.opt.indoor else 17
           
        self.backbone = torchvision.models.densenet121(pretrained=True)
        
        #FoV estimator
        self.estimator = nn.Sequential([
            nn.Sequential(
                nn.Conv2d(self.n_semantic, opt.ngf, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ), # 64, 56, 56
            nn.Sequential(
                self.backbone.features.denseblock1,
                self.backbone.features.transition1,
            ), # 128, 28, 28
            nn.Sequential(
                self.backbone.features.denseblock2,
                self.backbone.features.transition2    
            ), # 256, 14, 14
            nn.Sequential(
                self.backbone.features.denseblock3,
                self.backbone.features.transition3  
            ), # 512, 7, 
            nn.Sequential(
                self.backbone.features.denseblock4,
                nn.BatchNorm2d(1024)
            )
        ])
        
        self.roll_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.pitch_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.focal_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.distor_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.fcs = nn.ModuleList([
            self.roll_fc,
            self.pitch_fc,
            self.focal_fc,
            self.distor_fc
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input, edge, edge_weight_map=None):
        edge_feat = self.edge_attention_module[0](edge)

        for i, est_layer in enumerate(self.estimator):
            # edge attention
            if self.opt.w_type.lower() != 'none':
                weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                weighted_edge_feat = edge_feat * weight_map 
            else:
                weighted_edge_feat = edge_feat
        
            output = est_layer(weighted_edge_feat*input+input)
            edge_feat = self.edge_attention_module[i+1](edge_feat)
        
            input = output

        if self.opt.w_type.lower() != 'none':
            weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
            weighted_edge_feat = edge_feat * weight_map 
        else:
            weighted_edge_feat = edge_feat
        
        output = F.relu(weighted_edge_feat*input+input, inplace=True)
        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
    
        roll, pitch, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.focal_fc(output), self.distor_fc(output)

        return (roll, pitch, focal, distor)


class SingleEdgeAttentionPerceptualNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        self.n_semantic = 20 if self.opt.indoor else 17
           
        self.backbone = torchvision.models.densenet121(pretrained=True)
        
        #FoV estimator
        self.estimator = nn.ModuleList([
          self.backbone.features[0:4], 
          # 64, 56, 56
            nn.Sequential(
                self.backbone.features.denseblock1,
                self.backbone.features.transition1,
            ), # 128, 28, 28
            nn.Sequential(
                self.backbone.features.denseblock2,
                self.backbone.features.transition2    
            ), # 256, 14, 14
            nn.Sequential(
                self.backbone.features.denseblock3,
                self.backbone.features.transition3  
            ), # 512, 7, 
            nn.Sequential(
            self.backbone.features.denseblock4,
            nn.BatchNorm2d(1024)
            )
        ])

         # edge attention module
        self.edge_attention_module = nn.ModuleList([
            #BasicConvBlock(3, self.n_semantic, stride=1, dir=-1), # n_semantic, 224, 224
            nn.Sequential(
                BasicConvBlock(in_channel=3, out_channel=64, stride=2, dir=-1), # 64, 112, 112
                BasicConvBlock(in_channel=64, out_channel=64, stride=2, dir=-1), # 64, 56, 56
            ),
            BasicConvBlock(in_channel=64, out_channel=128, stride=2, dir=-1), # 128, 28, 28
            BasicConvBlock(in_channel=128, out_channel=256, stride=2, dir=-1), # 256, 14, 14
            BasicConvBlock(in_channel=256, out_channel=512, stride=2, dir=-1), # 512, 7, 7
            BasicConvBlock(in_channel=512, out_channel=1024, stride=1, dir=-1) # 1024, 7, 7
        ])
        
        setattr(self.edge_attention_module, 'init_weights', True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        
        self.roll_fc = nn.Linear(1024, 256)
        self.offset_fc = nn.Linear(1024, 256)
        self.fov_fc = nn.Linear(1024, 256)

        self.fcs = nn.ModuleList([
            self.roll_fc,
            self.offset_fc,
            self.fov_fc
        ])


    def forward(self, input, edge, edge_weight_map=None):
        edge_feat = edge
        b_weighted = True if edge_weight_map.max() != -1 else False

        for i, est_layer in enumerate(self.estimator):
            # edge attention
            if b_weighted:
                weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                weighted_edge_feat = self.sigmoid(edge_feat * weight_map * input)
            else:
                weighted_edge_feat = edge_feat

            output = est_layer(weighted_edge_feat+input)
            edge_feat = self.edge_attention_module[i](edge_feat)
        
            input = output

        input = F.relu(input, inplace=True)
        if b_weighted:
            weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
            weighted_edge_feat = self.sigmoid(edge_feat * weight_map * input)
        else:
            weighted_edge_feat = edge_feat
        output = self.avg_pool(weighted_edge_feat+input)
        output = torch.flatten(output, 1)
        
        roll, offset, fov = self.roll_fc(output), self.offset_fc(output), self.fov_fc(output)
        return (roll, offset, fov)


class SingleEdgeAttentionDeepFocalNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc

        self.n_semantic = 20 if self.opt.indoor else 17
           
        self.backbone = torchvision.models.alexnet(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #FoV estimator
        self.estimator = nn.ModuleList([
            self.backbone.features[0:3], 
            # 8, 64, 27, 27
            self.backbone.features[3:6],
            # 8, 192, 13, 13
            self.backbone.features[6:8],
            # 8, 384, 13, 13
            self.backbone.features[8:10],
            # 8, 256, 13, 13
            self.backbone.features[10:]
            # 8, 256, 6, 6
        ])

        self.edge_attention_module = nn.ModuleList([
            #BasicConvBlock(3, self.n_semantic, stride=1, dir=-1), # n_semantic, 224, 224
            nn.Sequential(
                BasicConvBlock(in_channel=3, out_channel=64, stride=4, dir=-1), # 64, 112, 112
                BasicConvBlock(in_channel=64, out_channel=64, stride=2, dir=-1), # 64, 56, 56
            ),
            BasicConvBlock(in_channel=64, out_channel=128, stride=2, dir=-1), # 128, 28, 28
            BasicConvBlock(in_channel=128, out_channel=256, stride=2, dir=-1), # 256, 14, 14
            BasicConvBlock(in_channel=256, out_channel=512, stride=2, dir=-1), # 512, 7, 7
            BasicConvBlock(in_channel=512, out_channel=1024, stride=1, dir=-1) # 1024, 7, 7
        ])
        
        self.fov_fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)
        )

        self.fcs = nn.ModuleList([
            self.fov_fc
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input, edge, edge_weight_map=None):
        edge_feat = edge
        b_weighted = True if edge_weight_map.max() != -1 else False

        for i, est_layer in enumerate(self.estimator):
            # edge attention
            if b_weighted:
                weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                weighted_edge_feat = self.sigmoid(edge_feat * weight_map * input)
            else:
                weighted_edge_feat = edge_feat

            output = est_layer(weighted_edge_feat+input)
            edge_feat = self.edge_attention_module[i](edge_feat)
        
            input = output

        input = F.relu(input, inplace=True)
        if b_weighted:
            weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
            weighted_edge_feat = self.sigmoid(edge_feat * weight_map * input)
        else:
            weighted_edge_feat = edge_feat
    
        output = self.avg_pool(weighted_edge_feat+input)
        output = torch.flatten(output, 1)
        
        output = self.fov_fc(output)

        return output


class SingleEdgeAttentionDeepCalibNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[299, 299])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        self.n_semantic = 20 if self.opt.indoor else 17
           
        self.backbone = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        self.backbone2 = torchvision.models.inception_v3(pretrained=True, aux_logits=False)

        
         #FoV estimator
        self.estimator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_semantic, 32, kernel_size=3, stride=2, bias=False), 
                nn.BatchNorm2d(32, eps=0.001)
            ),
            self.backbone.Conv2d_2a_3x3,
            self.backbone.Conv2d_2b_3x3,
            self.backbone.Conv2d_3b_1x1,
            self.backbone.Conv2d_4a_3x3,
            self.backbone.Mixed_5b,
            self.backbone.Mixed_5c,
            self.backbone.Mixed_5d,
            self.backbone.Mixed_6a,
            self.backbone.Mixed_6b,
            self.backbone.Mixed_6c,
            self.backbone.Mixed_6d,
            self.backbone.Mixed_6e,
            self.backbone.Mixed_7a,
            self.backbone.Mixed_7b,
            self.backbone.Mixed_7c,
        ])

         # edge attention module
        self.edge_attention_module = nn.ModuleList([
            BasicConvBlock(3, self.n_semantic, stride=1, dir=-1),
            nn.Sequential(
                nn.Conv2d(self.n_semantic, 32, kernel_size=3, stride=2, bias=False), 
                nn.BatchNorm2d(32, eps=0.001)
            ),
            self.backbone2.Conv2d_2a_3x3,
            self.backbone2.Conv2d_2b_3x3,
            self.backbone2.Conv2d_3b_1x1,
            self.backbone2.Conv2d_4a_3x3,
            self.backbone2.Mixed_5b,
            self.backbone2.Mixed_5c,
            self.backbone2.Mixed_5d,
            self.backbone2.Mixed_6a,
            self.backbone2.Mixed_6b,
            self.backbone2.Mixed_6c,
            self.backbone2.Mixed_6d,
            self.backbone2.Mixed_6e,
            self.backbone2.Mixed_7a,
            self.backbone2.Mixed_7b,
            self.backbone2.Mixed_7c,
        ])
        
        self.fc_focal = nn.Linear(2048, 46)
        self.fc_distor = nn.Linear(2048, 61)

        self.fcs = nn.ModuleList([
            self.fc_focal,
            self.fc_distor,
        ])


    def forward(self, input, edge, edge_weight_map=None):
        edge_feat = self.edge_attention_module[0](edge)

        for i, est_layer in enumerate(self.estimator):
            # edge attention
            weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
            weighted_edge_feat = edge_feat * weight_map
            if i == 3 or i == 5:
                maxpool_layer = self.backbone.maxpool1 if i == 3 else self.backbone.maxpool2
                output = est_layer(maxpool_layer(weighted_edge_feat*input+input))
                edge_feat = self.edge_attention_module[i+1](maxpool_layer(weighted_edge_feat+edge_feat))
            else:    
                output = est_layer(weighted_edge_feat*input+input)
                edge_feat = self.edge_attention_module[i+1](weighted_edge_feat+edge_feat)
        
            input = output

        weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
        weighted_edge_feat = edge_feat * weight_map
        
        output = F.relu(weighted_edge_feat*input+input, inplace=True)
        output = self.backbone.avgpool(output)
        output = self.backbone.dropout(output)
        output = torch.flatten(output, 1)

        out_focal = self.fc_focal(output)
        out_distor = self.fc_distor(output)

        return (out_focal, out_distor)


class DualEdgeAttentionCalibNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        self.n_semantic = 20 if self.opt.indoor else 17
           
        self.backbone = torchvision.models.densenet121(pretrained=True)
        self.backbone2 = torchvision.models.densenet121(pretrained=True)
        
        #FoV estimator
        self.estimator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_semantic, opt.ngf, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ), # 64, 56, 56
            nn.Sequential(
                self.backbone.features.denseblock1,
                self.backbone.features.transition1,
            ), # 128, 28, 28
            nn.Sequential(
                self.backbone.features.denseblock2,
                self.backbone.features.transition2    
            ), # 256, 14, 14
            nn.Sequential(
                self.backbone.features.denseblock3,
                self.backbone.features.transition3  
            ), # 512, 7, 7
            self.backbone.features.denseblock4,
            nn.BatchNorm2d(1024)
        ])
        
        self.roll_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.pitch_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.focal_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.distor_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.fcs = nn.ModuleList([
            self.roll_fc,
            self.pitch_fc,
            self.focal_fc,
            self.distor_fc
        ])

        # edge attention module
        self.edge_attention_module = nn.ModuleList([
             nn.Sequential(
                nn.Conv2d(self.n_semantic, opt.ngf, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ), # 64, 56, 56
            nn.Sequential(
                self.backbone2.features.denseblock1,
                self.backbone2.features.transition1,
            ), # 128, 28, 28
            nn.Sequential(
                self.backbone2.features.denseblock2,
                self.backbone2.features.transition2    
            ), # 256, 14, 14
            nn.Sequential(
                self.backbone2.features.denseblock3,
                self.backbone2.features.transition3  
            ), # 512, 7, 7
            self.backbone2.features.denseblock4,
            nn.BatchNorm2d(1024)
        ])
        
        self.edge_roll_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.edge_pitch_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.edge_focal_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.edge_distor_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.edge_fcs = nn.ModuleList([
            self.edge_roll_fc,
            self.edge_pitch_fc,
            self.edge_focal_fc,
            self.edge_distor_fc
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input, edge, edge_weight_map=None):
        edge_feat = edge
        for i, est_layer in enumerate(self.estimator):
            # edge attention
            weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
            weighted_edge_feat = edge_feat * weight_map
            # output = est_layer(weighted_edge_feat.clone().detach()*input+input)
            output = est_layer(weighted_edge_feat.clone().detach()*input+input)
            edge_feat = self.edge_attention_module[i](weighted_edge_feat+edge_feat)
        
            input = output

        weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
        weighted_edge_feat = edge_feat * weight_map
        output = F.relu(weighted_edge_feat.clone().detach()*input+input)
        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
    
        roll, pitch, focal, distor = self.roll_fc(output), self.pitch_fc(output), self.focal_fc(output), self.distor_fc(output)

        edge_output = F.relu(weighted_edge_feat+edge_feat, inplace=True)
        edge_output = self.avg_pool(edge_output)
        edge_output = torch.flatten(edge_output, 1)
    
        edge_roll, edge_pitch, edge_focal, edge_distor = self.edge_roll_fc(edge_output), self.edge_pitch_fc(edge_output), self.edge_focal_fc(edge_output), self.edge_distor_fc(edge_output)

        return (roll, pitch, focal, distor, edge_roll, edge_pitch, edge_focal, edge_distor)


class EdgeAttentionPerceptualNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[224, 224])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        
        norm_layer = get_norm_layer(norm_type=opt.norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.netG = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
        self.netD = NLayerDiscriminator(3+3, opt.ndf, n_layers=3, norm_layer=norm_layer)
        self.n_semantic = 20 if self.opt.indoor else 17

        if self.opt.pretrained_gan:
            for param in self.netG.parameters():
                param.requires_grad = False
            for param in self.netD.parameters():
                param.requires_grad = False
        
        self.backbone = torchvision.models.densenet121(pretrained=True)
        # edge attention module
        self.edge_attention_module = nn.ModuleList([
            nn.Sequential(
                BasicConvBlock(in_channel=3, out_channel=self.n_semantic, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=self.n_semantic, out_channel=64, stride=1, dir=-1),
                nn.Upsample((56, 56)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=64, out_channel=128, stride=1, dir=-1),
                nn.Upsample((28, 28)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=128, out_channel=256, stride=1, dir=-1),
                nn.Upsample((14, 14)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=256, out_channel=512, stride=1, dir=-1),
                nn.Upsample((7, 7)),
                nn.Sigmoid()
            )
        ])
        
        #FoV estimator
        self.estimator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_semantic, opt.ngf, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ), # 64, 56, 56
            nn.Sequential(
                self.backbone.features.denseblock1,
                self.backbone.features.transition1,
            ), # 128, 28, 28
            nn.Sequential(
                self.backbone.features.denseblock2,
                self.backbone.features.transition2    
            ), # 256, 14, 14
            nn.Sequential(
                self.backbone.features.denseblock3,
                self.backbone.features.transition3  
            ), # 512, 7, 7
            self.backbone.features.denseblock4,
            nn.BatchNorm2d(1024)
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.roll_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.offset_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )
        self.fov_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, output_nc),
        )


    def forward(self, input, edge_weight_map=None, mode="generator"):
        #generate edge
        if mode == "generator":
            self.fake_B = self.netG(input)
            return self.fake_B

        if mode == "discriminator":
            pred = self.netD(input)
            return pred

        if mode == "estimation":
            n_attention = len(self.edge_attention_module)
            edge_feat = self.fake_B.detach()
            for i, layer in enumerate(self.estimator):
                if i < n_attention:
                    # edge attention
                    edge_feat = self.edge_attention_module[i](edge_feat)
                    weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                    weighted_edge_feat = edge_feat * weight_map
                    output = layer(weighted_edge_feat*input+input)
                else:
                    output = layer(input)

                input = output

            output = F.relu(output, inplace=True)
            output = self.avg_pool(output)
            output = torch.flatten(output, 1)
        
            roll, offset, fov = self.roll_fc(output), self.offset_fc(output), self.fov_fc(output)
            return (roll, offset, fov)


class EdgeAttentionDeepFocalNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[256, 256])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        
        norm_layer = get_norm_layer(norm_type=opt.norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.netG = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
        self.netD = NLayerDiscriminator(3+3, opt.ndf, n_layers=3, norm_layer=norm_layer)
        self.n_semantic = 20 if self.opt.indoor else 17

        if self.opt.pretrained_gan:
            for param in self.netG.parameters():
                param.requires_grad = False
            for param in self.netD.parameters():
                param.requires_grad = False
        
        self.backbone = torchvision.models.alexnet(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
         #FoV estimator
        self.estimator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_semantic, opt.ngf, kernel_size=11, stride=4, padding=2), # 64, 31, 31
                self.backbone.features[1:3]
            ),
             nn.Sequential(
                self.backbone.features[3:6]
            ),
            nn.Sequential(
                self.backbone.features[6:8]
            ),
            nn.Sequential(
                self.backbone.features[8:]
            )
        ])
        # 192, 15, 15,
        # 384, 15, 15,
        # 256, 15, 15,
        
        self.fov_fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)
        )

        # edge attention module
        self.edge_attention_module = nn.ModuleList([
            nn.Sequential(
                BasicConvBlock(in_channel=3, out_channel=self.n_semantic, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=self.n_semantic, out_channel=64, stride=1, dir=-1),
                nn.Upsample((31,31)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=64, out_channel=192, stride=1, dir=-1),
                nn.Upsample((15,15)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=192, out_channel=384, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=384, out_channel=256, stride=1, dir=-1),
                nn.Sigmoid()
            )
        ])


    def forward(self, input, edge_weight_map=None, mode="generator"):
        #generate edge
        if mode == "generator":
            self.fake_B = self.netG(input)
            return self.fake_B

        if mode == "discriminator":
            pred = self.netD(input)
            return pred

        if mode == "estimation":
            n_attention = len(self.edge_attention_module)
            edge_feat = self.fake_B.detach()
            for i, layer in enumerate(self.estimator):
                if i < n_attention:
                    # edge attention
                    edge_feat = self.edge_attention_module[i](edge_feat)
                    weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                    weighted_edge_feat = edge_feat * weight_map
                    output = layer(weighted_edge_feat*input+input)
                else:
                    output = layer(input)

                input = output

            output = self.backbone.avgpool(output)
            output = torch.flatten(output, 1)

            output = self.fov_fc(output)

            return output


class EdgeAttentionDeepCalibNet(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=[299, 299])
        parser.set_defaults(output_nc=256)
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = self.opt.output_nc
        
        norm_layer = get_norm_layer(norm_type=opt.norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.netG = ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
        self.netD = NLayerDiscriminator(3+3, opt.ndf, n_layers=3, norm_layer=norm_layer)
        self.n_semantic = 20 if self.opt.indoor else 17

        if self.opt.pretrained_gan:
            for param in self.netG.parameters():
                param.requires_grad = False
            for param in self.netD.parameters():
                param.requires_grad = False
        
        self.backbone = torchvision.models.inception_v3(pretrained=True, aux_logits=False)

        #FoV estimator
        self.estimator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.n_semantic, 32, kernel_size=3, stride=2, bias=False), 
                nn.BatchNorm2d(32, eps=0.001)
            ),
            self.backbone.Conv2d_2a_3x3,
            nn.Sequential(
                self.backbone.Conv2d_2b_3x3,
                self.backbone.maxpool1
            ),
            self.backbone.Conv2d_3b_1x1,
            nn.Sequential(
                self.backbone.Conv2d_4a_3x3,
                self.backbone.maxpool2
            ),
            self.backbone.Mixed_5b,
            self.backbone.Mixed_5c,
            self.backbone.Mixed_5d,
            self.backbone.Mixed_6a,
            self.backbone.Mixed_6b,
            self.backbone.Mixed_6c,
            self.backbone.Mixed_6d,
            self.backbone.Mixed_6e,
            self.backbone.Mixed_7a,
            self.backbone.Mixed_7b,
            self.backbone.Mixed_7c,
            self.backbone.avgpool,
            self.backbone.dropout
        ])
    
        self.fc_focal = nn.Linear(2048, 46)
        self.fc_distor = nn.Linear(2048, 61)

        # edge attention module
        self.edge_attention_module = nn.ModuleList([
            nn.Sequential(
                BasicConvBlock(in_channel=3, out_channel=self.n_semantic, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=self.n_semantic, out_channel=32, stride=1, dir=-1),
                nn.Upsample((149,149)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=32, out_channel=32, stride=1, dir=-1),
                nn.Upsample((147,147)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=32, out_channel=64, stride=1, dir=-1),
                nn.Upsample((73,73)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=64, out_channel=80, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=80, out_channel=192, stride=1, dir=-1),
                nn.Upsample((35,35)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=192, out_channel=256, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=256, out_channel=288, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=288, out_channel=288, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=288, out_channel=768, stride=1, dir=-1),
                nn.Upsample((17,17)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=768, out_channel=768, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=768, out_channel=768, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=768, out_channel=768, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=768, out_channel=768, stride=1, dir=-1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=768, out_channel=1280, stride=1, dir=-1),
                nn.Upsample((8,8)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                BasicConvBlock(in_channel=1280, out_channel=2048, stride=1, dir=-1),
                nn.Sigmoid()
            )
        ])


    def forward(self, input, edge_weight_map=None, mode="generator"):
        #generate edge
        if mode == "generator":
            self.fake_B = self.netG(input)
            self.fake_B = nn.Upsample((self.opt.load_size[0], self.opt.load_size[1]))(self.fake_B)
            return self.fake_B

        if mode == "discriminator":
            pred = self.netD(input)
            return pred

        if mode == "estimation":
            n_attention = len(self.edge_attention_module)
            edge_feat = self.fake_B.detach()
            for i, layer in enumerate(self.estimator):
                if i < n_attention:
                    # edge attention
                    edge_feat = self.edge_attention_module[i](edge_feat)
                    weight_map = torch.nn.Upsample(edge_feat.shape[2:])(edge_weight_map.unsqueeze(1))
                    weighted_edge_feat = edge_feat * weight_map
                    output = layer(weighted_edge_feat*input+input)
                else:
                    output = layer(input)

                input = output

            output = torch.flatten(output, 1)

            out_focal = self.fc_focal(output)
            out_distor = self.fc_distor(output)

            return (out_focal, out_distor)
        
"""
From this, the code is referenced from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator


            Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            n_layers_D
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)
                )] # 0

        n_downsampling = 3
        for i in range(n_downsampling):  # add downsampling layers, 1~3
            mult = 2 ** i
            model += [nn.Sequential(
                        nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)
                    )]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks, 4~12
            model += [ResnetBlock(ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers 13~15
            mult = 2 ** (n_downsampling - i)
            output_padding = 1 if i == 1 else 0
            model += [nn.Sequential(
                        nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=output_padding,
                                         bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)
                    )]

        # 16
        model += [nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)



class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, input_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(input_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, input_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(input_dim, out_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(out_dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(out_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)