from dominate.tags import label
import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class DfovModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.losses = {}
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.scaler = torch.cuda.amp.GradScaler()

        self.optimizers = []
        self.schedulers = []

        self.net = self.initialize_networks(opt)
        self.softmax = torch.nn.Softmax(1)
        self.MSE = torch.nn.MSELoss()
        if "perceptual" in opt.network :
            self.bins = np.load("./bins.npz")

        # set loss functions
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.multi_CE = networks.MultitargetCrossEntropyLoss()
        self.KL_div = torch.nn.KLDivLoss(reduction='batchmean')
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.cosh = networks.LogCoshLoss()


    def initialize_networks(self, opt):
        net = networks.define_Net(opt)

        if not opt.isTrain or opt.continue_train:
            net = util.load_network(net, 'Net', opt.which_epoch, opt)

        return net
    
    def create_optimizers(self, opt):
        if "dual" in opt.network:
            params_estimator = list(self.net.estimator.parameters()) + list(self.net.fcs.parameters())
            self.optimizer_estimator = torch.optim.Adam(params_estimator, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_estimator)

            params_edge_estimator = list(self.net.edge_attention_module.parameters()) + list(self.net.edge_fcs.parameters())
            self.optimizer_edge_estimator = torch.optim.Adam(params_edge_estimator, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_edge_estimator)
        elif "single" in opt.network:
            params_estimator = list()
            for param in self.net.estimator.parameters():
                if param.requires_grad:
                    params_estimator.append(param)
            for param in self.net.edge_attention_module.parameters():
                if param.requires_grad:
                    params_estimator.append(param)
            params_estimator += list(self.net.fcs.parameters())
            if "deepfocal" in opt.network:
                self.optimizer_estimator = torch.optim.SGD(params_estimator, lr=opt.lr, momentum=0.9, weight_decay=1e-4)
            else:
                self.optimizer_estimator = torch.optim.Adam(params_estimator, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_estimator)
        elif "edgeattention" in opt.network:
            if not opt.pretrained_gan:
                # initialize optimizers and schedulers
                self.bin_gate_G = [p for p in self.net.netG.parameters() if getattr(p, 'bin_gate', False)]
                params_G = [{'params': [p for p in self.net.netG.parameters() if not getattr(p, 'bin_gate', False)]},
                {'params': self.bin_gate_G, 
                'lr': opt.lr * 10.0, 'weight_decay': 0, 'betas': (opt.beta1, opt.beta2)}]
                self.optimizer_G = torch.optim.Adam(params_G, lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_G)

                self.bin_gate_D = [p for p in self.net.netD.parameters() if getattr(p, 'bin_gate', False)]
                params_D =  [{'params': [p for p in self.net.netD.parameters() if not getattr(p, 'bin_gate', False)]},
                {'params': self.bin_gate_D, 
                'lr': opt.lr * 10.0, 'weight_decay': 0, 'betas': (opt.beta1, opt.beta2)}]
                self.optimizer_D = torch.optim.Adam(params_D, lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)

            if not opt.only_gan:
                if "edgeattentioncalib" in opt.network:
                    params_estimator = list(self.net.estimator.parameters()) + list(self.net.edge_attention_module.parameters())
                    params_estimator += list(self.net.roll_fc.parameters()) + list(self.net.pitch_fc.parameters()) + list(self.net.focal_fc.parameters()) + list(self.net.distor_fc.parameters()) 
                    self.optimizer_estimator = torch.optim.Adam(params_estimator, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
                elif "perceptual" in opt.network:
                    params_estimator = list(self.net.estimator.parameters()) + list(self.net.edge_attention_module.parameters())
                    params_estimator += list(self.net.roll_fc.parameters()) + list(self.net.offset_fc.parameters()) + list(self.net.fov_fc.parameters())
                    self.optimizer_estimator = torch.optim.Adam(params_estimator, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
                elif "deepcalib" in opt.network:
                    params_estimator = list(self.net.estimator.parameters()) + list(self.net.edge_attention_module.parameters())
                    params_estimator += list(self.net.fc_focal.parameters()) + list(self.net.fc_distor.parameters())
                    self.optimizer_estimator = torch.optim.Adam(params_estimator, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
                elif "deepfocal" in opt.network:
                    params_estimator = list()
                    for param in self.net.estimator.parameters():
                        if param.requires_grad:
                            params_estimator.append(param)
                    params_estimator += list(self.net.edge_attention_module.parameters())
                    params_estimator += list(self.net.fov_fc.parameters())
                    self.optimizer_estimator = torch.optim.SGD(params_estimator, lr=opt.lr, momentum=0.9, weight_decay=1e-4)
                self.optimizers.append(self.optimizer_estimator)
            
        elif opt.network in "deepcalib" or opt.network in "perceptual" or opt.network in "simple":
            params = list()
            for param in self.net.parameters():
                if param.requires_grad:
                    params.append(param)
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            
        elif opt.network in "deepfocal":
            params = list()
            for param in self.net.parameters():
                if param.requires_grad:
                    params.append(param)
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizers.append(self.optimizer)
        else:
            params = list()
            for param in self.net.parameters():
                if param.requires_grad:
                    params.append(param)
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer)

        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if opt.continue_train:
            self.optimizers, self.schedulers, self.scaler = util.load_optimizer(self.optimizers, self.schedulers, self.scaler, 'Net', opt.which_epoch, opt)
        
        return self.optimizers, self.schedulers


    # preprocess the input, such as moving the tensors to GPUs
    # and transforming the label map to one-hot encoding (for SIS)
    # |data|: dictionary of the input data
    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
           data = data.cuda()
           
        return data


    def forward(self, data):
        if self.opt.phase == "train":
            return self.inference_fov(data)
        else:
            with torch.no_grad():
                return self.inference_fov(data)
        # else:
        #     with torch.no_grad():
        #         return self.test(data)

                
    def inference_fov(self, data):
        
        def bin2Val(bin_id, bin_edges):
            vals = []
            for id in bin_id:      
                if id == 0 and bin_edges[0] == -np.inf:
                    val = bin_edges[1]
                elif id == 255 and bin_edges[-1] == np.inf:
                    val = bin_edges[id]
                else:
                    val = (bin_edges[id] + bin_edges[id+1]) / 2.
                vals.append(val)
            return torch.tensor(vals).to(self.device)
        def offsetToPitch(rolls, offsets, fovs, height):
            slopes = torch.tan(torch.deg2rad(rolls))
            b_ps = offsets * torch.sqrt( ( torch.square(slopes) + 1 ) )
            focal_lengths = height / torch.tan(torch.deg2rad(fovs)/2)
            pitchs = torch.arctan(b_ps/(2*focal_lengths)*149.5)
            return torch.rad2deg(pitchs).to(self.device)
            
        images = data["image"]
        labels = data["label"]

        if self.opt.network in 'deepcalib':
            #with torch.cuda.amp.autocast():
            (pred_focal, pred_distor) = self.net(images)
            
            net_h, net_w = self.opt.load_size
            img_h, img_w = self.opt.img_size
            ratio = net_h/img_h
            focal_start = 50. * ratio
            focal_end = 500. * ratio
            classes_focal = torch.linspace(focal_start, focal_end, 46, device=self.device)
            classes_distortion = torch.linspace(0., 1.2, 61, device=self.device)

            focal_CE, distor_CE = self.cross_entropy(F.softmax(pred_focal, dim=1), labels["focal_label"].long()), self.cross_entropy(F.softmax(pred_distor, dim=1), labels["distor_label"].long())
            self.losses['focal'] = { "CE": focal_CE }
            self.losses['distor'] = { "CE": distor_CE }
            
            if self.opt.phase == "train":
                self.optimizer.zero_grad()
                loss = self.losses['focal']["CE"]+self.losses['distor']["CE"]
                loss.backward()
                self.optimizer.step()
            
            pred_focal = classes_focal[torch.argmax(pred_focal, dim=1)]
            pred_distor = classes_distortion[torch.argmax(pred_distor, dim=1)]
            pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * pred_focal)))
            return (pred_focal, pred_distor, pred_fov), None
        elif self.opt.network in 'perceptual':
            #with torch.cuda.amp.autocast():
            (pred_roll, pred_offset, pred_fov) = self.net(images)
            
            pred_roll_bin, pred_offset_bin, pred_fov_bin= torch.argmax(pred_roll, 1), torch.argmax(pred_offset, 1), torch.argmax(pred_fov, 1)
            slope_bin_edges, offset_bin_edges, fov_bin_edges = self.bins["slope_bins"], self.bins["offset_bins"], torch.linspace(33., 143., 257)
            pred_roll_vals = torch.rad2deg(bin2Val(pred_roll_bin, slope_bin_edges))
            pred_offset_vals = bin2Val(pred_offset_bin, offset_bin_edges)
            pred_fov_vals = bin2Val(pred_fov_bin, fov_bin_edges)
            net_h, net_w = self.opt.load_size
            img_h, img_w = self.opt.img_size
            pred_pitch_vals = offsetToPitch(pred_roll_vals, pred_offset_vals, pred_fov_vals, img_h)
            
            KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), labels["slope_label"].float())
            KL_div_offset = self.KL_div(F.log_softmax(pred_offset, dim=1), labels["offset_label"].float())
            KL_div_fov = self.KL_div(F.log_softmax(pred_fov, dim=1), labels["fov_label"].float())
            self.losses['roll'] = { "KL_div": KL_div_roll}
            self.losses['offset'] = { "KL_div": KL_div_offset}
            self.losses['fov'] = { "KL_div": KL_div_fov}
            if self.opt.phase == "train":
                self.optimizer.zero_grad()
                loss = self.losses['roll']["KL_div"]+self.losses['offset']["KL_div"]+self.losses['fov']["KL_div"]
                loss.backward()
                self.optimizer.step()
            return (pred_roll_vals, pred_offset_vals, pred_fov_vals, pred_pitch_vals), None
        elif self.opt.network in 'deepfocal':
            (pred_fov) = self.net(images)
            self.losses['fov'] = { "MSE": self.MSE(pred_fov.squeeze(), labels["fov_deg"].float()) }
            if self.opt.phase == "train":
                self.optimizer.zero_grad()
                loss = self.losses['fov']["MSE"]
                loss.backward()
                self.optimizer.step()
            return (pred_fov), None
        elif self.opt.network in 'basic':
            (pred_roll, pred_pitch, pred_focal, pred_distor) = self.net(images)

            extrinsic_vec = torch.linspace(-15., 15., 256, device='cuda')
            SORD_roll = util.encodeSORD_with_ref(labels["roll"] * 10, extrinsic_vec * 10)
            SORD_pitch = util.encodeSORD_with_ref(labels["pitch"] * 10, extrinsic_vec * 10) 

            net_h, net_w = self.opt.load_size
            img_h, img_w = self.opt.img_size
            ratio = net_h/img_h
            focal_start = 50. * ratio
            focal_end = 500. * ratio
            focal_vec = torch.linspace(focal_start, focal_end, 256, device='cuda')
            SORD_focal = util.encodeSORD_with_ref(labels["focal"], focal_vec)

            distor_vec = torch.linspace(0, 1.0, 256, device='cuda') 
            SORD_distor = util.encodeSORD_with_ref(labels["distor"] * 100, distor_vec * 100)
            
            KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), SORD_roll)
            KL_div_pitch = self.KL_div(F.log_softmax(pred_pitch, dim=1), SORD_pitch)
            KL_div_focal = self.KL_div(F.log_softmax(pred_focal, dim=1), SORD_focal)
            KL_div_distor = self.KL_div(F.log_softmax(pred_distor, dim=1), SORD_distor)
            # multi_CE_x = self.multi_CE(pred_x, SORD_labels[0])
            # multi_CE_y = self.multi_CE(pred_y, SORD_labels[1])

            w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor = self.weighted_sum((pred_roll, pred_pitch, pred_focal, pred_distor), (extrinsic_vec, extrinsic_vec, focal_vec, distor_vec))
            pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * w_sum_focal)))

            RMSE_roll = torch.sqrt(self.MSE(w_sum_roll, labels["roll"]))
            RMSE_pitch = torch.sqrt(self.MSE(w_sum_pitch, labels["pitch"]))
            RMSE_focal = torch.sqrt(self.MSE(w_sum_focal, labels["focal"]))
            RMSE_distor = torch.sqrt(self.MSE(w_sum_distor, labels["distor"]))

            self.losses['roll'] = { "KL_Div": KL_div_roll, "RMSE": RMSE_roll }
            self.losses['pitch'] = { "KL_Div": KL_div_pitch, "RMSE": RMSE_pitch }
            self.losses['focal'] = { "KL_Div": KL_div_focal, "RMSE": RMSE_focal }
            self.losses['distor'] = { "KL_Div": KL_div_distor, "RMSE": RMSE_distor }


            loss_roll = self.opt.ce_w * self.losses['roll']['KL_Div'] + self.opt.mse_w * self.losses['roll']['RMSE']
            loss_pitch = self.opt.ce_w * self.losses['pitch']['KL_Div'] + self.opt.mse_w * self.losses['pitch']['RMSE']
            loss_focal = self.opt.ce_w * self.losses['focal']['KL_Div'] + self.opt.mse_w * self.losses['focal']['RMSE']
            loss_distor = self.opt.ce_w * self.losses['distor']['KL_Div'] + self.opt.mse_w * self.losses['distor']['RMSE']

            loss = loss_roll + loss_pitch + loss_focal + loss_distor

            if self.opt.phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            preds = (w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor, pred_fov)

            return preds, None
        elif 'single' in self.opt.network:
            preds = None
            edges = data["edge"]
            semantic = data["semantic"]

            if 'edgeattentioncalib' in self.opt.network:
                #with torch.cuda.amp.autocast():
                net_h, net_w = self.opt.load_size
                img_h, img_w = self.opt.img_size
                ratio = net_h/img_h
                focal_start = 50. * ratio
                focal_end = 500. * ratio

                (pred_roll, pred_pitch, pred_focal, pred_distor) = self.net(images, edges, edge_weight_map=data["weight_map"])
                
                extrinsic_vec = torch.linspace(-15., 15., 256, device=self.device)
                focal_vec = torch.linspace(focal_start, focal_end, 256, device=self.device)
                distor_vec = torch.linspace(0, 1.0, 256, device=self.device) 

                if self.opt.non_SORD == True:
                    oh_roll = util.encode(labels["roll"], extrinsic_vec)
                    oh_pitch = util.encode(labels["pitch"], extrinsic_vec)
                    oh_focal = util.encode(labels["focal"], focal_vec)
                    oh_distor = util.encode(labels["distor"], distor_vec)

                    KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), oh_roll)
                    KL_div_pitch = self.KL_div(F.log_softmax(pred_pitch, dim=1), oh_pitch)
                    KL_div_focal = self.KL_div(F.log_softmax(pred_focal, dim=1), oh_focal)
                    KL_div_distor = self.KL_div(F.log_softmax(pred_distor, dim=1), oh_distor)        
                else:
                    SORD_roll = util.encodeSORD_with_ref(labels["roll"], extrinsic_vec)
                    SORD_pitch = util.encodeSORD_with_ref(labels["pitch"], extrinsic_vec) 
                    SORD_focal = util.encodeSORD_with_ref(labels["focal"], focal_vec)
                    #SORD_distor = util.encodeSORD_with_ref(labels["distor"], distor_vec)
                    oh_distor = util.encode(labels["distor"], distor_vec)
                
                    KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), SORD_roll)
                    KL_div_pitch = self.KL_div(F.log_softmax(pred_pitch, dim=1), SORD_pitch)
                    KL_div_focal = self.KL_div(F.log_softmax(pred_focal, dim=1), SORD_focal)
                    KL_div_distor = self.KL_div(F.log_softmax(pred_distor, dim=1), oh_distor)

                w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor = self.weighted_sum((pred_roll, pred_pitch, pred_focal, pred_distor), (extrinsic_vec, extrinsic_vec, focal_vec, distor_vec))
                pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * w_sum_focal)))

                RMSE_roll = torch.sqrt(self.MSE(w_sum_roll, labels["roll"]))
                RMSE_pitch = torch.sqrt(self.MSE(w_sum_pitch, labels["pitch"]))
                RMSE_focal = torch.sqrt(self.MSE(w_sum_focal, labels["focal"]))
                RMSE_distor = torch.sqrt(self.MSE(w_sum_distor, labels["distor"]))

                self.losses['roll'] = { "KL_Div": KL_div_roll, "RMSE": RMSE_roll }
                self.losses['pitch'] = { "KL_Div": KL_div_pitch, "RMSE": RMSE_pitch }
                self.losses['focal'] = { "KL_Div": KL_div_focal, "RMSE": RMSE_focal }
                self.losses['distor'] = { "KL_Div": KL_div_distor, "RMSE": RMSE_distor }


                loss_roll = self.opt.ce_w * self.losses['roll']['KL_Div'] + self.opt.mse_w * self.losses['roll']['RMSE']
                loss_pitch = self.opt.ce_w * self.losses['pitch']['KL_Div'] + self.opt.mse_w * self.losses['pitch']['RMSE']
                loss_focal = self.opt.ce_w * self.losses['focal']['KL_Div'] + self.opt.mse_w * self.losses['focal']['RMSE']
                loss_distor = self.opt.ce_w * self.losses['distor']['KL_Div'] + self.opt.mse_w * self.losses['distor']['RMSE']
                loss_estimator = loss_roll + loss_pitch + loss_focal + loss_distor

                preds = (w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor, pred_fov)
                if self.opt.phase == "train":
                    self.optimizer_estimator.zero_grad()
                    loss_estimator.backward()
                    self.optimizer_estimator.step()
                    #self.scaler.scale(loss_estimator).backward()
                    #self.scaler.step(self.optimizer_estimator)
                    #self.scaler.update()
                return preds, None

            elif 'perceptual' in self.opt.network:
                #with torch.cuda.amp.autocast():
                (pred_roll, pred_offset, pred_fov) = self.net(images, edges, edge_weight_map=data["weight_map"])
        
                pred_roll_bin, pred_offset_bin, pred_fov_bin = torch.argmax(pred_roll, 1), torch.argmax(pred_offset, 1), torch.argmax(pred_fov, 1)
                slope_bin_edges, offset_bin_edges, fov_bin_edges = self.bins["slope_bins"], self.bins["offset_bins"], torch.linspace(33., 143., 257)
                pred_roll_vals = torch.rad2deg(bin2Val(pred_roll_bin, slope_bin_edges))
                pred_offset_vals = bin2Val(pred_offset_bin, offset_bin_edges)
                pred_fov_vals = bin2Val(pred_fov_bin, fov_bin_edges)
                net_h, net_w = self.opt.load_size
                img_h, img_w = self.opt.img_size
                pred_pitch_vals = offsetToPitch(pred_roll_vals, pred_offset_vals, pred_fov_vals, img_h)
                
                KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), labels["slope_label"].float())
                KL_div_offset = self.KL_div(F.log_softmax(pred_offset, dim=1), labels["offset_label"].float())
                KL_div_fov = self.KL_div(F.log_softmax(pred_fov, dim=1), labels["fov_label"].float())
                self.losses['roll'] = { "KL_div": KL_div_roll}
                self.losses['offset'] = { "KL_div": KL_div_offset}
                self.losses['fov'] = { "KL_div": KL_div_fov}

                loss_estimator = self.losses['roll']["KL_div"]+self.losses['offset']["KL_div"]+self.losses['fov']["KL_div"]
                preds = (pred_roll_vals, pred_offset_vals, pred_fov_vals, pred_pitch_vals)
            elif 'deepfocal' in self.opt.network:
                with torch.cuda.amp.autocast():
                    (pred_fov) = self.net(images, edges, edge_weight_map=data["weight_map"])
                self.losses['fov'] = { "MSE": self.MSE(pred_fov.squeeze(), labels["fov_deg"].float()) }

                loss_estimator = self.losses['fov']["MSE"]
                preds = (pred_fov)
            elif 'deepcalib' in self.opt.network:
                with torch.cuda.amp.autocast():
                    (pred_focal, pred_distor) = self.net(images, edges, edge_weight_map=data["weight_map"])

                net_h, net_w = self.opt.load_size
                img_h, img_w = self.opt.img_size
                ratio = net_h/img_h
                focal_start = 50. * ratio
                focal_end = 500. * ratio
                classes_focal = torch.linspace(focal_start, focal_end, 46, device=self.device)
                classes_distortion = torch.linspace(0., 1.2, 61, device=self.device)
    
                focal_CE, distor_CE = self.cross_entropy(F.softmax(pred_focal, dim=1), labels["focal_label"].long()), self.cross_entropy(F.softmax(pred_distor, dim=1), labels["distor_label"].long())
                self.losses['focal'] = { "CE": focal_CE }
                self.losses['distor'] = { "CE": distor_CE }
                
                loss_estimator = self.losses['focal']["CE"]+self.losses['distor']["CE"]
                
                pred_focal = classes_focal[torch.argmax(pred_focal, dim=1)]
                pred_distor = classes_distortion[torch.argmax(pred_distor, dim=1)]
                pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * pred_focal)))
                preds = (pred_focal, pred_distor, pred_fov)

            if self.opt.phase == "train":
                self.optimizer_estimator.zero_grad()
                loss_estimator.backward()
                self.optimizer_estimator.step()
                #self.scaler.scale(loss_estimator).backward()
                #self.scaler.step(self.optimizer_estimator)
                #self.scaler.update()
            return preds, None
        
        elif 'simple' in self.opt.network:
            preds = None
            #with torch.cuda.amp.autocast():
            (pred_roll, pred_pitch, pred_focal, pred_distor) = self.net(images)

            net_h, net_w = self.opt.load_size
            img_h, img_w = self.opt.img_size
            ratio = net_h/img_h
            focal_start = 50. * ratio
            focal_end = 500. * ratio
            
            extrinsic_vec = torch.linspace(-15., 15., 256, device=self.device)
            focal_vec = torch.linspace(focal_start, focal_end, 256, device=self.device)
            distor_vec = torch.linspace(0, 1.0, 256, device=self.device) 

            if self.opt.non_SORD == True:
                oh_roll = util.encode(labels["roll"], extrinsic_vec)
                oh_pitch = util.encode(labels["pitch"], extrinsic_vec)
                oh_focal = util.encode(labels["focal"], focal_vec)
                oh_distor = util.encode(labels["distor"], distor_vec)

                KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), oh_roll)
                KL_div_pitch = self.KL_div(F.log_softmax(pred_pitch, dim=1), oh_pitch)
                KL_div_focal = self.KL_div(F.log_softmax(pred_focal, dim=1), oh_focal)
                KL_div_distor = self.KL_div(F.log_softmax(pred_distor, dim=1), oh_distor)        
            else:
                SORD_roll = util.encodeSORD_with_ref(labels["roll"], extrinsic_vec)
                SORD_pitch = util.encodeSORD_with_ref(labels["pitch"], extrinsic_vec) 
                SORD_focal = util.encodeSORD_with_ref(labels["focal"], focal_vec)
                #SORD_distor = util.encodeSORD_with_ref(labels["distor"], distor_vec)
                oh_distor = util.encode(labels["distor"], distor_vec)
            
                KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), SORD_roll)
                KL_div_pitch = self.KL_div(F.log_softmax(pred_pitch, dim=1), SORD_pitch)
                KL_div_focal = self.KL_div(F.log_softmax(pred_focal, dim=1), SORD_focal)
                KL_div_distor = self.KL_div(F.log_softmax(pred_distor, dim=1), oh_distor)        

            w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor = self.weighted_sum((pred_roll, pred_pitch, pred_focal, pred_distor), (extrinsic_vec, extrinsic_vec, focal_vec, distor_vec))
            pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * w_sum_focal)))
            
            RMSE_roll = torch.sqrt(self.MSE(w_sum_roll, labels["roll"]))
            RMSE_pitch = torch.sqrt(self.MSE(w_sum_pitch, labels["pitch"]))
            RMSE_focal = torch.sqrt(self.MSE(w_sum_focal, labels["focal"]))
            RMSE_distor = torch.sqrt(self.MSE(w_sum_distor, labels["distor"]))

            self.losses['roll'] = { "KL_Div": KL_div_roll, "RMSE": RMSE_roll }
            self.losses['pitch'] = { "KL_Div": KL_div_pitch, "RMSE": RMSE_pitch }
            self.losses['focal'] = { "KL_Div": KL_div_focal, "RMSE": RMSE_focal }
            self.losses['distor'] = { "KL_Div": KL_div_distor, "RMSE": RMSE_distor }


            loss_roll = self.opt.ce_w * self.losses['roll']['KL_Div'] + self.opt.mse_w * self.losses['roll']['RMSE']
            loss_pitch = self.opt.ce_w * self.losses['pitch']['KL_Div'] + self.opt.mse_w * self.losses['pitch']['RMSE']
            loss_focal = self.opt.ce_w * self.losses['focal']['KL_Div'] + self.opt.mse_w * self.losses['focal']['RMSE']
            loss_distor = self.opt.ce_w * self.losses['distor']['KL_Div'] + self.opt.mse_w * self.losses['distor']['RMSE']
            loss_estimator = loss_roll + loss_pitch + loss_focal + loss_distor

            preds = (w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor, pred_fov)
                
            if self.opt.phase == "train":
                self.optimizer.zero_grad()
                loss_estimator.backward()
                self.optimizer.step()
                #self.scaler.scale(loss_estimator).backward()
                #self.scaler.step(self.optimizer)
                #self.scaler.update()
            return preds, None

        elif 'dual' in self.opt.network:
            preds = None
            edges = data["edge"]
                    
            # estimator backprop
            if self.opt.phase == "train":
                self.optimizer_estimator.zero_grad()
                    
            semantic = data["semantic"]
            images_with_semantic = torch.cat((images, semantic), axis=1)
            edges_with_semantic = torch.cat((edges, semantic), axis=1)

            if 'edgeattentioncalib' in self.opt.network:
                (pred_roll, pred_pitch, pred_focal, pred_distor, edge_pred_roll, edge_pred_pitch, edge_pred_focal, edge_pred_distor) = self.net(images_with_semantic, edges_with_semantic, edge_weight_map=data["weight_map"])

                extrinsic_vec = torch.linspace(-15., 15., 256, device=self.device)
                SORD_roll = util.encodeSORD_with_ref(labels["roll"] * 10, extrinsic_vec * 10)
                SORD_pitch = util.encodeSORD_with_ref(labels["pitch"] * 10, extrinsic_vec * 10) 
                # SORD_yaw = util.encodeSORD_with_ref(labels["yaw"] * 10, extrinsic_vec * 10)

                net_h, net_w = self.opt.load_size
                img_h, img_w = self.opt.img_size
                ratio = net_h/img_h
                focal_start = 50. * ratio
                focal_end = 500. * ratio
                focal_vec = torch.linspace(focal_start, focal_end, 256, device=self.device)
                SORD_focal = util.encodeSORD_with_ref(labels["focal"], focal_vec)

                distor_vec = torch.linspace(0, 1.0, 256, device=self.device) 
                SORD_distor = util.encodeSORD_with_ref(labels["distor"] * 100, distor_vec * 100)
            
                KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), SORD_roll)
                KL_div_pitch = self.KL_div(F.log_softmax(pred_pitch, dim=1), SORD_pitch)
                KL_div_focal = self.KL_div(F.log_softmax(pred_focal, dim=1), SORD_focal)
                KL_div_distor = self.KL_div(F.log_softmax(pred_distor, dim=1), SORD_distor)

                w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor = self.weighted_sum((pred_roll, pred_pitch, pred_focal, pred_distor), (extrinsic_vec, extrinsic_vec, focal_vec, distor_vec))
                pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * w_sum_focal)))

                RMSE_roll = torch.sqrt(self.MSE(w_sum_roll, labels["roll"]))
                RMSE_pitch = torch.sqrt(self.MSE(w_sum_pitch, labels["pitch"]))
                RMSE_focal = torch.sqrt(self.MSE(w_sum_focal, labels["focal"]))
                RMSE_distor = torch.sqrt(self.MSE(w_sum_distor, labels["distor"]))

                self.losses['roll'] = { "KL_Div": KL_div_roll, "RMSE": RMSE_roll }
                self.losses['pitch'] = { "KL_Div": KL_div_pitch, "RMSE": RMSE_pitch }
                self.losses['focal'] = { "KL_Div": KL_div_focal, "RMSE": RMSE_focal }
                self.losses['distor'] = { "KL_Div": KL_div_distor, "RMSE": RMSE_distor }


                loss_roll = self.opt.ce_w * self.losses['roll']['KL_Div'] + self.opt.mse_w * self.losses['roll']['RMSE']
                loss_pitch = self.opt.ce_w * self.losses['pitch']['KL_Div'] + self.opt.mse_w * self.losses['pitch']['RMSE']
                loss_focal = self.opt.ce_w * self.losses['focal']['KL_Div'] + self.opt.mse_w * self.losses['focal']['RMSE']
                loss_distor = self.opt.ce_w * self.losses['distor']['KL_Div'] + self.opt.mse_w * self.losses['distor']['RMSE']
                loss_estimator = loss_roll + loss_pitch + loss_focal + loss_distor

                ####### for edge #######
                edge_KL_div_roll = self.KL_div(F.log_softmax(edge_pred_roll, dim=1), SORD_roll)
                edge_KL_div_pitch = self.KL_div(F.log_softmax(edge_pred_pitch, dim=1), SORD_pitch)
                edge_KL_div_focal = self.KL_div(F.log_softmax(edge_pred_focal, dim=1), SORD_focal)
                edge_KL_div_distor = self.KL_div(F.log_softmax(edge_pred_distor, dim=1), SORD_distor)

                edge_w_sum_roll, edge_w_sum_pitch, edge_w_sum_focal, edge_w_sum_distor = self.weighted_sum((edge_pred_roll, edge_pred_pitch, edge_pred_focal, edge_pred_distor), (extrinsic_vec, extrinsic_vec, focal_vec, distor_vec))
                edge_pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * edge_w_sum_focal)))

                edge_RMSE_roll = torch.sqrt(self.MSE(edge_w_sum_roll, labels["roll"]))
                edge_RMSE_pitch = torch.sqrt(self.MSE(edge_w_sum_pitch, labels["pitch"]))
                edge_RMSE_focal = torch.sqrt(self.MSE(edge_w_sum_focal, labels["focal"]))
                edge_RMSE_distor = torch.sqrt(self.MSE(edge_w_sum_distor, labels["distor"]))

                self.losses['edge_roll'] = { "KL_Div": edge_KL_div_roll, "RMSE": edge_RMSE_roll }
                self.losses['edge_pitch'] = { "KL_Div": edge_KL_div_pitch, "RMSE": edge_RMSE_pitch }
                self.losses['edge_focal'] = { "KL_Div": edge_KL_div_focal, "RMSE": edge_RMSE_focal }
                self.losses['edge_distor'] = { "KL_Div": edge_KL_div_distor, "RMSE": edge_RMSE_distor }


                edge_loss_roll = self.opt.ce_w * self.losses['edge_roll']['KL_Div'] + self.opt.mse_w * self.losses['edge_roll']['RMSE']
                edge_loss_pitch = self.opt.ce_w * self.losses['edge_pitch']['KL_Div'] + self.opt.mse_w * self.losses['edge_pitch']['RMSE']
                edge_loss_focal = self.opt.ce_w * self.losses['edge_focal']['KL_Div'] + self.opt.mse_w * self.losses['edge_focal']['RMSE']
                edge_loss_distor = self.opt.ce_w * self.losses['edge_distor']['KL_Div'] + self.opt.mse_w * self.losses['edge_distor']['RMSE']
                edge_loss_estimator = edge_loss_roll + edge_loss_pitch + edge_loss_focal + edge_loss_distor


                preds = (w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor, pred_fov, edge_w_sum_roll, edge_w_sum_pitch, edge_w_sum_focal, edge_w_sum_distor, edge_pred_fov)

                if self.opt.phase == "train":
                    edge_loss_estimator.backward()
                    self.optimizer_edge_estimator.step()
                    loss_estimator.backward()
                    self.optimizer_estimator.step()
            return preds, None


        # edgeattention or edgeattentioncalib
        elif 'edgeattention' in self.opt.network:
            preds = None
            real_A = images
            real_B = data["edge"]
            fake_B = self.net(real_A, mode="generator")

            if not self.opt.pretrained_gan:
                # discriminator backprop
                if self.opt.phase == "train":
                    self.set_requires_grad(self.net.netD, True)  # enable backprop for D
                    self.optimizer_D.zero_grad()
                # fake
                fake_AB =  torch.cat((real_A, fake_B), 1)
                pred_fake = self.net(fake_AB.detach(), mode="discriminator")
                loss_D_fake = self.criterionGAN(pred_fake, False)
                # real
                real_AB = torch.cat((real_A, real_B), 1)
                pred_real = self.net(real_AB, mode="discriminator")
                loss_D_real = self.criterionGAN(pred_real, True)
                # combine loss and calculate gradients
                self.losses["D"] = {"loss": (loss_D_fake + loss_D_real) * 0.5}
                if self.opt.phase == "train":
                    self.losses["D"]["loss"].backward()
                    self.optimizer_D.step()
                    for p in self.bin_gate_D:
                        p.data.clamp_(min=0, max=1)

                # generator backprop
                if self.opt.phase == "train":
                    self.set_requires_grad(self.net.netD, False) 
                    self.optimizer_G.zero_grad()
                # First, G(A) should fake the discriminator
                fake_AB =  torch.cat((real_A, fake_B), 1)
                pred_fake = self.net(fake_AB, mode="discriminator")
                loss_G_GAN = self.criterionGAN(pred_fake, True)
                # Second, G(A) = B
                loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1
                # combine loss and calculate gradients
                self.losses["G"] = {"loss": loss_G_GAN + loss_G_L1}
                if self.opt.phase == "train":
                    self.losses["G"]["loss"].backward()
                    self.optimizer_G.step()
                    for p in self.bin_gate_G:
                        p.data.clamp_(min=0, max=1)
                    
            # estimator backprop
            if not self.opt.only_gan:
                if self.opt.phase == "train":
                    self.optimizer_estimator.zero_grad()
                    
                semantic = data["semantic"]
                images_with_semantic = torch.cat((real_A, semantic), axis=1)

                if 'edgeattentioncalib' in self.opt.network:
                    # (pred_roll, pred_pitch, pred_yaw, pred_focal, pred_distor) = self.net(images_with_semantic, edge_weight_map=data["weight_map"], mode="estimation")
                    (pred_roll, pred_pitch, pred_focal, pred_distor) = self.net(images_with_semantic, edge_weight_map=data["weight_map"], mode="estimation")

                    extrinsic_vec = torch.linspace(-15., 15., 256, device=self.device)
                    SORD_roll = util.encodeSORD_with_ref(labels["roll"] * 10, extrinsic_vec * 10)
                    SORD_pitch = util.encodeSORD_with_ref(labels["pitch"] * 10, extrinsic_vec * 10) 
                    # SORD_yaw = util.encodeSORD_with_ref(labels["yaw"] * 10, extrinsic_vec * 10)

                    net_h, net_w = self.opt.load_size
                    img_h, img_w = self.opt.img_size
                    ratio = net_h/img_h
                    focal_start = 50. * ratio
                    focal_end = 500. * ratio
                    focal_vec = torch.linspace(focal_start, focal_end, 256, device=self.device)
                    SORD_focal = util.encodeSORD_with_ref(labels["focal"], focal_vec)

                    distor_vec = torch.linspace(0, 1.0, 256, device=self.device) 
                    SORD_distor = util.encodeSORD_with_ref(labels["distor"] * 100, distor_vec * 100)
                
                    KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), SORD_roll)
                    KL_div_pitch = self.KL_div(F.log_softmax(pred_pitch, dim=1), SORD_pitch)
                    # KL_div_yaw = self.KL_div(F.log_softmax(pred_yaw, dim=1), SORD_yaw)
                    KL_div_focal = self.KL_div(F.log_softmax(pred_focal, dim=1), SORD_focal)
                    KL_div_distor = self.KL_div(F.log_softmax(pred_distor, dim=1), SORD_distor)

                    # w_sum_roll, w_sum_pitch, w_sum_yaw, w_sum_focal, w_sum_distor = self.weighted_sum((pred_roll, pred_pitch, pred_yaw, pred_focal, pred_distor), (extrinsic_vec, extrinsic_vec, extrinsic_vec, focal_vec, distor_vec))
                    w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor = self.weighted_sum((pred_roll, pred_pitch, pred_focal, pred_distor), (extrinsic_vec, extrinsic_vec, focal_vec, distor_vec))
                    pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * w_sum_focal)))

                    RMSE_roll = torch.sqrt(self.MSE(w_sum_roll, labels["roll"]))
                    RMSE_pitch = torch.sqrt(self.MSE(w_sum_pitch, labels["pitch"]))
                    # RMSE_yaw = torch.sqrt(self.MSE(w_sum_yaw, labels["yaw"]))
                    RMSE_focal = torch.sqrt(self.MSE(w_sum_focal, labels["focal"]))
                    RMSE_distor = torch.sqrt(self.MSE(w_sum_distor, labels["distor"]))

                    self.losses['roll'] = { "KL_Div": KL_div_roll, "RMSE": RMSE_roll }
                    self.losses['pitch'] = { "KL_Div": KL_div_pitch, "RMSE": RMSE_pitch }
                    # self.losses['yaw'] = { "KL_Div": KL_div_yaw, "RMSE": RMSE_yaw }
                    self.losses['focal'] = { "KL_Div": KL_div_focal, "RMSE": RMSE_focal }
                    self.losses['distor'] = { "KL_Div": KL_div_distor, "RMSE": RMSE_distor }


                    loss_roll = self.opt.ce_w * self.losses['roll']['KL_Div'] + self.opt.mse_w * self.losses['roll']['RMSE']
                    loss_pitch = self.opt.ce_w * self.losses['pitch']['KL_Div'] + self.opt.mse_w * self.losses['pitch']['RMSE']
                    # loss_yaw = self.opt.ce_w * self.losses['yaw']['KL_Div'] + self.opt.mse_w * self.losses['yaw']['RMSE']
                    loss_focal = self.opt.ce_w * self.losses['focal']['KL_Div'] + self.opt.mse_w * self.losses['focal']['RMSE']
                    loss_distor = self.opt.ce_w * self.losses['distor']['KL_Div'] + self.opt.mse_w * self.losses['distor']['RMSE']

                    loss_estimator = loss_roll + loss_pitch + loss_focal + loss_distor
                    preds = (w_sum_roll, w_sum_pitch, w_sum_focal, w_sum_distor, pred_fov)
                elif 'perceptual' in self.opt.network:
                    (pred_roll, pred_offset, pred_fov) = self.net(images_with_semantic, edge_weight_map=data["weight_map"], mode="estimation")
            
                    pred_roll_bin, pred_offset_bin, pred_fov_bin = torch.argmax(pred_roll, 1), torch.argmax(pred_offset, 1), torch.argmax(pred_fov, 1)
                    slope_bin_edges, offset_bin_edges, fov_bin_edges = self.bins["slope_bins"], self.bins["offset_bins"], torch.linspace(33., 143., 257)
                    pred_roll_vals = torch.rad2deg(bin2Val(pred_roll_bin, slope_bin_edges))
                    pred_offset_vals = bin2Val(pred_offset_bin, offset_bin_edges)
                    pred_fov_vals = bin2Val(pred_fov_bin, fov_bin_edges)
                    net_h, net_w = self.opt.load_size
                    img_h, img_w = self.opt.img_size
                    pred_pitch_vals = offsetToPitch(pred_roll_vals, pred_offset_vals, pred_fov_vals, img_h)
                    
                    KL_div_roll = self.KL_div(F.log_softmax(pred_roll, dim=1), labels["slope_label"].float())
                    KL_div_offset = self.KL_div(F.log_softmax(pred_offset, dim=1), labels["offset_label"].float())
                    KL_div_fov = self.KL_div(F.log_softmax(pred_fov, dim=1), labels["fov_label"].float())
                    self.losses['roll'] = { "KL_div": KL_div_roll}
                    self.losses['offset'] = { "KL_div": KL_div_offset}
                    self.losses['fov'] = { "KL_div": KL_div_fov}

                    loss_estimator = self.losses['roll']["KL_div"]+self.losses['offset']["KL_div"]+self.losses['fov']["KL_div"]
                    preds = (pred_roll_vals, pred_offset_vals, pred_fov_vals, pred_pitch_vals)
                elif 'deepfocal' in self.opt.network:
                    (pred_fov) = self.net(images_with_semantic, edge_weight_map=data["weight_map"], mode="estimation")
                    self.losses['fov'] = { "MSE": self.MSE(pred_fov.squeeze(), labels["fov_deg"].float()) }

                    loss_estimator = self.losses['fov']["MSE"]
                    preds = (pred_fov)
                elif 'deepcalib' in self.opt.network:
                    (pred_focal, pred_distor) = self.net(images_with_semantic, edge_weight_map=data["weight_map"], mode="estimation")

                    net_h, net_w = self.opt.load_size
                    img_h, img_w = self.opt.img_size
                    ratio = net_h/img_h
                    focal_start = 50. * ratio
                    focal_end = 500. * ratio
                    classes_focal = torch.linspace(focal_start, focal_end, 46, device=self.device)
                    classes_distortion = torch.linspace(0., 1.2, 61, device=self.device)
        
                    focal_CE, distor_CE = self.cross_entropy(F.softmax(pred_focal, dim=1), labels["focal_label"].long()), self.cross_entropy(F.softmax(pred_distor, dim=1), labels["distor_label"].long())
                    self.losses['focal'] = { "CE": focal_CE }
                    self.losses['distor'] = { "CE": distor_CE }
                    
                    loss_estimator = self.losses['focal']["CE"]+self.losses['distor']["CE"]
                    
                    pred_focal = classes_focal[torch.argmax(pred_focal, dim=1)]
                    pred_distor = classes_distortion[torch.argmax(pred_distor, dim=1)]
                    pred_fov = torch.rad2deg(2 * torch.arctan(net_w / (2 * pred_focal)))
                    preds = (pred_focal, pred_distor, pred_fov)
                  
                if self.opt.phase == "train":
                    loss_estimator.backward()
                    self.optimizer_estimator.step()

            return preds, fake_B
        else:
            (pred_x, pred_y) = self.net(images)
            SORD_labels = util.encodeSORD(labels)
            KL_div_x = self.KL_div(F.log_softmax(pred_x, dim=1), SORD_labels[0].float())
            KL_div_y = self.KL_div(F.log_softmax(pred_y, dim=1), SORD_labels[1].float())
            # multi_CE_x = self.multi_CE(pred_x, SORD_labels[0])
            # multi_CE_y = self.multi_CE(pred_y, SORD_labels[1])

            w_sum_x, w_sum_y = self.weighted_sum((pred_x, pred_y))
            RMSE_x, RMSE_y = torch.sqrt(self.MSE(w_sum_x, labels[0].float())), torch.sqrt(self.MSE(w_sum_y, labels[1].float()))

            self.losses['fovx'] = { "KL_Div": KL_div_x, "RMSE": RMSE_x }
            self.losses['fovy'] = { "KL_Div": KL_div_y, "RMSE": RMSE_y }

        if self.opt.phase == "train":
            self.optimizer.zero_grad()
            loss_x = self.opt.ce_w * self.losses['fovx']['KL_Div'] + self.opt.mse_w * self.losses['fovx']['RMSE']
            loss_y = self.opt.ce_w * self.losses['fovy']['KL_Div'] + self.opt.mse_w * self.losses['fovy']['RMSE']
            loss = loss_x+loss_y
            loss.backward()
            self.optimizer().step()

        return (pred_x, pred_y), None


    def get_current_losses(self):
        return self.losses

    def comput_metric(self, preds, labels):          
        metrics = {}
        
        if self.opt.only_gan:
            return metrics

        if 'deepfocal' in self.opt.network:
            MSE_fov = self.MSE(preds.squeeze(), labels["fov_deg"].float()).item()
            
            metrics['fov'] = {"RMSE": np.sqrt(MSE_fov)}
        elif 'deepcalib' in self.opt.network:
            pred_focal = preds[0]
            pred_distor = preds[1]
            pred_fov = preds[2]
            MSE_focal, MSE_distor, MSE_fov = self.MSE(pred_focal, labels["focal"]).item(), self.MSE(pred_distor, labels["distor"]).item(), self.MSE(pred_fov, labels["fov_deg"]).item()
            
            metrics['focal'] = {"RMSE": np.sqrt(MSE_focal)}
            metrics['distor'] = {"RMSE": np.sqrt(MSE_distor)}
            metrics['fov'] = {"RMSE": np.sqrt(MSE_fov)}
        elif "dual" in self.opt.network:
            MSE_roll, MSE_pitch, MSE_focal, MSE_fov, MSE_distor = self.MSE(preds[0], labels["roll"]).item(), self.MSE(preds[1], labels["pitch"]).item(), self.MSE(preds[2], labels["focal"]).item(), self.MSE(preds[4], labels["fov_deg"]).item(), self.MSE(preds[3], labels["distor"]).item()

            metrics['roll'] = {"RMSE": np.sqrt(MSE_roll)}
            metrics['pitch'] = {"RMSE": np.sqrt(MSE_pitch)}
            metrics['focal'] = {"RMSE": np.sqrt(MSE_focal)}
            metrics['fov'] = {"RMSE": np.sqrt(MSE_fov)}
            metrics['distor'] = {"RMSE": np.sqrt(MSE_distor)}

            edge_MSE_roll, edge_MSE_pitch, edge_MSE_focal, edge_MSE_fov, edge_MSE_distor = self.MSE(preds[5], labels["roll"]).item(), self.MSE(preds[6], labels["pitch"]).item(), self.MSE(preds[7], labels["focal"]).item(), self.MSE(preds[9], labels["fov_deg"]).item(), self.MSE(preds[8], labels["distor"]).item()

            metrics['edge_roll'] = {"RMSE": np.sqrt(edge_MSE_roll)}
            metrics['edge_pitch'] = {"RMSE": np.sqrt(edge_MSE_pitch)}
            metrics['edge_focal'] = {"RMSE": np.sqrt(edge_MSE_focal)}
            metrics['edge_fov'] = {"RMSE": np.sqrt(edge_MSE_fov)}
            metrics['edge_distor'] = {"RMSE": np.sqrt(edge_MSE_distor)}

            ensemble_MSE_roll, ensemble_MSE_pitch, ensemble_MSE_focal, ensemble_MSE_fov, ensemble_MSE_distor = self.MSE((preds[0]+preds[5])/2, labels["roll"]).item(), self.MSE((preds[1]+preds[6])/2, labels["pitch"]).item(), self.MSE((preds[2]+preds[7])/2, labels["focal"]).item(), self.MSE((preds[4]+preds[9])/2, labels["fov_deg"]).item(), self.MSE((preds[3]+preds[8])/2, labels["distor"]).item()

            metrics['ensemble_roll'] = {"RMSE": np.sqrt(ensemble_MSE_roll)}
            metrics['ensemble_pitch'] = {"RMSE": np.sqrt(ensemble_MSE_pitch)}
            metrics['ensemble_focal'] = {"RMSE": np.sqrt(ensemble_MSE_focal)}
            metrics['ensemble_fov'] = {"RMSE": np.sqrt(ensemble_MSE_fov)}
            metrics['ensemble_distor'] = {"RMSE":np.sqrt(ensemble_MSE_distor)}
        elif "edgeattentioncalib" in self.opt.network or self.opt.network in "basic" or self.opt.network in "simple":       
            MSE_roll, MSE_pitch, MSE_focal, MSE_fov, MSE_distor = self.MSE(preds[0], labels["roll"]).item(), self.MSE(preds[1], labels["pitch"]).item(), self.MSE(preds[2], labels["focal"]).item(), self.MSE(preds[4], labels["fov_deg"]).item(), self.MSE(preds[3], labels["distor"]).item()

            metrics['roll'] = {"RMSE": np.sqrt(MSE_roll)}
            metrics['pitch'] = {"RMSE": np.sqrt(MSE_pitch)}
            metrics['focal'] = {"RMSE": np.sqrt(MSE_focal)}
            metrics['fov'] = {"RMSE": np.sqrt(MSE_fov)}
            metrics['distor'] = {"RMSE": np.sqrt(MSE_distor)}
        elif 'perceptual' in self.opt.network:     
            MSE_roll, MSE_offset, MSE_fov, MSE_pitch = self.MSE(preds[0], labels["roll"]).item(), self.MSE(preds[1], labels["offset"]).item(), self.MSE(preds[2], labels["fov_deg"]).item(), self.MSE(preds[3], labels["pitch"]).item()
            
            metrics['roll'] = {"RMSE": np.sqrt(MSE_roll)}
            metrics['pitch'] = {"RMSE": np.sqrt(MSE_pitch)}
            metrics['offset'] = {"RMSE": np.sqrt(MSE_offset)}
            metrics['fov'] = {"RMSE": np.sqrt(MSE_fov)}
        else:    
            pred_x = preds[0]
            pred_y = preds[1]
            w_sum_x, w_sum_y = self.weighted_sum((pred_x, pred_y))
            
            MSE_x, MSE_y = self.MSE(w_sum_x, labels[0].float()).item(), self.MSE(w_sum_y, labels[1].float()).item()

            pred_x = torch.argmax(preds[0], 1)
            pred_y = torch.argmax(preds[1], 1)
            acc_cnt_x = (pred_x == torch.round(labels[0].float())).sum().item()
            acc_cnt_y = (pred_y == torch.round(labels[1].float())).sum().item()

            metrics['fovx'] = {"acc": acc_cnt_x, "RMSE": np.sqrt(MSE_x)}
            metrics['fovy'] = {"acc": acc_cnt_y, "RMSE": np.sqrt(MSE_y)}

        return metrics


    ############################################################################
    # Private helper methods
    ############################################################################
    
    def weighted_sum(self, preds, ref_vec=None):
        batch_size = preds[0].shape[0]
        ref_vec = torch.linspace(0, 180., 181, device=self.device).repeat(len(preds),1) if ref_vec == None else ref_vec

        w_sums = []
        for i, pred in enumerate(preds):
            w_sum = torch.sum(torch.mul(self.softmax(pred), ref_vec[i]), 1)
            w_sums.append(w_sum.float())

        return w_sums
        # w_sum_x = torch.sum(torch.mul(self.softmax(preds[0]), deg_vec), 1)
        # w_sum_y = torch.sum(torch.mul(self.softmax(preds[1]), deg_vec), 1)

        # return w_sum_x, w_sum_y
        
    
    def sum_squaredError(self, preds, labels):
        sub = torch.subtract(preds, labels)
        squared = torch.square(sub)
        return torch.sum(squared)


    def save(self, epoch):
        util.save_network(self.net, self.optimizers[0], 'Net', epoch, self.opt, self.schedulers[0], self.scaler)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

