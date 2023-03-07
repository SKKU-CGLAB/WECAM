from models.networks.sync_batchnorm import DataParallelWithCallback
from models.dfov_model import DfovModel
import torch


class DfovTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.dfov_model = DfovModel(opt)
        if len(opt.gpu_ids) > 0:
            self.dfov_model = torch.nn.DataParallel(self.dfov_model, opt.gpu_ids)
            self.dfov_model_on_one_gpu = self.dfov_model.module
        else:
            self.dfov_model_on_one_gpu = self.dfov_model

        if opt.isTrain:
            self.optimizers, self.schedulers = \
                self.dfov_model_on_one_gpu.create_optimizers(opt)


    def run_one_step(self, data):
        if len(self.opt.gpu_ids) > 0:
            data["image"] = data["image"].cuda()
            for key, label in data['label'].items():
                data['label'][key] = label.cuda().float()
            # data["label"][0] = data["label"][0].cuda()
            # data["label"][1] = data["label"][1].cuda()
            if "edgeattention" in self.opt.network:
                data["edge"] = data["edge"].cuda()
                if not self.opt.only_gan:
                    data["semantic"] = data["semantic"].cuda()
                    data["weight_map"] = data["weight_map"].cuda()
        else:
            for key, label in data['label'].items():
                data['label'][key] = label.float()

        preds, _ = self.dfov_model(data)
        self.losses = self.dfov_model_on_one_gpu.get_current_losses()
        self.metrics = self.dfov_model_on_one_gpu.comput_metric(preds, data["label"])

        self.preds = preds
        self.edge = _

    def get_latest_losses(self):
        return {**self.losses}

    def get_latest_metrics(self):
        return {**self.metrics}

    def get_latest_inferenced(self):
        return self.preds

    def get_latest_generated(self):
        return self.edge

    def get_learning_rate(self):
        lrs = []
        for optimizer in self.optimizers:
            lr = []
            for param_group in optimizer.param_groups:
                lr.append(param_group['lr'])
            lrs.append(lr)

        return lrs

    def update_learning_rate(self, val_loss=None):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for i, scheduler in enumerate(self.schedulers):
            optimizer = self.optimizers[i]
            old_lr = []
            
            for param_group in optimizer.param_groups:
                old_lr.append(param_group['lr'])

            if self.opt.lr_policy == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            new_lr = []    
            for param_group in optimizer.param_groups:
                new_lr.append(param_group['lr'])
                
            print(f'update learning rate: {old_lr} -> {new_lr}')


    def save(self, epoch):
        self.dfov_model_on_one_gpu.save(epoch)


    # def print_networks(self):
    #     self.dfov_model_on_one_gpu.print_networks()
    