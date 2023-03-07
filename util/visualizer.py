import os
import ntpath
import time
import json
from . import util
from . import html
import numpy as np
from torch.utils.tensorboard import SummaryWriter
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            if opt.phase == "train":
                self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, f"logs")
            elif opt.phase == "test":
                self.log_dir = os.path.join(opt.results_dir, opt.name, "logs", opt.which_epoch)
                os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain and opt.phase == "train":
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Strat (%s) ================\n' % now)
        else: 
            self.log_name = os.path.join(opt.results_dir, opt.name, f"{opt.which_epoch}_res.txt")
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Test results (%s) ================\n' % now)
                
            self.res_json_name = os.path.join(opt.results_dir, opt.name, f"{opt.which_epoch}_res.json")

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step, tag=""):
        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if self.tf_log: # show images in tensorboard output
            for label, image_numpy in visuals.items():
                image_numpy = util.put_text(image_numpy, label)
                # if len(image_numpy.shape) < 4:
                #     image_numpy = np.expand_dims(image_numpy, 0)
                image_numpy = np.transpose(image_numpy, (2, 0, 1))
                self.writer.add_image(f"{self.opt.phase}_{tag}/Epoch{epoch}", image_numpy, step)
            
        if self.use_html: # save images to a html file
            img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.7d.png' % (epoch, step))
            visuals_lst = []
            for label, image_numpy in visuals.items():
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                visuals_lst.append(image_numpy)
            image_cath = np.concatenate(visuals_lst, axis=0)
            util.save_image(image_cath, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.7d_%s_%d.png' % (n, step, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.7d.png' % (n, step)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # |visuals|: dictionary of images to display or save
    def display_current_tests(self, visuals, epoch, step):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if self.use_html: # save images to a html file
            img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.7d.png' % (epoch, step))
            visuals_lst = []
            for label, image_numpy in visuals.items():
                for i in range(len(image_numpy)):
                    img_path = 'epoch%.3d_iter%.7d_%s_%d.png' % (epoch, step, label, i)
                    util.save_image(image_numpy[i], img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.7d_%s_%d.png' % (n, step, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.7d.png' % (n, step)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = sum(value.values()).mean().float()
                self.tf.summary.scalar(tag, value.cpu().detach().numpy(), step)

    # errors: dictionary of error labels and values
    def plot_epoch_errors(self, tag, errors, step):
        if self.tf_log:
            self.writer.add_scalars(tag, errors, step)

    # errors: dictionary of error labels and values
    def plot_current_lr(self, name, lrs, step):
        if self.tf_log:
            for i, lr in enumerate(lrs):
                if len(lrs) > 1:
                    if i == 0:
                        tag_name = name+"/lr" 
                    elif i == 1:
                        tag_name = name+"/bin" 
                else:
                    tag_name = name

                self.writer.add_scalar(tag_name, lr, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        if epoch == -1:
            message = '(phase: %s, iters: %d, time: %.3f) ' % (self.opt.phase, i, t)
        else:
            message = '(phase: %s, epoch: %d, iters: %d, time: %.3f) ' % (self.opt.phase, epoch, i, t)

        message += str(errors)    
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    
    def write_log(self, msg):
        print(msg)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % msg)
            
    def write_results_json(self, res_json):
        with open(self.res_json_name, "w") as json_file:
            json.dump(res_json, json_file)


    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                t = util.tensor2label(t, self.opt.label_nc, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        visuals_lst = []
        image_name = '%s.png' % name
        save_path = os.path.join(image_dir, image_name)
        for label, image_numpy in visuals.items():
            visuals_lst.append(image_numpy)

        image_cath = np.concatenate(visuals_lst, axis=1)
        util.save_image(image_cath, save_path, create_dir=True)
