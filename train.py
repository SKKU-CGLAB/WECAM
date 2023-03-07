import sys
from collections import OrderedDict

import numpy as np

import data
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util.util import init_nested_dict
from trainers.dfov_trainer import DfovTrainer
from tqdm import tqdm
import torch

import random
import os
import shutil

'''
reproducible
'''
random_seed = 777

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(random_seed)
random.seed(random_seed)


# parse options
opt = TrainOptions().parse()
# create tool for visualization
visualizer = Visualizer(opt)
# print options to help debugging
visualizer.write_log(' '.join(sys.argv))

# load the dataset
tr_dataloader, _ = data.create_dataloader(opt)

if _ is not None:
    val_dataloader = _
else: 
    opt.phase = "val"
    val_dataloader, _ = data.create_dataloader(opt)
    opt.phase = "train"

# create tool for counting iterations
tr_dataset_len = len(tr_dataloader.dataset)
tr_iter_len = len(tr_dataloader)
iter_counter = IterationCounter(opt, tr_dataset_len)

val_dataset_len = len(val_dataloader.dataset)
val_iter_len = len(val_dataloader)

visualizer.write_log(f"Train cache path: {os.path.join(opt.cache_root, tr_dataloader.dataset.cache_name)}")
visualizer.write_log(f"Val cache path: {os.path.join(opt.cache_root, val_dataloader.dataset.cache_name)}")

# create trainer for our model
trainer = DfovTrainer(opt)

# opt.phase = "summary"
# summary(trainer.dfov_model_on_one_gpu, (3, opt.load_size[0], opt.load_size[1]), batch_size=opt.batchSize)
# opt.phase = "train"

best_metric = float('inf')

for epoch in tqdm(iter_counter.training_epochs(), position=0, leave=True, desc="Epoch"):
    iter_counter.record_epoch_start(epoch)
    opt.phase = "train"
    lrs = trainer.get_learning_rate()
    for i, lr in enumerate(lrs):
        if i == 0:
            name = "lr_G"
        elif i == 1:
            name = "lr_D"
        elif i == 2:
            name = "lr_Estimator"
            
        if len(lrs) == 1:
            name = "lr"
        visualizer.plot_current_lr(name, lr, epoch)

    trainer.dfov_model.train()
    visualizer.write_log(f"================ Epoch {epoch} Training Strat ================")
    for i, data_i in enumerate(tqdm(tr_dataloader, position=1, leave=True, desc="Iteration"), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        tr_batch_size = data_i["image"].shape[0]
        # Training
        trainer.run_one_step(data_i)
        losses = trainer.get_latest_losses()
        metrics = trainer.get_latest_metrics()

        if i == 0:
            epoch_loss = init_nested_dict(losses)
            epoch_metric = init_nested_dict(metrics)

        for (key1, value1) in losses.items():
            for (key2, value2) in value1.items():
                val = round(value2.item(), 2)
                losses[key1][key2] = val
                epoch_loss[key1][key2] += val

        for (key1, value1) in metrics.items():
            for (key2, value2) in value1.items():
                if key2 == "acc":
                    val2 = round(value2/tr_batch_size*100, 2)
                    epoch_metric[key1][key2] += value2
                else:
                    val2 = round(value2, 2)
                    epoch_metric[key1][key2] += val2

                metrics[key1][key2] = val2

        # Visualizations
        if iter_counter.needs_printing():
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            {"loss": losses, "metric": metrics}, iter_counter.time_per_iter)


        if iter_counter.needs_displaying():
            preds = trainer.get_latest_inferenced()
            if not opt.only_gan:
                if "edgeattentioncalib" in opt.network or "basic" in opt.network or "simple" in opt.network:
                    img_id = data_i['path'][0].split("\\")[-1]
                    fov = data_i['label']["fov_deg"][0].item()
                    roll = data_i['label']["roll"][0].item()
                    pitch = data_i['label']["pitch"][0].item()
                    # yaw = data_i['label']["yaw"][0].item()
                    focal = data_i['label']["focal"][0].item()
                    distor = data_i['label']["distor"][0].item()
                    
                    # visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f}\nyaw: {yaw:.2f}, focal: {focal:.2f}, distor: {distor:.2f} =>\nfov: {preds[5][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[1][0].item():.2f}\nyaw: {preds[2][0].item():.2f}, focal: {preds[3][0].item():.2f}, distor: {preds[4][0].item():.2f}", data_i['image'][0])])
                    visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f}\nfocal: {focal:.2f}, distor: {distor:.2f} =>\nfov: {preds[4][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[1][0].item():.2f}\nfocal: {preds[2][0].item():.2f}, distor: {preds[3][0].item():.2f}", data_i['image'][0])])
                elif "perceptual" in opt.network:
                    img_id = data_i['path'][0].split("\\")[-1]
                    fov = data_i['label']["fov_deg"][0].item()
                    roll = data_i['label']["roll"][0].item()
                    offset = data_i['label']["offset"][0].item()
                    pitch = data_i['label']["pitch"][0].item()
                    
                    visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f}\n=> fov: {preds[2][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[3][0].item():.2f}", data_i['image'][0])])
                elif "deepfocal" in opt.network:
                    visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}\nfov: {data_i['label']['fov_deg'][0]:.2f} => fov: {preds.squeeze()[0]}", data_i['image'][0])])
                elif "deepcalib" in opt.network:
                    img_id = data_i['path'][0].split("\\")[-1]
                    
                    focal_label = data_i['label']["focal"][0].item()
                    focal_pred = preds[0][0].item()
                    distor_label = data_i['label']["distor"][0].item()
                    distor_pred = preds[1][0].item()
                    fov_label = data_i['label']["fov_deg"][0].item()
                    fov_pred = preds[2][0].item()
                    
                    visuals = OrderedDict([(f"ID: {img_id}\nfocal: {focal_label:.2f}, distor: {distor_label:.2f}, fov: {fov_label:.2f}\n=> focal: {focal_pred:.2f}, distor: {distor_pred:.2f}, fov: {fov_pred:.2f}", data_i['image'][0])])
                else:
                    visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}\nfovx: {data_i['label'][0][0]:.2f}, fovy: {data_i['label'][1][0]:.2f} => fovx: {torch.argmax(preds[0][0])}, fovy: {torch.argmax(preds[1][0])}", data_i['image'][0])])

                visualizer.display_current_results(visuals, epoch, iter_counter.epoch_iter, "fov")

            if "edgeattention" in opt.network:
                edge_gt = data_i["edge"][0]
                if "dual" in opt.network or "single" in opt.network:
                    edge = edge_gt 
                else:
                    edge = trainer.get_latest_generated()[0]
                    edge = torch.cat((edge_gt, edge), dim=2)
                visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}", edge)])
                visualizer.display_current_results(visuals, epoch, iter_counter.epoch_iter, "edge")
                if opt.only_gan:
                    visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}",  data_i['image'][0])])
                    visualizer.display_current_results(visuals, epoch, iter_counter.epoch_iter, "input")
                    

        # if iter_counter.needs_saving():
        #     print('saving the latest model (epoch %d, total_steps %d)' %
        #           (epoch, iter_counter.total_steps_so_far))
        #     trainer.save('latest')
        #     iter_counter.record_current_iter()


    iter_counter.record_epoch_end()

    print('saving the model at the end of epoch %d, iters %d' %
            (epoch, iter_counter.total_steps_so_far))
    trainer.save('latest')
    trainer.save(epoch)

    for (key1, value1) in epoch_loss.items():
        for (key2, value2) in value1.items():
            epoch_loss[key1][key2] = round(value2 / tr_iter_len, 2)

    for (key1, value1) in epoch_metric.items():
        for (key2, value2) in value1.items():
            if key2 == "acc":
                val2 = round(value2/tr_dataset_len*100, 2)
            else:
                val2 = round(value2/tr_iter_len, 2)
                
            epoch_metric[key1][key2] = val2
    
    visualizer.write_log(f"================ Epoch {epoch} Train End ================")
    visualizer.write_log(f"Loss: {str(epoch_loss)}, Metric: {str(epoch_metric)}")
    visualizer.write_log(f"================ Epoch {epoch} Train End ================")
        

    # validation
    opt.phase = "val"
    val_epoch_loss = init_nested_dict(losses)
    val_epoch_metric = init_nested_dict(metrics)
    
    val_iter_counter = IterationCounter(opt, val_dataset_len)

    trainer.dfov_model.eval()
    visualizer.write_log(f"================ Epoch {epoch} Validation Strat ================")
    for i, data_i in enumerate(tqdm(val_dataloader, position=1, leave=True, desc="validation")):
        val_iter_counter.record_one_iteration()
        val_batch_size = data_i["image"].shape[0]

        trainer.run_one_step(data_i)
        val_losses = trainer.get_latest_losses()
        val_metrics = trainer.get_latest_metrics()

        for (key1, value1) in val_losses.items():
            for (key2, value2) in value1.items():
                val = round(value2.item(), 2)
                val_losses[key1][key2] = val
                val_epoch_loss[key1][key2] += val

        for (key1, value1) in val_metrics.items():
            for (key2, value2) in value1.items():
                if key2 == "acc":
                    val2 = round(value2/val_batch_size*100, 2)
                    val_epoch_metric[key1][key2] += value2
                else:
                    val2 = round(value2, 2)
                    val_epoch_metric[key1][key2] += val2
                val_metrics[key1][key2] = val2

        if val_iter_counter.needs_printing():
            visualizer.print_current_errors(epoch, val_iter_counter.epoch_iter,
                                            {"loss": val_losses, "metric": val_metrics}, val_iter_counter.time_per_iter)

        if val_iter_counter.needs_displaying():
            preds = trainer.get_latest_inferenced()
            if not opt.only_gan:
                if "edgeattentioncalib" in opt.network or "basic" in opt.network or "simple" in opt.network:
                    img_id = data_i['path'][0].split("\\")[-1]
                    fov = data_i['label']["fov_deg"][0].item()
                    roll = data_i['label']["roll"][0].item()
                    pitch = data_i['label']["pitch"][0].item()
                    # yaw = data_i['label']["yaw"][0].item()
                    focal = data_i['label']["focal"][0].item()
                    distor = data_i['label']["distor"][0].item()
                    
                    # visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f}\nyaw: {yaw:.2f}, focal: {focal:.2f}, distor: {distor:.2f} =>\nfov: {preds[5][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[1][0].item():.2f}\nyaw: {preds[2][0].item():.2f}, focal: {preds[3][0].item():.2f}, distor: {preds[4][0].item():.2f}", data_i['image'][0])])
                    visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f}\nfocal: {focal:.2f}, distor: {distor:.2f} =>\nfov: {preds[4][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[1][0].item():.2f}\nfocal: {preds[2][0].item():.2f}, distor: {preds[3][0].item():.2f}", data_i['image'][0])])
                elif "perceptual" in opt.network:
                    img_id = data_i['path'][0].split("\\")[-1]
                    fov = data_i['label']["fov_deg"][0].item()
                    roll = data_i['label']["roll"][0].item()
                    offset = data_i['label']["offset"][0].item()
                    pitch = data_i['label']["pitch"][0].item()
                    
                    visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f}\n=> fov: {preds[2][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[3][0].item():.2f}", data_i['image'][0])])
                elif "deepfocal" in opt.network:
                    visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}\nfov: {data_i['label']['fov_deg'][0]:.2f} => fov: {preds.squeeze()[0]}", data_i['image'][0])])
                elif "deepcalib" in opt.network:
                    img_id = data_i['path'][0].split("\\")[-1]
                    
                    focal_label = data_i['label']["focal"][0].item()
                    focal_pred = preds[0][0].item()
                    distor_label = data_i['label']["distor"][0].item()
                    distor_pred = preds[1][0].item()
                    fov_label = data_i['label']["fov_deg"][0].item()
                    fov_pred = preds[2][0].item()
                    
                    visuals = OrderedDict([(f"ID: {img_id}\nfocal: {focal_label:.2f}, distor: {distor_label:.2f}, fov: {fov_label:.2f}\n=> focal: {focal_pred:.2f}, distor: {distor_pred:.2f}, fov: {fov_pred:.2f}", data_i['image'][0])])
                else:
                    visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}\nfovx: {data_i['label'][0][0]:.2f}, fovy: {data_i['label'][1][0]:.2f} => fovx: {torch.argmax(preds[0][0])}, fovy: {torch.argmax(preds[1][0])}", data_i['image'][0])])

                visualizer.display_current_results(visuals, epoch, val_iter_counter.epoch_iter, "fov")

            if "edgeattention" in opt.network:
                edge_gt = data_i["edge"][0]
                if "dual" in opt.network or "single" in opt.network:
                    edge = edge_gt 
                else:
                    edge = trainer.get_latest_generated()[0]
                    edge = torch.cat((edge_gt, edge), dim=2)
                visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}", edge)])
                visualizer.display_current_results(visuals, epoch, val_iter_counter.epoch_iter, "edge")

            

    val_iter_counter.record_last()

    print('Validate the model at the end of epoch %d' % (epoch))

    for (key1, value1) in val_epoch_loss.items():
        for (key2, value2) in value1.items():
            val_epoch_loss[key1][key2] = round(value2 / val_iter_len, 2)

    for (key1, value1) in val_epoch_metric.items():
        for (key2, value2) in value1.items():
            if key2 == "acc":
                val2 = round(value2/val_dataset_len*100, 2)
            else:
                val2 = round(value2/val_iter_len, 2)
            val_epoch_metric[key1][key2] = val2
    
    visualizer.write_log(f"================ Epoch {epoch} Validation End ================")
    visualizer.write_log(f"Loss: {str(val_epoch_loss)}, Metric: {str(val_epoch_metric)}")
    visualizer.write_log(f"================ Epoch {epoch} Validation End ================")

    val_metric = 0
    for (key1, value1) in val_epoch_metric.items():
        val_metric += sum(value1.values()) / len(value1)
    val_metric /= len(val_epoch_metric)

    val_loss = 0
    for (key1, value1) in val_epoch_loss.items():
        val_loss += sum(value1.values()) / len(value1)
    val_loss /= len(val_epoch_loss)

    # best epoch
    if best_metric > val_metric:
       best_metric = val_metric
       trainer.save('best') 
       visualizer.write_log(f"=================Best epoch {epoch}=================")
    
    epoch_loss_res = {"train": epoch_loss, "val": val_epoch_loss}
    epoch_metric_res = {"train": epoch_metric, "val": val_epoch_metric}
    
    # avg_loss = {
    #     "train": {
    #         "fovx": 0,
    #         "fovy": 0,
    #     },
    #     "val": {
    #         "fovx": 0,
    #         "fovy": 0,
    #     }
    # }
    # initialize loss for logging
    avg_loss = {}
    for (phase, v1) in epoch_loss_res.items():
        avg_loss[phase] = {}
        for (cate, v2) in v1.items():
            avg_loss[phase][cate] = 0

    # initialize metric for logging
    # avg_metric = {
    #     "train": {
    #         "acc": 0,
    #         "MSE": 0
    #     },
    #     "val": {
    #         "acc": 0,
    #         "MSE": 0
    #     },
    # }
    avg_metric = {}
    for (phase, v1) in epoch_metric_res.items():
        avg_metric[phase] = {}
        for (cate, v2) in v1.items():
            for (metric_fn, v3) in v2.items():
                avg_metric[phase][metric_fn] = 0

    for (k1, v1) in epoch_loss_res["train"].items():
        tag = f"loss/{k1}"
        tr_avg = sum(epoch_loss_res["train"][k1].values())/len(epoch_loss_res["train"][k1])
        val_avg = sum(epoch_loss_res["val"][k1].values())/len(epoch_loss_res["val"][k1])
        visuals = {"train": tr_avg, "validation": val_avg} 
        visualizer.plot_epoch_errors(tag, visuals, epoch)

        avg_loss["train"][k1] = tr_avg
        avg_loss["val"][k1] = val_avg

        for (k2, v2) in v1.items():
            tag = f"loss_{k1}/{k2}"
            visuals = {"train": epoch_loss_res["train"][k1][k2], "validation": epoch_loss_res["val"][k1][k2]}
            visualizer.plot_epoch_errors(tag, visuals, epoch)

    tr_avg_loss = sum(avg_loss["train"].values()) / len(avg_loss["train"])
    val_avg_loss = sum(avg_loss["val"].values()) / len(avg_loss["val"])
    visualizer.plot_epoch_errors("loss", {"train": tr_avg_loss, "validation": val_avg_loss}, epoch)


    for (k1, v1) in epoch_metric_res["train"].items():
        for (k2, v2) in v1.items():
            tag = f"metric_{k1}/{k2}"
            visuals = {"train": epoch_metric_res["train"][k1][k2], "validation": epoch_metric_res["val"][k1][k2]}
            visualizer.plot_epoch_errors(tag, visuals, epoch)
            avg_metric["train"][k2] += epoch_metric_res["train"][k1][k2] / len(epoch_metric_res["train"])
            avg_metric["val"][k2] += epoch_metric_res["val"][k1][k2] / len(epoch_metric_res["val"])

    for (k1, v1) in avg_metric["train"].items():
        tag = f"metric/{k1}"
        visuals = {"train": v1, "validation": avg_metric["val"][k1]}
        visualizer.plot_epoch_errors(tag, visuals, epoch)

    trainer.update_learning_rate(val_loss)
    
    
visualizer.writer.close()

print('Training was successfully finished.')

print('Removing caches...')
shutil.rmtree(os.path.join(opt.cache_root, tr_dataloader.dataset.cache_name))
shutil.rmtree(os.path.join(opt.cache_root, val_dataloader.dataset.cache_name))
print('Complete to remove caches...')