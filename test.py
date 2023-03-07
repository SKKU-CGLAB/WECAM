import os
from collections import OrderedDict

import torch

import data
from options.test_options import TestOptions
from models.dfov_model import DfovModel
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util.util import init_nested_dict
from tqdm import tqdm
import torch
import numpy as np
import random
import math

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

opt = TestOptions().parse()
visualizer = Visualizer(opt)

dataloader, _ = data.create_dataloader(opt)
dataset_len = len(dataloader.dataset)
iter_len = len(dataloader)
# create tool for counting iterations
iter_counter = IterationCounter(opt, iter_len)

model = DfovModel(opt)
model.eval()

print('Number of images: ', len(dataloader))

infer_res_list = []

for i, data_i in enumerate(tqdm(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break

    batch_size = data_i['image'].shape[0]
    iter_counter.record_one_iteration()
    if len(opt.gpu_ids) > 0:
        data_i["image"] = data_i["image"].cuda()
        for key, label in data_i['label'].items():
            data_i['label'][key] = label.cuda().float()
        # data["label"][0] = data["label"][0].cuda()
        # data["label"][1] = data["label"][1].cuda()
        if opt.network in "edgeattention" or "edgeattentioncalib" in opt.network or "single" in opt.network:
            data_i["edge"] = data_i["edge"].cuda()
            data_i["semantic"] = data_i["semantic"].cuda()
            data_i["weight_map"] = data_i["weight_map"].cuda()

    preds, _ = model(data_i)
    metrics = model.comput_metric(preds, data_i['label'])
    
    ex_keys = ['slope_label', 'offset_label', 'fov_label', 'yaw', 'offset'] 
    for b_idx in range(batch_size):
        idx = data_i['path'][b_idx].split("\\")[-1]
        labels = {}
        for key, item in data_i['label'].items():
            if key in ex_keys:
                continue
            labels[key] = item[b_idx].item()
        cur_pred = []    
        for pred in preds:
            cur_pred.append(pred[b_idx].item())
        infer_res = {"id":idx, "label": labels, "preds": cur_pred}
        infer_res_list.append(infer_res)
    

    if i == 0:
        metric_res = init_nested_dict(metrics)

    for (key1, value1) in metrics.items():
        for (key2, value2) in value1.items():
            if key2 == "acc":
                val2 = round(value2/batch_size*100, 2)
                metric_res[key1][key2] += value2
            else:
                val2 = round(value2, 2)
                metric_res[key1][key2] += val2 ** 2

            metrics[key1][key2] = val2
    
    # Visualizations
    if iter_counter.needs_printing():
        visualizer.print_current_errors(-1, iter_counter.epoch_iter, {"ID": os.path.basename(data_i['path'][0]), "metric": metrics}, iter_counter.time_per_iter)

    if iter_counter.needs_displaying():
        if "deepfocal" in opt.network:
            visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}\nfov: {data_i['label']['fov_deg'][0]:.2f} => fov: {preds[0][0]}", data_i['image'][0])])
        elif "deepcalib" in opt.network:
            img_id = data_i['path'][0].split("\\")[-1]
            
            focal_label = data_i['label']["focal"][0].item()
            focal_pred = preds[0][0].item()
            distor_label = data_i['label']["distor"][0].item()
            distor_pred = preds[1][0].item()
            fov_label = data_i['label']["fov_deg"][0].item()
            fov_pred = preds[2][0].item()
            
            visuals = OrderedDict([(f"ID: {img_id}\nfocal: {focal_label:.2f}, distor: {distor_label:.2f}, fov: {fov_label:.2f}\n=> focal: {focal_pred:.2f}, distor: {distor_pred:.2f}, fov: {fov_pred:.2f}", data_i['image'][0])])
        elif "perceptual" in opt.network:
            img_id = data_i['path'][0].split("\\")[-1]
            fov = data_i['label']["fov_deg"][0].item()
            roll = data_i['label']["roll"][0].item()
            offset = data_i['label']["offset"][0].item()
            pitch = data_i['label']["pitch"][0].item()
            
            visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f} => fov: {preds[2][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[3][0].item():.2f}", data_i['image'][0])])
        elif "edgeattentioncalib" in opt.network or "basic" in opt.network or "simple" in opt.network:
            img_id = data_i['path'][0].split("\\")[-1]
            fov = data_i['label']["fov_deg"][0].item()
            roll = data_i['label']["roll"][0].item()
            pitch = data_i['label']["pitch"][0].item()
            focal = data_i['label']["focal"][0].item()
            distor = data_i['label']["distor"][0].item()
            
            visuals = OrderedDict([(f"ID: {img_id}\nfov: {fov:.2f}, roll: {roll:.2f}, pitch: {pitch:.2f},\nfocal: {focal:.2f}, distor: {distor:.2f} =>\nfov: {preds[4][0].item():.2f}, roll: {preds[0][0].item():.2f}, pitch: {preds[1][0].item():.2f}\nfocal: {preds[2][0].item():.2f}, distor: {preds[3][0].item():.2f}", data_i['image'][0])])
        else:
            visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}\nfovx: {data_i['label'][0][0]:.2f}, fovy: {data_i['label'][1][0]:.2f} => fovx: {torch.argmax(preds[0][0])}, fovy: {torch.argmax(preds[1][0])}", data_i['image'][0])])
    
        visualizer.display_current_results(visuals, -1, iter_counter.total_steps_so_far, "fov")

        #if "edgeattention" in opt.network:
        #    edge_gt = data_i["edge"][0]
        #    edge = _[0]
        #    edge_cat = torch.cat((edge_gt, edge), dim=2)
        #    visuals = OrderedDict([(f"ID: {os.path.basename(data_i['path'][0])}", edge_cat)])
        #    visualizer.display_current_results(visuals, -1, iter_counter.epoch_iter, "edge")

iter_counter.record_last()


for (key1, value1) in metric_res.items():
    for (key2, value2) in value1.items():
        if key2 == "acc":
            val2 = round(value2/dataset_len*100, 2)
        else:
            val2 = round(math.sqrt(value2/iter_len), 2)
            
        metric_res[key1][key2] = val2

visualizer.write_log(f"================ Network: {opt.network} Test End ================")
visualizer.write_log(f" Metric: {str(metric_res)}")
visualizer.write_log(f"================ Network: {opt.network} Test End ================")

visualizer.write_results_json(infer_res_list)

