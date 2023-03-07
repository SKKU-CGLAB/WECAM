import cv2
import glob
import os
from tqdm import tqdm
import numpy as np

roots = ["add-test"]
phases = ["Train", "Val"]
phases = ["train"]

for root in roots:
    print(f"generating edges for {root}\n")
    for phase in phases:
        img_list = glob.glob(f"./{root}/{phase}/Image*/*.png", recursive=True)
        #img_list = glob.glob(f"F:\\project\\dfov\\ref\\src\\DeepPTZ\\matlab\\distorted\\focal225degree15\\{phase}\\Image*\\*.png", recursive=True)
        for i, img_path in enumerate(tqdm(img_list, desc="generating edge")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            med_val = np.median(img) 
            lower = int(max(0 ,0.7*med_val))
            upper = int(min(255,1.3*med_val))
            canny = cv2.Canny(img,lower,upper)
            save_path = img_path.replace(phase, f"{phase}/edge")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, canny)