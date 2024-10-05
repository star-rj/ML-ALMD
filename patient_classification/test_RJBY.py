import torch
import csv
import pandas
import numpy as np
import torch.nn as nn
import os 
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import PatientDataset, AEPatientDataset
from models import MLP

import random
random.seed(1)
# num2label = {
#     0: 'abnormal',
#     1: 'normal',
# }

# dataset_path = '~/codes/patient_classification/features/20240220_RJBY_M_POX.xlsx'
dataset_path = '~/codes/patient_classification/features/20240325_holdout_M_POX_CE.xlsx'
# pth_path = "~/codes/patient_classification/save_model/20231212_M_POX/split_0/epoch_449.pth"
data_nums = 10

for i in range(data_nums):
    print(f"running {i} data")
    pth_root = f'~/codes/patient_classification/save_model/20240227_RJ_M_POX_CE/split_{i}'
    pth = [os.path.join(pth_root, f) for f in os.listdir(pth_root)]
    if len(pth) == 0:
        raise KeyError
    pth_path = pth[-1]
    test_sheet = f'Sheet1'
    cls_num = 3
    test_dataset = PatientDataset(dataset_path, test_sheet, cls_num=cls_num)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = MLP(in_channels=22, cls_num=cls_num)
    model.load_state_dict(torch.load(pth_path))
    model.eval()
    preds = []
    gts = []
    right = 0
    with torch.no_grad():
        for idx, (feature, label, gt) in enumerate(test_dataloader):
            output = model(feature)
            pred = np.argmax(output.detach().numpy())
            label = np.argmax(label).detach().numpy()
            # preds.append(num2label[pred])
            preds.append(pred)
            gts.append(gt)
            if pred == label:
                right += 1
        acc = right /len(preds)
    preds_name = []
    for p in preds:
        if p == 0:
            preds_name.append("ALL")
        elif p == 1:
            preds_name.append("M3")
        else:
            preds_name.append("Non-APL AML")
    print(f"pred = {preds_name}")
    print(f"gt = {gts}")
    print(f"acc = {acc}")

