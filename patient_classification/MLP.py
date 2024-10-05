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
# num2label = {
#     0: 'abnormal',
#     1: 'normal',
# }

dataset_path = 'patient_classification/features/20240227_RJ_M_CE.xlsx'
data_nums = 10  # 10 folds

for i in range(data_nums):
    print(f"running {i} data")
    train_sheet = f'train_{i}'
    test_sheet = f'test_{i}'
    cls_num = 3
    train_dataset = PatientDataset(dataset_path, train_sheet, cls_num=cls_num)
    test_dataset = PatientDataset(dataset_path, test_sheet, cls_num=cls_num)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    EPOCH = 450
    model = MLP(in_channels=19, cls_num=cls_num)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)   # optimize all logistic parameters

    loss_func = nn.CrossEntropyLoss()
    out_info = {
        "best_acc": 0,
        "best_epoch": 0,
        "AE_best_acc": 0
    }
    for epoch in range(EPOCH):
        l = 0
        model.train()
        for idx, (feature, label, gt) in enumerate(train_dataloader):
            for ind, f in enumerate(feature[0,:]):
                if 0<=f<=1:
                    continue
                else:
                    feature[0][ind] = 0
            output = model(feature)
            loss = loss_func(output, label)
            l += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"epoch {epoch}, loss: {l} ")
        # test
        preds = []
        gts = []
        right = 0
        model.eval()
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
            out_info['best_acc'] = max(out_info['best_acc'], acc)
            if out_info['best_acc'] == acc:
                out_info['best_epoch'] = epoch
                out_info['preds'] = preds
                out_info["gts"] = gts
                best_model_state = model.state_dict()
            if epoch == EPOCH-1:
                print("gt:", gts)
                print("pred:", preds)
                print(f"epoch {epoch} acc: {acc}")
    print("Best:", out_info["preds"])
    file_name = dataset_path.split("/")[-1].split(".")[0]
    save_dir = f'save_model/{file_name}/split_{i}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{out_info['best_epoch']}.pth")
    torch.save(best_model_state, save_path)




