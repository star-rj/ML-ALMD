import torch
import pandas
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PatientDataset(Dataset):
    def __init__(self, file_path, sheet_name, cls_num):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.cls_num = cls_num
        if self.cls_num == 3:
            self.cls2label = {
                'ALL': 0,
                'M3': 1,
                'M2': 2,
                'M5': 2,
                'M4': 2
            }
        elif self.cls_num == 5:
            self.cls2label = {
                'ALL': 0,
                'M2': 1,
                'M3': 2,
                'M4': 3,
                'M5': 4
            }
        elif self.cls_num == 2:
            self.cls2label = {
                'ALL': 0,
                'M2': 0,
                'M3': 0,
                'M4': 0,
                'M5': 0,
                'normal': 1
            }
        self.gt = {
                'ALL': 0,
                'M2': 1,
                'M3': 2,
                'M4': 3,
                'M5': 4
            }
        df = pandas.read_excel(self.file_path, self.sheet_name, header=0, index_col=0)
        feature = df.iloc[:, :-1].values
        float_features = []
        for row in feature:
            float_features.append(row.astype(float))

        label = df.iloc[:, -1].values
        gts = []
        for i in range(len(label)):
            gts.append(self.gt[label[i]])
            label[i] = self.cls2label[label[i]]

        self.feature = torch.from_numpy(np.array(float_features, dtype=np.float32))
        label = torch.from_numpy(np.array(label, dtype=int))
        label = label.unsqueeze(1)
        label = torch.zeros(len(label), self.cls_num).scatter_(1, label, 1)
        self.label = label
        self.gts = gts

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.feature[index], self.label[index], self.gts[index]

class AEPatientDataset(Dataset):
    def __init__(self, file_path, sheet_name, remove=False):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.remove = remove
        if self.remove:
            self.cls2label = {
                'ALL': -1,
                'M2': 0,
                'M3': -1,
                'M4': 1,
                'M5': 2
                }
        else:
            self.cls2label = {
                'ALL': 0,
                'M2': 1,
                'M3': 2,
                'M4': 3,
                'M5': 4
                }
        df = pandas.read_excel(self.file_path, self.sheet_name, header=0, index_col=0)
        feature = df.iloc[:, :-1].values
        float_features = []
        for row in feature:
            float_features.append(row.astype(float))

        label = df.iloc[:, -1].values
        for i in range(len(label)):
            label[i] = self.cls2label[label[i]]
        self.feature = torch.from_numpy(np.array(float_features, dtype=np.float32))
        if self.remove:
            keep_index = np.where((label==0)|(label==1)|(label==2))
            self.feature = self.feature[keep_index, :][0]
            label = label[keep_index]

        label = torch.from_numpy(np.array(label, dtype=int))
        label = label.unsqueeze(1)
        label = torch.zeros(len(label), len(label.unique())).scatter_(1, label, 1)

        self.label = label


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

