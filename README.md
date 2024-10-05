# Codes


---

**TABLE OF CONTENTS**

- [Codes](#codes)
  - [Installation](#installation)
    - [Environment](#environment)
  - [Dataset Preparation](#dataset-preparation)
  - [Feature Generation and Fusion](#feature-generation-and-fusion)
    - [Train](#train)
    - [Inference](#inference)
      - [Step 1. Cell-level Feature Generation](#step-1-cell-level-feature-generation)
      - [Step 2 Patient-level Feature Fusion](#step-2-patient-level-feature-fusion)
  - [Leukemia Classification](#leukemia-classification)
    - [Train](#train-1)
    - [Inference](#inference-1)


---


## Installation

### Environment
+ Operating System: Ubuntu 18.04 LTS
+ Python 3.8 (Anaconda)



The implementation is based on [MMDetection (an open source object detection toolbox based on PyTorch)](https://github.com/open-mmlab/mmdetection). Detailed installation guides and issues might be found at [installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).

Example installation commands: 

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

conda install pytorch torchvision -c pytorch

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
```

In the following guides, 
+ we suppose you have clone the codes to the directory `~/codes` on your device.
+ for simplicity, we set aliases for stainings as:
   + `M`: Wright
   + `POX`: POX
   + `CE`: CE

## Dataset Preparation

Images might be placed in the directory `~/codes/data`, in a structure similar to `~/codes/sample_case/`:
+ `~/codes/sample_case/<type>_<any_idx_you_like>`: specifies a case/patient, with,
  +  `<type>`: disease type
  +  `<any_idx_you_like>`: any index you may add
+ `~/codes/sample_case/<type>_<any_idx_you_like>/<staining>/jpg`: contains the images of a staining. There are NO requirements of image filenames.


## Feature Generation and Fusion


### Train

```bash
cd mmdetection

# format: `bash tools/dist_train.sh <config_file> <cnt>`
bash tools/dist_train.sh configs/CE/cascade_rcnn_r101_CE.py 1
```

where,
+ `<config_file>`: path to the config file,
    + CE: `configs/CE/cascade_rcnn_r101_CE.py`
    + M: `configs/M/cascade_rcnn_r101_M_17.py`
    + POX: `configs/POX/cascade_rcnn_r101_POX_concat.py`
+ `<cnt>`: the number of GPUs used



### Inference

#### Step 1. Cell-level Feature Generation

Download the trained weight files from [Google Drive](https://drive.google.com/drive/folders/1BNPOzD6hfCdkaSkspQknnIV_U6B92wj_?usp=sharing), and, 
+ place `cascade_rcnn_r101_M_linan_dataset_apl__epoch_19.pth` in the directory `~/codes/mmdetection/work_dirs/cascade_rcnn_r101_M_linan_dataset_apl/`, as `~/codes/mmdetection/work_dirs/cascade_rcnn_r101_M_linan_dataset_apl/cascade_rcnn_r101_M_linan_dataset_apl__epoch_19.pthh`
+ place `cascade_rcnn_r101_POX_concat__epoch_24.pth` in the directory `~/codes/mmdetection/work_dirs/cascade_rcnn_r101_POX_concat/`, as `~/codes/mmdetection/work_dirs/cascade_rcnn_r101_POX_concat/cascade_rcnn_r101_POX_concat__epoch_24.pth`
+ place `ce_swin_T_1025.pth` in the directory `~/codes/mmdetection/CE/POX/`, as `~/codes/mmdetection/pth/CE/ce_swin_T_1025.pth`
+ place `pox_refactor.pth` in the directory `~/codes/mmdetection/pth/POX/`, as `"~/codes/mmdetection/pth/POX/pox_refactor.pth"`


*[Optional]* Edit the model/data/output paths in Line #106-#135 in `~/codes/mmdetection/tools/feature_generator.py`, where,
+ `root`: path of the input images
+ `typ`: target staining. Allowed values: `'M'`, `'POX'`, `'CE'`, `'RJBY_POX'`
+ `date`: filename of the output feature
+ `csv_path`: directory of the output feature
+ `arg_pth_dict`: paths to the model weight files and corresponding config files.


For each of the three stainings, run:

```bash
python ~/codes/mmdetection/tools/feature_generator.py
```

The result file will be generated as `<csv_path>/<date>.csv</date>` (e.g., `csv/20240312_holdout_POX.xlsx` for POX staining).


#### Step 2 Patient-level Feature Fusion

Edit the paths of the previously [generated](#step-1-cell-level-feature-generation) features in Line #4-#12 in `~/codes/mmdetection/tools/merge_excel.py`, where,
+ `excel1`: *(required)* specify the path to the output feature of `M` (e.g., `'csv/20240312_holdout_M.csv'`)
+ `excel2`: *(optional)* specify the path to the output feature of `POX` (e.g., `'csv/20240312_holdout_POX.csv'`)
+ `excel3`: *(optional)* specify the path to the output feature of `CE` (e.g., `'csv/20240312_holdout_CE.csv'`)
+ `merged_df`: specify the target staining(s), where,
  + `M` only: `merged_df = pd.concat([excel1], axis=1)`
  + `M` + `POX`: `merged_df = pd.concat([excel1, excel2], axis=1)`
  + `M` + `CE`: `merged_df = pd.concat([excel1, excel3], axis=1)`
  + `M` + `POX` + `CE`: `merged_df = pd.concat([excel1, excel2, excel3], axis=1)`
+ `merged_df.to_excel(<output_path>/<output_filename>.xlsx)`: specifies the path of the output fused features （e.g., `'csv/20240312_holdout_M.xlsx'`）


Run:
```bash
cd ~/codes/mmdetection
python tools/merge_excel.py
```

The result file will be generated as `<output_path>.xlsx`.



## Leukemia Classification

Copy the previously [generated](#step-2-patient-level-feature-fusion) fused feature file, from `<output_path>/<output_filename>.xlsx` to `~/codes/patient_classification/features/<output_path>/<output_filename>.xlsx`.

### Train

Run:
```bash
cd ~/codes/mmdetection
python MLP.py
```

where, 
+ `dataset_path`: path of the fused feature file (e.g., `'~/codes/patient_classification/features/20240325_holdout_M_POX_CE.xlsx'`)


### Inference

Edit the paths in Line #21-#27 in `~/codes/patient_classification/test_RJBY.py`, where,
+ `dataset_path`: path of the fused feature file (e.g., `'~/codes/patient_classification/features/20240325_holdout_M_POX_CE.xlsx'`)
+ `pth_root`: path of the model, where,
  + `M` only: `pth_root = f'~/codes/patient_classification/save_model/20240227_RJ_M/split_{i}'`
  + `M` + `POX`: `pth_root = f'~/codes/patient_classification/save_model/20240227_RJ_M_POX/split_{i}'`
  + `M` + `CE`: `pth_root = f'~/codes/patient_classification/save_model/20240227_RJ_M_CE/split_{i}'`
  + `M` + `POX` + `CE`: `pth_root = f'~/codes/patient_classification/save_model/20240227_RJ_M_POX_CE/split_{i}'`

Run:
```bash
cd ~/codes/patient_classification
python test_RJBY.py > result.txt
```

The results will be saved in `~/codes/patient_classification/result.txt`.


