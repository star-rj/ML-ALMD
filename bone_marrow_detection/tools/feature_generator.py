import csv
import os
import cv2
import numpy as np
from tqdm import tqdm

from patients_detect import prepare_model
from mmdet.apis import inference_detector
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def adjust_dataset(patient_root, typ):
    """
    把数据集按照病例调整成需要的格式，并且把tif/png转为jpg
    Args:
        patient_root:
        typ: 染色方式
    Returns:

    """
    for root, dirs, files in os.walk(patient_root):
        for dir in dirs:
            p = os.path.join(root, dir)
            path_root = os.path.join(p, typ)
            # 判断是否是训练集
            for r, ds, fs in os.walk(path_root):
                if 'jpg' in ds:
                    jpg_path_root = os.path.join(r, 'jpg')
                    for jpg_name in os.listdir(jpg_path_root):
                        jpg_path = os.path.join(jpg_path_root, jpg_name)
                        if jpg_path[-3:] == 'tif' or jpg_path[-3:] == 'png':
                            tif = cv2.imread(jpg_path)
                            jpg = jpg_path[:-3]+'jpg'
                            cv2.imwrite(jpg, tif)
                    print(f'{path_root} already has jpg file, pass')
                    break
                if 'xml' not in ds:  # 测试集，讲tif文件全部转为jpg
                    print(f'converting {path_root}...')
                    jpg_root = os.path.join(path_root, 'jpg')
                    if not os.path.exists(jpg_root):
                        os.mkdir(jpg_root)
                    for f in fs:
                        if f.endswith('.tif') or f.endswith('.png'):
                            tif_path = os.path.join(path_root, f)
                            img_tif = cv2.imread(tif_path, 1)
                            end = f.split('.')[-1]
                            save_jpg_path = os.path.join(jpg_root, f.strip(end) + 'jpg')
                            cv2.imwrite(save_jpg_path, img_tif)
                    break
                else:  # 训练集
                    tif_root = os.path.join(r, '原图')
                    jpg_root = os.path.join(r, 'jpg')
                    if not os.path.exists(jpg_root):
                        os.mkdir(jpg_root)
                    for _, _, tifs in os.walk(tif_root):
                        for tif in tifs:
                            if tif.endswith('.tif'):
                                tif_path = os.path.join(tif_root, tif)
                                img_tif = cv2.imread(tif_path, 1)
                                end = tif.split('.')[-1]
                                save_jpg_path = os.path.join(jpg_root, tif.strip(end) + 'jpg')
                                cv2.imwrite(save_jpg_path, img_tif)
                    break
    print(f"ALl images in {typ} dataset have been converted to jpg style")
    return


def test_patient(model, data_root, confidence=0.3):
    """

    Args:
        model: 加载后的模型
        data_root: 图像所在文件夹，注意，这个文件夹下不再有子文件夹
        confidence: 保留的proposal 阈值
    Returns:

    """
    patient_name = data_root.split('/')[-3]
    print(f"handling {patient_name}")
    img_path_list = os.listdir(data_root)
    results = []
    for ind, img in tqdm(enumerate(img_path_list)):
        img_path = os.path.join(data_root, img)
        if img_path[-3:] != 'jpg' and img_path[-3:] != 'tif':
            continue
        result = np.array([inference_detector(model, img_path)])
        results.extend(result)

    img_num = len(results)
    class_num = len(results[0])
    class_dets = [0]*class_num

    for i in range(img_num):
        img_proposals = results[i]
        for cls in range(class_num):
            proposals = img_proposals[cls]
            keep = np.where(proposals[:, -1] > confidence)
            proposals = proposals[keep]
            after_keep_proposal_nums = len(proposals)
            class_dets[cls] += after_keep_proposal_nums

    return class_dets

if __name__ == '__main__':
    # parameters
    # date
    date = '20240312_holdout_POX'
    # 写入csv文件路径
    csv_path = '~/codes/bone_marrow_detection/csv'
    # 测试的染色类型 'M' 'POX' 'CE' 'RJBY_POX'
    typ = 'POX'
    # 病例路径
    root = "/data2/ruijin/holdout_ruijin/holdout"
    # 配置文件和权重文件
    arg_pth_dict = {
        "M": {
            "arg_path": "~/codes/bone_marrow_detection/configs/M/cascade_rcnn_r101_M_linan_dataset_apl.py",
            # "model_path": "~/codes/bone_marrow_detection/work_dirs/cascade_rcnn_r101_M_linan_dataset_apl/epoch_19.pth"
            "model_path": "~/codes/bone_marrow_detection/work_dirs/cascade_rcnn_r101_M_linan_dataset_apl/cascade_rcnn_r101_M_linan_dataset_apl__epoch_19.pth"
        },
        "POX": {
            "arg_path": "~/codes/bone_marrow_detection/configs/POX/cascade_rcnn_r101_POX_concat.py",
            # "model_path": "~/codes/bone_marrow_detection/work_dirs/cascade_rcnn_r101_POX_concat/epoch_24.pth"
            "model_path": "~/codes/bone_marrow_detection/work_dirs/cascade_rcnn_r101_POX_concat/cascade_rcnn_r101_POX_concat__epoch_24.pth"
        },
        "RJBY_POX":{
            "arg_path":"~/codes/bone_marrow_detection/pth/POX/pox_refactor.py",
            "model_path":"~/codes/bone_marrow_detection/pth/POX/pox_refactor.pth",
        },
        "CE":{
            "arg_path": '~/codes/bone_marrow_detection/pth/CE/ce_swin_T_1025.py',
            "model_path": "~/codes/bone_marrow_detection/pth/CE/ce_swin_T_1025.pth"
        },
    }
    # 
    arg_path = arg_pth_dict[typ]["arg_path"]
    model_path = arg_pth_dict[typ]["model_path"]
    if typ == "RJBY_POX":
        typ = "POX"
        
    # 主程序
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    csv_file = os.path.join(csv_path, date+'.csv')

    model = prepare_model(arg_path, model_path)
    # 以下配置为不同染色配置个性化设置
    # M, POX
    ignore = ['忽略']

    cnt_cls = []
    for item in model.CLASSES:
        if item not in ignore:
            cnt_cls.append(item)
    print(f"需要考虑的类别为{cnt_cls}")

    header = [''] + [cls for cls in cnt_cls]
    with open(csv_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # 准备训练数据和测试数据,将tif或png格式的图片转为tif
    # adjust_dataset(root, typ)
    # 开始测试
    for r, dirs, files in os.walk(root):
        for dir in dirs:
            if dir.startswith('ALL'):
                patient_class = 'ALL'
            else:
                patient_class = dir[0:2]

            patient_root = os.path.join(r, dir)
            type_root = os.path.join(patient_root, typ)
            jpg_root = os.path.join(type_root, 'jpg')
            dets_per_class = test_patient(model, jpg_root)

            # POX, CE, M
            dets_per_class = dets_per_class[:-1]
            assert len(dets_per_class) == len(cnt_cls)
            # 计算比例
            sum_dets = sum(dets_per_class)
            if sum_dets == 0:
                print(f"Warning, {patient_root} type {typ} didn't detect anything")
                ratio = [0 for i in dets_per_class]
            else:
                ratio = ['%.2f' % (i/sum_dets) for i in dets_per_class]
            # 写入 csv
            with open(csv_file, 'a', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                row = [dir] + [str(n) for n in ratio] + [patient_class]
                writer.writerow(row)
        break
