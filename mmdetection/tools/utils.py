import cv2
import os
import numpy as np

from scipy import signal

import matplotlib.pyplot as plt


def mask_single_cell(file_path, isAE, save=False):
    """

    Args:
        file_path: jpg图像路径

    Returns:
        mask_img: ndarray(h, w),灰度图
    """
    if isAE:
        n = 'AE'
    else:
        n = 'AE_NAF'
    img = cv2.imread(file_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_gray = f'/home/zxw/project/mmdetection/yuanyou/{n}/gray'
    if not os.path.exists(save_gray):
        os.mkdir(save_gray)
    save_gray_file = os.path.join(save_gray, file_path.split('/')[-1])
    cv2.imwrite(save_gray_file, gray_img)

    # _, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 第二个参数为灰度阈值，大于这个阈值的赋值为第三个参数
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = list(contours)
    # contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    # res = cv2.drawContours(img, contours, 0, (0, 0, 0), cv2.FILLED)
    # # 转为灰度图
    # gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # _, thresh2 = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
    # mask_img = cv2.bitwise_or(gray_img, thresh2)
    # if save:
    #     save_path_root = '/home/zxw/project/mmdetection/yuanyou/mask_single_cell'
    #     if not os.path.exists(save_path_root):
    #         os.mkdir(save_path_root)
    #     img_name = file_path.split('/')[-1]
    #     save_path = os.path.join(save_path_root, img_name)
    #     cv2.imwrite(save_path, mask_img)
    # return mask_img
    return gray_img


def mask_single_cell_BGR(file_path, save=False):
    """

    Args:
        file_path: jpg图像路径

    Returns:
        mask_img: ndarray(h, w, c),BGR图
    """
    img = cv2.imread(file_path)
    img_copy = img.copy()
    h, w, _ = img.shape
    if h >= w:
        if h/w >= 1.5:
            return None
    else:
        if w/h >= 1.5:
            return None
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    _, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 第二个参数为灰度阈值，大于这个阈值的赋值为第三个参数
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    res = cv2.drawContours(img, contours, 0, (0, 0, 0), cv2.FILLED)
    # 转为灰度图
    gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, thresh2 = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.bitwise_not(thresh2)
    mask_img = cv2.bitwise_and(img_copy, img_copy, mask=thresh2)
    if save:
        patient = file_path.split('/')[4]
        if not os.path.exists('/home/zxw/project/Swin-Transformer-Object-Detection/yuanyou/test_case'):
            os.mkdir('/home/zxw/project/Swin-Transformer-Object-Detection/yuanyou/test_case')
        save_path_root = os.path.join('/home/zxw/project/Swin-Transformer-Object-Detection/yuanyou/test_case', patient)
        if not os.path.exists(save_path_root):
            os.mkdir(save_path_root)
        img_name = file_path.split('/')[-1]
        save_path_root = os.path.join(save_path_root, 'origin')
        if not os.path.exists(save_path_root):
            os.mkdir(save_path_root)
        save_path = os.path.join(save_path_root, img_name)
        cv2.imwrite(save_path, img_copy)
    return mask_img


def gray_threshold_mask(file_path, gray_img, isAE, threshold=-1, save=False):
    """
    利用灰度值进行阈值分割
    Returns:

    """
    if isAE:
        n = 'AE'
    else:
        n = 'AE_NAF'
    gray_img = cv2.normalize(gray_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # plt.imshow(gray_img, cmap='gray')
    # plt.show()
    _, thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    if save:
        save_path_root = f'/home/zxw/project/mmdetection/yuanyou/{n}/final_mask_gray'
        if not os.path.exists(save_path_root):
            os.mkdir(save_path_root)
        img_name = file_path.split('/')[-1]
        save_path = os.path.join(save_path_root, img_name)
        cv2.imwrite(save_path, thresh)

    return thresh


def gradient_mask(file_path, ratio=0.65, save=False):
    """
    利用灰度梯度变化进行色点分割
    Args:
        file_path: 图像路径
        ratio: 多少比例保留
        save:

    Returns:

    """
    img = cv2.imread(file_path)

    img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=5)

    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()
    img_gray = cv2.normalize(img_gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    kernelLaplace1 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # Laplacian kernel
    imgLaplace = signal.convolve2d(img_gray, kernelLaplace1, boundary='symm', mode='same')  # same 卷积

    imgLaplace = cv2.normalize(imgLaplace, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    T = ratio * imgLaplace.max()
    # for h in range(hImg):
    #     for w in range(wImg):
    #         if imgLaplace[h, w] > T:
    #             imgLaplace[h, w] = 0  # 二值处理
    #         else:
    #             imgLaplace[h, w] = 255
    pos_0 = imgLaplace > T
    pos_255 = imgLaplace <= T
    imgLaplace[pos_0] = 0
    imgLaplace[pos_255] = 255

    kernel = np.ones((2, 2), np.uint8)
    imgLaplace = cv2.erode(imgLaplace.astype('uint8'), kernel, iterations=1)
    imgLaplace = cv2.dilate(imgLaplace.astype('uint8'), kernel, iterations=2)
    imgLaplace = cv2.erode(imgLaplace.astype('uint8'), kernel, iterations=1)

    # plt.imshow(imgLaplace, cmap='gray')
    # plt.show()
    if save:
        save_path_root = '/home/zxw/project/Swin-Transformer-Object-Detection/yuanyou/final_mask_gradient'
        if not os.path.exists(save_path_root):
            os.mkdir(save_path_root)
        img_name = file_path.split('/')[-2] + '_' + file_path.split('/')[-1]
        save_path = os.path.join(save_path_root, img_name)
        # save_tmp = os.path.join(save_path_root, 'ori_'+img_name)
        cv2.imwrite(save_path, imgLaplace)
        # cv2.imwrite(save_tmp, img)

    return imgLaplace


def binary_statistics(binary_img):
    """
    统计二值图中，黑色点的面积
    Args:
        binary_img: ndarray(h,w)

    Returns:
    """
    nums = binary_img[binary_img==0].size
    return nums


# 根据病例进行不同化学染色检测的文件
def patient_detection(data_root, model):
    """

    Args:
        data_root: 病例文件的根路径
        model:

    Returns:

    """
    for root, dirs, files in os.walk(data_root):
        for dir in dirs:
            if dir.startswith('ALL'):
                # todo:ALL病例
                continue
            elif dir.startswith('ETP'):
                # todo:ETP病例
                continue
            else:
                cls = patient_class2num[dir[0:2]]
                p = os.path.join(root, dir)
                print(dir)
                AE_path_root = os.path.join(p, 'AE')
                NAF_path_root = os.path.join(p, 'AE+NaF')
                if not os.path.exists(NAF_path_root):  # 调整一下文件夹名称的不统一问题
                    NAF_path_root = os.path.join(p, 'AE+NAF')
                # 3.1 先处理AE
                jpg_path_root = os.path.join(AE_path_root, 'jpg')
                img_path_list = [os.path.join(jpg_path_root, img) for img in os.listdir(jpg_path_root)]
                for ind, img in tqdm(enumerate(img_path_list)):
                    result = inference_detector(model, img)

                    # 这是最开始存小图的地方，在每个病例的文件夹里
                    # save_path = os.path.join(AE_path_root, 'split_image_1')

                    # 为了给分割做数据集，现在统一将存储路径设定在/data2/yuanyou_split
                    save_path = '/data2/yuanyou_split'
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    split_image(img, result, save_path_root=save_path)
                # 3.2 处理AE+NAF
                jpg_path_root = os.path.join(NAF_path_root, 'jpg')
                img_path_list = [os.path.join(jpg_path_root, img) for img in os.listdir(jpg_path_root)]
                for ind, img in tqdm(enumerate(img_path_list)):
                    result = inference_detector(model, img)

                    # 这是最开始存小图的地方，在每个病例的文件夹里
                    # save_path = os.path.join(NAF_path_root, 'split_image_1')

                    # 为了给分割做数据集，现在统一将存储路径设定在/data2/yuanyou_split
                    save_path = '/data2/yuanyou_split'

                    split_image(img, result, save_path_root=save_path)
        break  # 只循环第一层

    return
