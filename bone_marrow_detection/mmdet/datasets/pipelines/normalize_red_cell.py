# Copyright (c) OpenMMLab. All rights reserved.
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Normalize_Red:

    def __init__(self):
        self.lower_red = np.array([156, 43, 46])
        self.higher_red = np.array([180, 255, 255])
        self.lower_red_h = int(self.lower_red[0])

        self.norm_h = 156  # 0-180
        self.norm_s = 53  # 0-255
        self.norm_v = 230  # 0-255

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        # img_path = results.get('filename')
        # contrast = cv2.imread(img_path)

        img = results.get('img')  # BGR, HWC
        # img = img[:, :, ::-1]
        # plt.imshow(img)
        # plt.show()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, self.lower_red, self.higher_red)  # 红色部分对应mask为255
        # keep = np.where(mask_red==255)
        h, s, v = hsv[:, :, 0].astype(np.uint16), hsv[:, :, 1].astype(np.uint16), hsv[:, :, 2].astype(np.uint16)
        # keeped_h = h[keep]
        hist_h = cv2.calcHist([hsv], [0], mask_red, [25], [self.lower_red_h, 180])
        hist_s = cv2.calcHist([hsv], [1], mask_red, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], mask_red, [256], [0, 256])
        #
        mean_h = np.where(hist_h == max(hist_h))[0]
        mean_s = np.where(hist_s == max(hist_s))[0]
        mean_v = np.where(hist_v == max(hist_v))[0]
        if len(mean_h) != 1:
            mean_h = int(mean_h[0])+self.lower_red_h
        else:
            mean_h = int(mean_h)+self.lower_red_h
        if len(mean_s) != 1:
            mean_s = int(mean_s[0])
        else:
            mean_s = int(mean_s)
        if len(mean_v) != 1:
            mean_v = int(mean_v[0])
        else:
            mean_v = int(mean_v)
        mask_white = cv2.inRange(s, 50, 255)
        keep = np.where(mask_white == 255)

        delta_h = mean_h - self.norm_h
        delta_s = mean_s - self.norm_s
        delta_v = mean_v - self.norm_v

        # delta_h = 0
        # delta_s = 0
        # delta_v = -40

        h[keep] = h[keep] - delta_h
        # # s = s - delta_s
        s[keep] = s[keep] - delta_s
        v[keep] = v[keep] - delta_v
        #
        # h = h[np.newaxis, :].clip(0, 180)
        # s = s[np.newaxis, :].clip(0, 255)
        # v = v[np.newaxis, :].clip(0, 255)
        # img_hsv = np.concatenate((h, s, v), axis=0).transpose(1, 2, 0)
        h = h[:,:, np.newaxis].clip(0, 180)
        s = s[:,:, np.newaxis].clip(0, 255)
        v = v[:,:, np.newaxis].clip(0, 255)
        img_hsv = np.concatenate((h, s, v), axis=2).astype(np.uint8)
        # img_hsv[:,:,2] = 255
        # hsv[:,:,2][keep] = 0
        # img_hsv = hsv
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


        # plt.subplot(1, 3, 1)
        # plt.imshow(img[:, :, ::-1])
        # plt.title('before')
        # plt.subplot(1, 3, 2)
        # plt.imshow(mask_white, cmap='gray')
        # plt.title('mask')
        # plt.subplot(1, 3, 3)
        # plt.imshow(img_bgr[:, :, ::-1])
        # plt.title('after')
        # plt.suptitle(f'h={delta_h}, s={delta_s}, v={delta_v}')
        # plt.show()
        # plt.imshow(img_bgr[:, :, ::-1])
        # plt.show()

        results['img'] = img_bgr

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
