# Copyright (c) OpenMMLab. All rights reserved.
import sys
import staintools
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Stain_norm:

    def __init__(self, target):
        self.target = staintools.read_image(target)  # RGB

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        img = results.get('img')  # BGR, HWC
        to_transform = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        target = staintools.LuminosityStandardizer.standardize(self.target)
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

        normalizer = staintools.StainNormalizer(method='vahadane')
        normalizer.fit(target)
        transformed = normalizer.transform(to_transform)  # BGR

        # plt.subplot(1, 3, 1)
        # plt.imshow(self.target)
        # plt.title('target')
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(to_transform)
        # plt.title('before')
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(transformed)
        # plt.title('after')
        # plt.show()

        results['img'] = transformed[:, :, ::-1]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
