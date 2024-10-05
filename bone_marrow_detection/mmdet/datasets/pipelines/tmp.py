import numpy as np


def norm_red(data):
    """
    根据图像中红细胞的颜色进行归一化
    Args:
        data:

    Returns:

    """
    import matplotlib.pyplot as plt
    # histCV = cv2.calcHist(img_gray, [0], None, [256], [0, 256])
    histNP, bins = np.histogram(data.flatten(), 256)

    plt.figure(figsize=(6, 3))
    plt.subplot(121), plt.imshow(data, cmap='gray', vmin=0, vmax=255), plt.title("Original"), plt.axis('off')
    # plt.subplot(132, xticks=[], yticks=[]), plt.axis([0, 255, 0, np.max(histCV)])
    # plt.bar(range(256), histCV[:, 0]), plt.title("Gray Hist(cv2.calcHist)")
    plt.subplot(122, xticks=[], yticks=[]), plt.axis([0, 255, 0, np.max(histNP)])
    plt.bar(bins[:-1], histNP), plt.title("Gray Hist(np.histogram)")
    plt.show()