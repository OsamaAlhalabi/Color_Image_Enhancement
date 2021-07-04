import numpy as np
import cv2 as cv

import utils
from helper import Helper


class ImageEnhancement:
    def __init__(self, img, m, n, gamma, colored):
        if colored:  # in case of colored image we need to separate each color layer from each other.
            layer_b, layer_g, layer_r = cv.split(img)
            self.e_layer_b = utils.translate(layer_b, 0, 255, -1, 1)
            self.e_layer_g = utils.translate(layer_g, 0, 255, -1, 1)
            self.e_layer_r = utils.translate(layer_r, 0, 255, -1, 1)
            self.img = utils.real_scalar_multiplication(1 / 3, utils.add(utils.add(self.e_layer_b, self.e_layer_g), self.e_layer_r))

        else:  # in case of gray scale image; all the image consider as one layer.
            self.img = utils.translate(img, 0, 255, -1, 1)

        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.m = m
        self.n = n
        self.gamma = gamma  # fuzzification, defuzzification degree.

        self.helper = Helper(self.width, self.height, self.m, self.n)
        self.membership_matrix = self._calc_membership_matrix()

        self.means = np.zeros((m, n))
        self.variances = np.zeros((m, n))
        self.lambdas = np.zeros((m, n))  # lambda -> change in the image contrast.
        self.taos = np.zeros((m, n))     # tao -> change in the image brightness.

    def _calc_ps(self):  # fuzzy partitions (fuzzy sets) of the support D.
        res = np.zeros((self.m, self.n, self.width, self.height))
        for i in range(self.m):
            for j in range(self.n):
                for k in range(self.width):
                    for l in range(self.height):
                        res[i, j, k, l] = self.helper.pij(i, j, k, l)
        return res

    # The membership degrees of a point (x,y) of D ->to-> the fuzzy window Wij
    def _calc_membership_matrix(self):

        ps = self._calc_ps()

        res = np.zeros((self.m, self.n, self.width, self.height))

        for i in range(self.m):
            for j in range(self.n):
                num = np.power(ps[i, j], self.gamma)
                for k in range(self.width):
                    for l in range(self.height):
                        denom = np.sum(ps[:, :, k, l]) + utils.eps
                        res[i, j, k, l] = num[k, l] / denom
        return res

    def _card(self, w):
        return np.sum(w)

    def _fuzzy_mean(self, w, i, j):  # we use the mean values to enhance the brightness --> where tao = -1 * mean
        mean = 0
        for k in range(self.height):
            for l in range(self.width):
                mean = utils.add(mean, utils.real_scalar_multiplication(w[i, j, l, k], self.img[k, l]))
        mean /= self._card(w[i, j])
        return mean

    def _fuzzy_variance(self, w, i, j):  # we use the variance values to enhance the contrast --> where lambda = sqrt(
        # 1/3)/variance.
        variance = 0
        for k in range(self.height):
            for l in range(self.width):
                variance += w[i, j, l, k] * np.power(utils.norm(utils.sub(self.img[k, l], self.means[i, j])), 2)
        variance /= self._card(w[i, j])
        return variance

    def _lambda_(self, i, j):
        return np.sqrt(1 / 3) / (np.sqrt(self.variances[i, j]) + utils.eps)

    def _scan(self):  # calculate mean, variance & lambda for each window in one scan.
        for i in range(self.m):
            for j in range(self.n):
                self.means[i, j] = self._fuzzy_mean(self.membership_matrix, i, j)
                self.variances[i, j] = self._fuzzy_variance(self.membership_matrix, i, j)
                self.lambdas[i, j] = self._lambda_(i, j)
        self.taos = self.means.copy()

    def _window_enh(self, i, j, layer):
        # new window = lambda * (widows - mean) --> so simple ;)
        return utils.real_scalar_multiplication(self.lambdas[i, j], utils.sub(layer, self.taos[i, j]))

    def _apply_enhancement(self, layer):
        new_image = np.zeros(layer.shape)
        for i in range(self.m):
            for j in range(self.n):
                win = self._window_enh(i, j, layer)
                w1 = self.membership_matrix[i, j].T.copy()  # weight of membership
                for k in range(self.width):
                    for l in range(self.height):
                        new_image[l, k] = utils.add(new_image[l, k], utils.real_scalar_multiplication(w1[l, k], win[l, k]))
                        # note that the for each new window in the enhanced image
                        # = enhanced window * weight of its  membership.
        return new_image

    def _one_layer_enhancement(self, layer):  # gray scale images
        e_layer = self._apply_enhancement(layer)
        res_image = utils.translate(e_layer, -1, 1, 0, 255)
        res_image = np.round(res_image)
        res_image = res_image.astype(np.uint8)
        return res_image

    def _three_layers_enhancement(self):  # colored images
        res_r = self._one_layer_enhancement(self.e_layer_r)
        res_g = self._one_layer_enhancement(self.e_layer_g)
        res_b = self._one_layer_enhancement(self.e_layer_b)
        res_img = cv.merge([res_b, res_g, res_r])
        return res_img

    def enhance_gray_img(self, ):
        self._scan()
        result = self._one_layer_enhancement(self.img)
        return cv.cvtColor(result, cv.COLOR_BGR2RGB)

    def enhance_colored_img(self, ):
        self._scan()
        result = self._three_layers_enhancement()
        # return cv.cvtColor(result, cv.COLOR_BGR2RGB) #for ipynb
        return result  # for pyCharm
