import scipy.ndimage
import numpy as np
import torch
import h5py
import cv2


#h5 파일을 읽고 height, Width에 맞춰 인풋으로 뽑아내는 함수, H, W는 input의 H, W
def h5_DataAug(input_dir = str,label_dir = str, scale = 2):
    #20px
    f_in = h5py.File(input_dir, 'r')

    #10px
    f_out = h5py.File(label_dir, 'r')

    input = []
    label = []
    minmax_input = []

    for keys in f_in:
        #spline cubic interpolation input
        #zoomed_2d = scipy.ndimage.zoom(f_in[keys], zoom = 2, order = 2)
        #Bicubic interpolation input
        #temp = np.kron(f_in[keys], np.ones((1, 1)))
        #zoomed_2d = cv2.resize(temp, dsize=(200, 360), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        #kron input
        zoomed_2d = np.kron(f_in[keys], np.ones((2,2)))

        H, W = zoomed_2d.shape[0], zoomed_2d.shape[1]
        zoomed_1d = np.reshape(zoomed_2d, (-1))
        mi = min(zoomed_1d)
        Ma = max(zoomed_1d)
        #denormalize 위해 return
        minmax_input.append([mi,Ma])

        norm_in = np.reshape((zoomed_1d-mi)/(Ma-mi),(H, W))
        label_1d = np.reshape(f_out[keys],(-1))
        norm_la = np.reshape((label_1d-mi)/(Ma-mi), (H, W))

        input.append([norm_in])
        label.append([norm_la])

    return torch.tensor(input).float(), torch.tensor(label).float(), torch.tensor(minmax_input).float()