import time

import tensorflow as tf
import numpy as np
import scipy


def tf_ground_truth(inputs, kernel, mode="same", strides=(1, 1)):
    i = tf.constant(inputs, dtype=tf.float32)
    k = tf.constant(kernel, dtype=tf.float32)
    return tf.nn.conv2d(i, k, strides, mode.upper()).numpy()


def scipy_method(inputs, kernel, mode="same", strides=(1, 1)):
    # input: (n, h_i, w_i, c)
    # kernel: (kh, kw, c, n_f)
    # out: (n, h_o, w_o, n_f)
    n, h_i, w_i, c = inputs.shape
    kh, kw, _, n_f = kernel.shape
    if mode == "same":
        h_o = h_i
        w_o = w_i
    else:
        h_o = h_i - kh + 1
        w_o = w_i - kw + 1
    output = np.zeros((n, h_o, w_o, n_f), dtype='float32')

    for s in range(n):
        for f in range(n_f):
            for ch in range(c):
                output[s, :, :, f] += scipy.signal.correlate2d(inputs[s, :, :, ch], kernel[:, :, ch, f], mode)

    s_h = 1 if strides[0] > 1 else 0
    s_w = 1 if strides[1] > 1 else 0
    return output[:, s_h::strides[0], s_w::strides[1], :]


def simple(inputs, kernel, mode="same", strides=(1, 1)):
    # input: (n, h_i, w_i, c)
    # kernel: (kh, kw, c, n_f)
    # out: (n, h_o, w_o, n_f)
    n, h_i, w_i, c = inputs.shape
    kh, kw, _, n_f = kernel.shape
    str_h, str_w = strides
    if mode == "same":
        h_o = (h_i + str_h - 1) // str_h
        w_o = (w_i + str_w - 1) // str_w
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
        in_padded = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    else:
        h_o = (h_i - kh + str_h) // str_h
        w_o = (w_i - kw + str_w) // str_w
        in_padded = inputs
    output = np.zeros((n, h_o, w_o, n_f), dtype='float32')

    for i in range(h_o):
        for j in range(w_o):
            h_start = i * str_h + (1 if str_h > 1 else 0)
            h_end = h_start + kh
            w_start = j * str_w + (1 if str_w > 1 else 0)
            w_end = w_start + kw

            output[:, i, j, :] = np.sum(
                in_padded[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                kernel[np.newaxis, :, :, :, :],
                axis=(1, 2, 3)
            )

    return output


def kn2row(inputs, kernel, mode="same", strides=(1, 1)):
    # input: (n, h_i, w_i, c)
    # kernel: (kh, kw, c, n_f)
    # out: (n, h_o, w_o, n_f)
    n, h_i, w_i, c = inputs.shape
    kh, kw, _, n_f = kernel.shape
    str_h, str_w = strides
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    if mode == "same":
        in_padded = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    else:
        in_padded = inputs
    _, h_p, w_p, _ = in_padded.shape
    in_mat = in_padded.transpose((3, 0, 1, 2)).reshape((c, -1))  # c rows, n*h_i*w_i columns
    kern_mat = kernel.transpose((0, 1, 3, 2)).reshape((-1, c))  # kh*kw*n_f rows, c columns
    prod = np.matmul(kern_mat, in_mat)
    result = np.zeros((n_f, n * h_p * w_p), dtype='float32')
    samp_width = h_p * w_p  # width of single sample of batch within product/result matrix row
    for s in range(n):  # samples need separate handling
        samp_off = s * samp_width  # offset of sample within product+result matrices
        for y in range(kh):
            y_off = (y - ((kh - 1) // 2)) * w_p  # partial offset of these mask pixels in product row
            for x in range(kw):
                total_off = y_off + x - ((kw - 1) // 2)  # total offset of this mask pixel in product row
                prod_off = (y * kw + x) * n_f  # product offset in column
                if total_off < 0:
                    res_start = samp_off - total_off
                    res_end = samp_off + samp_width
                    prod_start = samp_off
                    prod_end = samp_off + samp_width + total_off
                else:
                    res_start = samp_off
                    res_end = samp_off + samp_width - total_off
                    prod_start = samp_off + total_off
                    prod_end = samp_off + samp_width
                result[:, res_start:res_end] += prod[prod_off:prod_off+n_f, prod_start:prod_end]
    s_h = pad_h + 1 if str_h > 1 else pad_h
    s_w = pad_w + 1 if str_w > 1 else pad_w
    return result.reshape((n_f, n, h_p, w_p)).transpose((1, 2, 3, 0))[:, s_h:-pad_h:str_h, s_w:-pad_w:str_w, :]


def run_test(n, h_i, w_i, c, kh, kw, n_f, mode, strides):
    a = np.random.normal(scale=2, size=(n, h_i, w_i, c))
    k = np.random.normal(scale=2, size=(kh, kw, c, n_f))
    start = time.perf_counter()
    gt = tf_ground_truth(a, k, mode, strides)
    gt_time = time.perf_counter() - start
    funcs = [scipy_method, simple, kn2row]
    others = []
    times = [gt_time]
    for f in funcs:
        start = time.perf_counter()
        res = f(a, k, mode, strides)
        times.append(time.perf_counter() - start)
        others.append(res)
    for i in range(len(others)):
        if others[i].shape != gt.shape:
            print("!!! Wrong shape at index", i)
    mxdif = max([np.max(np.abs(gt - x)) for x in others])
    mrdif = max([np.max(np.abs((gt - x) / gt)) for x in others])

    return gt, others, mxdif, mrdif, times


if __name__ == '__main__':
    n = 50
    h = 13
    w = 13
    c = 64
    kh = 5
    kw = 3
    n_f = 64
    for s in range(1, 4):
        std = (s, 1)
        print("same", std)
        t, o, md, mr, tm = run_test(n, h, w, c, kh, kw, n_f, "same", std)
        print("   ", md, mr, tm)
        print("valid", std)
        t, o, md, mr, tm = run_test(n, h, w, c, kh, kw, n_f, "valid", std)
        print("   ", md, mr, tm)
