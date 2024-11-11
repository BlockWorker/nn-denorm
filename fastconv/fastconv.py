import cython
from cython.parallel import prange
import numpy as np
from cython.cimports.libc.stdlib import abs as cabs
from cython.cimports.openmp import omp_get_thread_num


flush_counts = cython.declare(cython.ulonglong[128], [0] * 128)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cfunc
@cython.inline
@cython.nogil
def fz(x: cython.float, flush: cython.int) -> cython.float:
    global flush_counts
    if flush == 0:
        return x
    i: cython.int = cython.cast(cython.pointer(cython.int), cython.address(x))[0]
    e: cython.int = (i & 0x7F800000) >> 23
    if e < flush:
        if (e > 0) or ((i & 0x007FFFFF) != 0):
            tn: cython.int = omp_get_thread_num()
            flush_counts[tn] += 1
        return 0
    else:
        return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def fz_arr(x: np.ndarray, flush: int) -> np.ndarray:
    if flush == 0:
        return x
    shape = x.shape
    _flush: cython.int = flush
    _x: cython.float[:] = np.reshape(x, (-1))
    _y: cython.float[:] = np.empty_like(_x)
    _len: cython.Py_ssize_t = len(_x)
    i: cython.Py_ssize_t
    for i in prange(_len, nogil=True):
        _y[i] = fz(_x[i], _flush)
    return np.reshape(_y, shape)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def get_flush_count(clear=False):
    global flush_counts
    count = 0
    for i in range(128):
        count += flush_counts[i]
    if clear:
        for i in range(128):
            flush_counts[i] = 0
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.ccall
def tiled_matmul(a, b, flush=0):
    _a: cython.float[:, :] = a
    _b: cython.float[:, :] = np.transpose(b)
    _flush: cython.int = flush
    incr: cython.Py_ssize_t = 64
    rows: cython.Py_ssize_t = a.shape[0]
    cols: cython.Py_ssize_t = b.shape[1]
    inner: cython.Py_ssize_t = a.shape[1]
    assert inner == b.shape[0]
    res: cython.float[:, :] = np.zeros((rows, cols), dtype="float32")
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    km: cython.Py_ssize_t = (inner + incr - 1) // incr
    k: cython.Py_ssize_t
    _k: cython.Py_ssize_t
    x: cython.Py_ssize_t
    y: cython.Py_ssize_t
    z: cython.Py_ssize_t
    xm: cython.Py_ssize_t
    ym: cython.Py_ssize_t
    zm: cython.Py_ssize_t
    s: cython.float

    for i in prange(0, rows, incr, nogil=True):
        xm = min(i + incr, rows)
        for j in prange(0, cols, incr):
            ym = min(j + incr, cols)
            for _k in range(km):
                k = _k * incr
                zm = min(k + incr, inner)
                for x in range(i, xm):
                    for y in range(j, ym):
                        s = 0
                        for z in range(k, zm):
                            s = fz(s + fz(_a[x, z] * _b[y, z], _flush), _flush)
                        res[x, y] = fz(res[x, y] + s, _flush)
    return res.base


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def kn2row(inputs, kernel, mode="same", strides=(1, 1), flush=0):
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
    _flush: cython.int = flush
    prod: cython.float[:, :] = tiled_matmul(kern_mat, in_mat, flush)
    result: cython.float[:, :] = np.zeros((n_f, n * h_p * w_p), dtype='float32')
    samp_width: cython.Py_ssize_t = h_p * w_p  # width of single sample of batch within product/result matrix row
    _n: cython.Py_ssize_t = n
    _kh: cython.Py_ssize_t = kh
    _kw: cython.Py_ssize_t = kw
    _w_p: cython.Py_ssize_t = w_p
    _n_f: cython.Py_ssize_t = n_f
    s: cython.Py_ssize_t
    samp_off: cython.Py_ssize_t
    y: cython.Py_ssize_t
    y_off: cython.Py_ssize_t
    x: cython.Py_ssize_t
    total_off: cython.Py_ssize_t
    prod_off: cython.Py_ssize_t
    res_start: cython.Py_ssize_t
    prod_start: cython.Py_ssize_t
    si: cython.Py_ssize_t
    fi: cython.Py_ssize_t
    for s in prange(_n, nogil=True):  # samples need separate handling
        samp_off = s * samp_width  # offset of sample within product+result matrices
        for y in range(_kh):
            y_off = (y - ((_kh - 1) // 2)) * _w_p  # partial offset of these mask pixels in product row
            for x in range(_kw):
                total_off = y_off + x - ((_kw - 1) // 2)  # total offset of this mask pixel in product row
                prod_off = (y * _kw + x) * _n_f  # product offset in column
                if total_off < 0:
                    res_start = samp_off - total_off
                    prod_start = samp_off
                else:
                    res_start = samp_off
                    prod_start = samp_off + total_off
                for fi in range(_n_f):
                    for si in range(samp_width - cabs(total_off)):
                        result[fi, res_start+si] = fz(result[fi, res_start+si] + prod[prod_off+fi, prod_start+si], _flush)
    s_h = pad_h + 1 if str_h > 1 else pad_h
    s_w = pad_w + 1 if str_w > 1 else pad_w
    return np.array(result).reshape((n_f, n, h_p, w_p)).transpose((1, 2, 3, 0))[:, s_h:-pad_h:str_h, s_w:-pad_w:str_w, :]
