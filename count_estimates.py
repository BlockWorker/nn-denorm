import numpy as np


def mm_count(r1, c1, r2, c2):
    assert c1 == r2
    return 2 * r1 * c1 * c2


def kn_count(bi, ri, ci, ni, rk, ck, nk, mode="same"):
    if mode == "same":
        rix = ri + 2 * ((rk - 1) // 2)
        cix = ci + 2 * ((ck - 1) // 2)
    else:
        rix = ri
        cix = ci
    rmi = ni
    cmi = bi * rix * cix
    rmk = rk * ck * nk
    cmk = ni
    count = mm_count(rmk, cmk, rmi, cmi)
    count += bi * rk * ck * nk * rix * cix * (rix * cix) // (ri * ci)
    return count


def c2d_count(bi, ri, ci, ni, rk, ck, nk, mode="same", strides=(1, 1)):
    count = kn_count(bi, ri, ci, ni, rk, ck, nk, mode)
    pad_h = (rk - 1) // 2
    pad_w = (ck - 1) // 2
    if mode == "same":
        rix = ri + 2 * pad_h
        cix = ci + 2 * pad_w
    else:
        rix = ri
        cix = ci
    str_h, str_w = strides
    s_h = pad_h + 1 if str_h > 1 else pad_h
    s_w = pad_w + 1 if str_w > 1 else pad_w
    ro = (rix - s_h - pad_h) // str_h
    co = (cix - s_w - pad_w) // str_w
    count += bi * ro * co * nk
    return count


def dense_count(bi, ni, nk):
    count = mm_count(bi, ni, ni, nk)
    count += bi * nk
    return count


modeltype = "resnet"
count = 0
flushcount = 18342683092
batchsize = 50
batchcount = 10000 // batchsize

if modeltype == "alexnet":
    count = batchcount * (
        c2d_count(batchsize, 32, 32, 3, 11, 11, 96, "same", (2, 2))
        + c2d_count(batchsize, 8, 8, 96, 5, 5, 192, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 192, 3, 3, 384, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 384, 3, 3, 256, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 256, 3, 3, 256, "same", (1, 1))
        + dense_count(batchsize, 4096, 4096)
        + dense_count(batchsize, 4096, 1024)
        + dense_count(batchsize, 1024, 10)
    )
elif modeltype == "vgg":
    count = batchcount * (
        c2d_count(batchsize, 32, 32, 3, 3, 3, 64, "same", (1, 1))
        + c2d_count(batchsize, 32, 32, 64, 3, 3, 64, "same", (1, 1))
        + c2d_count(batchsize, 16, 16, 64, 3, 3, 128, "same", (1, 1))
        + c2d_count(batchsize, 16, 16, 128, 3, 3, 128, "same", (1, 1))
        + c2d_count(batchsize, 8, 8, 128, 3, 3, 256, "same", (1, 1))
        + c2d_count(batchsize, 8, 8, 256, 3, 3, 256, "same", (1, 1))
        + c2d_count(batchsize, 8, 8, 256, 3, 3, 256, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 256, 3, 3, 512, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 512, 3, 3, 512, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 512, 3, 3, 512, "same", (1, 1))
        + c2d_count(batchsize, 2, 2, 512, 3, 3, 512, "same", (1, 1))
        + c2d_count(batchsize, 2, 2, 512, 3, 3, 512, "same", (1, 1))
        + c2d_count(batchsize, 2, 2, 512, 3, 3, 512, "same", (1, 1))
        + dense_count(batchsize, 512, 512)
        + dense_count(batchsize, 512, 10)
    )
elif modeltype == "resnet":
    count = batchcount * (
        c2d_count(batchsize, 32, 32, 3, 3, 3, 64, "same", (1, 1))
        + c2d_count(batchsize, 32, 32, 64, 3, 3, 128, "same", (1, 1))
        + c2d_count(batchsize, 16, 16, 128, 3, 3, 128, "same", (1, 1))
        + c2d_count(batchsize, 16, 16, 128, 3, 3, 128, "same", (1, 1))
        + c2d_count(batchsize, 16, 16, 128, 3, 3, 256, "same", (1, 1))
        + c2d_count(batchsize, 8, 8, 256, 3, 3, 512, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 512, 3, 3, 512, "same", (1, 1))
        + c2d_count(batchsize, 4, 4, 512, 3, 3, 512, "same", (1, 1))
        + dense_count(batchsize, 2048, 10)
    )
else:
    exit(1)

print("Count:", count)
print("Flush %:", flushcount / count * 100)
