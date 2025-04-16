import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
import cv2

# For inpainting
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import argparse
import io
import multiprocessing
from typing import Union

import torch

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from src.helper import (
    download_model,
    load_img,
    norm_img,
    numpy_to_bytes,
    pad_img_to_modulo,
    resize_max_size,
)

NUM_THREADS = str(multiprocessing.cpu_count())

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

from scipy import ndimage as ndi

SEAM_COLOR = np.array([255, 200, 200])
SHOULD_DOWNSIZE = True
DOWNSIZE_WIDTH = 500
ENERGY_MASK_CONST = 100000.0
MASK_THRESHOLD = 10
USE_FORWARD_ENERGY = True

# === Các hàm xử lý seam carving ===

def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

def resize(image, width):
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    image = image.astype('float32')
    return cv2.resize(image, dim)

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)

def backward_energy(im):
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))
    return grad_mag

def forward_energy(im):
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    U, L, R = np.roll(im, 1, axis=0), np.roll(im, 1, axis=1), np.roll(im, -1, axis=1)
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    for i in range(1, h):
        mU, mL, mR = m[i-1], np.roll(m[i-1], 1), np.roll(m[i-1], -1)
        mULR, cULR = np.array([mU, mL, mR]), np.array([cU[i], cL[i], cR[i]])
        mULR += cULR
        argmins = np.argmin(mULR, axis=0)
        m[i], energy[i] = np.choose(argmins, mULR), np.choose(argmins, cULR)
    return energy

def add_seam(im, seam_idx):
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.mean(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.mean(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
    return output

def add_seam_grayscale(im, seam_idx):
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.mean(im[row, col: col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = np.mean(im[row, col - 1: col + 1])
            output[row, : col] = im[row, : col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]
    return output

def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))

def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))

def get_minimum_seam(im, mask=None, remove_mask=None):
    h, w = im.shape[:2]
    M = forward_energy(im) if USE_FORWARD_ENERGY else backward_energy(im)
    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST
    if remove_mask is not None:
        M[np.where(remove_mask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100
    return compute_shortest_path(M, im, h, w)

def compute_shortest_path(M, im, h, w):
    backtrack = np.zeros_like(M, dtype=np.int_)
    for i in range(1, h):
        for j in range(w):
            idx, min_energy = (0, M[i - 1, j]) if j == 0 else (np.argmin(M[i - 1, j - 1:j + 2]), None)
            idx += j - 1 if j > 0 else 0
            min_energy = M[i - 1, idx] if j > 0 else min_energy
            backtrack[i, j] = idx
            M[i, j] += min_energy
    seam_idx, boolmask = [], np.ones((h, w), dtype=np.bool_)
    j = np.argmin(M[-1])
    for i in range(h - 1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]
    seam_idx.reverse()
    return np.array(seam_idx), boolmask

def seams_removal(im, num_remove, mask=None, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, mask)
        if vis: visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask

def seams_insertion(im, num_add, mask=None, vis=False, rot=False):
    seams_record = []
    temp_im, temp_mask = im.copy(), mask.copy() if mask is not None else None
    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, temp_mask)
        if vis: visualize(temp_im, boolmask, rotate=rot)
        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)
    seams_record.reverse()
    for seam in seams_record:
        im = add_seam(im, seam)
        if vis: visualize(im, rotate=rot)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2
    return im, mask

def seam_carve(im, dy, dx, mask=None, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    if mask is not None:
        mask = mask.astype(np.float64)
    output = im
    if dx < 0:
        output, mask = seams_removal(output, -dx, mask, vis)
    elif dx > 0:
        output, mask = seams_insertion(output, dx, mask, vis)
    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, mask, vis, rot=True)
        output = rotate_image(output, False)
    elif dy > 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_insertion(output, dy, mask, vis, rot=True)
        output = rotate_image(output, False)
    return output

def object_removal(im, rmask, mask=None, vis=False, horizontal_removal=False):
    im, rmask = im.astype(np.float64), rmask.astype(np.float64)
    if mask is not None:
        mask = mask.astype(np.float64)
    if horizontal_removal:
        im, rmask = rotate_image(im, True), rotate_image(rmask, True)
        if mask is not None:
            mask = rotate_image(mask, True)
    while len(np.where(rmask > MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(im, mask, rmask)
        if vis: visualize(im, boolmask, rotate=horizontal_removal)
        im = remove_seam(im, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    if horizontal_removal:
        im = rotate_image(im, False)
    return im

# === Hàm chính để sử dụng trong pipeline ===

# def process_inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     """
#     Xử lý inpaint ảnh bằng OpenCV: xóa vùng chữ và khôi phục nền (dùng Telea hoặc Navier-Stokes).
#     Giữ nguyên kích thước ảnh gốc.
#     """
#     assert img.shape[:2] == mask.shape[:2], "Kích thước ảnh và mask không khớp"

#     # Chuẩn hóa mask về dạng nhị phân 8-bit
#     mask = (mask > 0).astype(np.uint8) * 255

#     # Inpainting với thuật toán Telea hoặc Navier-Stokes
#     inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

#     return inpainted
def process_inpaint(image, mask):
    import cv2
    from src.helper import resize_max_size, norm_img
    from src.model import run  # Giả định model run ở đây

    print(f"📏 Original image shape: {image.shape}")

    # Bước 1: Đảm bảo ảnh là RGB
    if image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Bước 2: Resize ảnh để tránh giảm chất lượng
    size_limit = 1024  # hoặc 2048 nếu RAM cho phép
    image = resize_max_size(image, size_limit=size_limit, interpolation=cv2.INTER_CUBIC)
    image = norm_img(image)

    # Bước 3: Chuẩn hoá mask
    if mask.ndim == 2:
        pass  # Giữ nguyên mask dạng 2D
    elif mask.shape[2] == 4:  # Nếu là RGBA
        mask = mask[:, :, 3]
    else:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = resize_max_size(mask, size_limit=size_limit, interpolation=cv2.INTER_NEAREST)
    mask = norm_img(mask)

    # ✅ Gọi model để inpaint
    res_np_img = run(image, mask)

    # Chuyển ảnh kết quả về định dạng hiển thị chuẩn
    if res_np_img.shape[2] == 1:
        res_np_img = cv2.cvtColor(res_np_img, cv2.COLOR_GRAY2RGB)
    else:
        res_np_img = cv2.cvtColor(res_np_img, cv2.COLOR_BGR2RGB)

    return res_np_img

