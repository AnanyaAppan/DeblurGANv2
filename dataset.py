import os
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm

import aug


def subsample(data: Iterable, bounds: Tuple[float, float], hash_fn: Callable, n_buckets=100, salt='', verbose=True):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}'
    if salt:
        msg += f'; salt is {salt}'
    if verbose:
        logger.info(msg)
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])


def hash_from_paths(x: Tuple[str, str], salt: str = '') -> str:
    path_a, path_b, path_c = x
    names = ''.join(map(os.path.basename, (path_a, path_b, path_c)))
    return sha1(f'{names}_{salt}'.encode()).hexdigest()


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt=''):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


def _read_img(x: str):
    img = imread(x)
    if img is None:
        logger.warning(f'Can not read image {x} with OpenCV, switching to scikit-image')
        img = imread(x)
    return img


class PairedDataset(Dataset):
    def __init__(self,
                 files_a: Tuple[str],
                 files_b: Tuple[str],
                 files_c: Tuple[str],
                 transform_fn: Callable,
                 normalize_fn: Callable,
                 corrupt_fn: Optional[Callable] = None,
                 preload: bool = True,
                 preload_size: Optional[int] = 0,
                 verbose=True):

        assert len(files_a) == len(files_b)
        assert len(files_a) == len(files_c)

        self.preload = preload
        self.data_a = files_a
        self.data_b = files_b
        self.data_c = files_c

        self.verbose = verbose
        self.corrupt_fn = corrupt_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        logger.info(f'Dataset has been created with {len(self.data_a)} samples')

        if preload:
            preload_fn = partial(self._bulk_preload, preload_size=preload_size)
            if files_a == files_b:
                self.data_a = self.data_b = preload_fn(self.data_a)
            else:
                self.data_a, self.data_b, self.data_c = map(preload_fn, (self.data_a, self.data_b, self.data_c))
            self.preload = True

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = [delayed(self._preload)(x, preload_size=preload_size) for x in data]
        jobs = tqdm(jobs, desc='preloading images', disable=not self.verbose)
        return Parallel(n_jobs=cpu_count(), backend='threading')(jobs)

    @staticmethod
    def _preload(x: str, preload_size: int):
        img = _read_img(x)
        if preload_size:
            h, w, *_ = img.shape
            h_scale = preload_size / h
            w_scale = preload_size / w
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
            assert min(img.shape[:2]) >= preload_size, f'weird img shape: {img.shape}'
        return img

    def _preprocess(self, img, res, attention_map, downsampled_attention_map):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))
        return map(transpose, list(self.normalize_fn(img, res)) + [attention_map] + [downsampled_attention_map])

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        blurred_s3, sharp_s3, amap_s3 = self.data_a[idx], self.data_b[idx], self.data_c[idx]
        if not self.preload:
            blurred_s3, sharp_s3, amap_s3 = map(_read_img, (blurred_s3, sharp_s3, amap_s3))
        blurred_s3, sharp_s3, amap_s3 = self.transform_fn(blurred_s3, sharp_s3, amap_s3)
        blurred_s2 = cv2.resize(blurred_s3,fx=0.5,fy=0.5, dsize=None)
        blurred_s1 = cv2.resize(blurred_s3,fx=0.25,fy=0.25, dsize=None)
        sharp_s2 = cv2.resize(sharp_s3,fx=0.5,fy=0.5, dsize=None)
        sharp_s1 = cv2.resize(sharp_s3,fx=0.25,fy=0.25, dsize=None)
        amap_s2 = cv2.resize(amap_s3,fx=0.5,fy=0.5, dsize=None)
        amap_s1 = cv2.resize(amap_s3,fx=0.25,fy=0.25, dsize=None)
        d_amap_s2 = cv2.resize(amap_s3,fx=0.125,fy=0.125, dsize=None)
        d_amap_s1 = cv2.resize(amap_s3,fx=0.0625,fy=0.0625, dsize=None)
        amap_s3 = amap_s3.reshape(256,256,1)
        amap_s2 = amap_s2.reshape(128,128,1)
        amap_s1 = amap_s1.reshape(64,64,1)
        d_amap_s3 = amap_s1
        d_amap_s2 = d_amap_s2.reshape(32,32,1)
        d_amap_s1 = d_amap_s1.reshape(16,16,1)
        # if self.corrupt_fn is not None:
        #     a = self.corrupt_fn(a)
        blurred_s1, sharp_s1, amap_s1, d_amap_s1 = self._preprocess(blurred_s1, sharp_s1, amap_s1, d_amap_s1)
        blurred_s2, sharp_s2, amap_s2, d_amap_s2 = self._preprocess(blurred_s2, sharp_s2, amap_s2,  d_amap_s2)
        blurred_s3, sharp_s3, amap_s3, d_amap_s3 = self._preprocess(blurred_s3, sharp_s3,  amap_s3, d_amap_s3)
        return {'amap_s1' : amap_s1, 'blurred_s1' : blurred_s1, 'sharp_s1' : sharp_s1, 'd_amap_s1' : d_amap_s1,
                'amap_s2' : amap_s2, 'blurred_s2' : blurred_s2, 'sharp_s2' : sharp_s2, 'd_amap_s2' : d_amap_s2,
                'amap_s3' : amap_s3, 'blurred_s3' : blurred_s3, 'sharp_s3' : sharp_s3, 'd_amap_s3' : d_amap_s3,
        }

    @staticmethod
    def from_config(config):
        config = deepcopy(config)
        files_a, files_b, files_c = map(lambda x: sorted(glob(config[x], recursive=True)), ('files_a', 'files_b', 'files_c'))
        transform_fn = aug.get_transforms(size=config['size'], scope=config['scope'], crop=config['crop'])
        normalize_fn = aug.get_normalize()
        corrupt_fn = aug.get_corrupt_function(config['corrupt'])

        hash_fn = hash_from_paths
        # ToDo: add more hash functions
        verbose = config.get('verbose', True)
        data = subsample(data=zip(files_a, files_b, files_c),
                         bounds=config.get('bounds', (0, 1)),
                         hash_fn=hash_fn,
                         verbose=verbose)

        files_a, files_b, files_c = map(list, zip(*data))

        return PairedDataset(files_a=files_a,
                             files_b=files_b,
                             files_c=files_c,
                             preload=config['preload'],
                             preload_size=config['preload_size'],
                             corrupt_fn=corrupt_fn,
                             normalize_fn=normalize_fn,
                             transform_fn=transform_fn,
                             verbose=verbose)
