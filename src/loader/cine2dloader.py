#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aya

script for Local All Pass Attention Network (LAPANet)

Developed at the University Hospital of Tübingen.
Copyright © 2026 University Hospital of Tübingen.

If you'd like to use or share this code, please get in touch with
Aya Ghoul <aya.ghoul@med.uni-tuebingen.de>.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import random
import traceback
from random import shuffle

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from utils import ifft2c, fft2c, crop, zpad


# ---------------------------------------------------------------------------
# HDF5 / MATLAB v7.3 helpers
# ---------------------------------------------------------------------------
def _to_complex(arr):
    """Convert an HDF5-stored complex array to numpy complex64.

    MATLAB v7.3 stores complex arrays as a compound dtype with fields
    'real' and 'imag'. Plain float arrays are cast directly.
    """
    if arr.dtype.names is not None and {'real', 'imag'} <= set(arr.dtype.names):
        return (arr['real'] + 1j * arr['imag']).astype(np.complex64)
    return arr.astype(np.complex64)


def _read_h5_key(h5file, key):
    """Read a dataset from an HDF5 file, handling MATLAB column-major layout.

    MATLAB stores arrays in Fortran order; h5py returns C order. For 2D+
    arrays we swap the last two axes. If the array is stored as a reference
    or wrapped in a 'data' subgroup, we follow it.
    """
    node = h5file[key]

    # Some MATLAB files wrap the array in a group with a 'data' dataset.
    if isinstance(node, h5py.Group):
        if 'data' in node:
            node = node['data']
        else:
            raise KeyError(
                f"Key '{key}' is a group without a 'data' dataset. "
                f"Children: {list(node.keys())}"
            )

    data = node[()]
    if isinstance(data, h5py.Reference):
        data = h5file[data][()]

    # MATLAB column-major -> swap last two axes
    if hasattr(data, 'ndim') and data.ndim >= 2:
        data = np.swapaxes(data, -1, -2)
    return data


def _list_h5_keys(h5file):
    """Recursively list all dataset keys in an HDF5 file.

    Falls back to top-level keys if ``visititems`` fails due to B-tree
    corruption, so a partially-corrupt file can still be inspected.
    """
    keys = []
    try:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)
        h5file.visititems(_visit)
        if keys:
            return keys
    except (RuntimeError, OSError) as e:
        print(f'[CMRxRecon]   WARNING: visititems failed ({e}); '
              f'falling back to top-level keys')

    # Fallback: only top-level names (may include groups)
    try:
        return list(h5file.keys())
    except Exception:
        return []


def _find_h5_key(h5file, candidates, required=True):
    """Return the first key in ``candidates`` that exists in ``h5file``.

    Matching is case-insensitive and tolerates naming variants
    (e.g. 'kspace', 'kspace_full', 'KSPACE', 'kspace_full_1', ...).
    """
    all_keys = _list_h5_keys(h5file)
    if not all_keys:
        if required:
            raise KeyError('HDF5 file has no readable keys (corrupted?)')
        return None

    lowered = {k.lower(): k for k in all_keys}

    for cand in candidates:
        c = cand.lower()
        if c in lowered:
            return lowered[c]
        # prefix / substring match
        for lk, real in lowered.items():
            if lk.startswith(c) or c in lk:
                return real

    if required:
        raise KeyError(
            f"None of {candidates} found in HDF5 file. "
            f"Available keys: {all_keys}"
        )
    return None


def _safe_load_kspace(mat_path, candidates, required=True, tag=''):
    """Open ``mat_path``, find a key among ``candidates``, return complex array.

    Returns ``None`` on any failure (corrupted file, missing key, ...) instead
    of raising, so the caller can skip and continue.
    """
    if mat_path is None or not os.path.isfile(mat_path):
        print(f'[CMRxRecon]   SKIP {tag}: file not found ({mat_path})')
        return None

    try:
        with h5py.File(mat_path, 'r') as f:
            keys = _list_h5_keys(f)
            print(f'[CMRxRecon]   {tag}: keys={keys}')

            key = _find_h5_key(f, candidates, required=required)
            if key is None:
                print(f'[CMRxRecon]   SKIP {tag}: none of {candidates} found')
                return None

            arr = _to_complex(_read_h5_key(f, key))
            print(f'[CMRxRecon]   {tag}: read "{key}" shape={arr.shape}')
            return arr
    except (OSError, RuntimeError, KeyError, ValueError) as e:
        print(f'[CMRxRecon]   SKIP {tag}: failed to read {mat_path} ({e})')
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CMRxReconCine2DDataset(Dataset):
    """Robust loader for the CMRxRecon dataset.

    Layout::

        <data_root_dir>/
            TrainingSet/
                AccFactor04/  <SubjectID>/  cine_sax.mat, cine_lax.mat
                AccFactor08/  <SubjectID>/  cine_sax.mat, cine_lax.mat
                AccFactor10/  <SubjectID>/  cine_sax.mat, cine_lax.mat
                FullSample/   <SubjectID>/  cine_sax.mat, cine_lax.mat

    Each AccFactorXX file contains only its own sub-sampled k-space
    (e.g. ``kspace_sub04``), while ``FullSample`` contains ``kspace_full``.
    The fully-sampled reference is therefore always read from the
    ``FullSample`` folder of the *same subject*.
    """

    ACC_KEY = {4: ['kspace_sub04', 'kspace_sub4', 'sub04', 'sub4'],
               8: ['kspace_sub08', 'kspace_sub8', 'sub08', 'sub8'],
               10: ['kspace_sub10', 'sub10']}
    ACC_DIR = {4: 'AccFactor04', 8: 'AccFactor08', 10: 'AccFactor10'}
    FULL_DIR = 'FullSample'
    FULL_KEY_CANDIDATES = ['kspace_full', 'kspace', 'full_kspace',
                           'kspacefull', 'kspaceFull']

    def __init__(self, config):
        # ---- generic config -------------------------------------------------
        self.mode = config.mode
        self.data_root_dir = config.data_dir

        self.out_shape = config.out_shape

        self.bounding_box_dir = getattr(config, 'bounding_box_dir', None)
        self.R_list = config.R_list
        self.csv_names = getattr(config, 'csv_names', None)

        # ---- CMRxRecon-specific config -------------------------------------
        self.view = getattr(config, 'view', 'sax')          # 'sax'|'lax'|'both'
        self.use_all_accelerations = getattr(config, 'use_all_accelerations', False)
        self.subject_list = getattr(config, 'subject_list', None)
        self.pair_frames = getattr(config, 'pair_frames', False)

        self.data_amount = 2 if config.mode == 'debug' else 20000000

        # ---- state ---------------------------------------------------------
        self.list_info = []
        self.ksp_us_list = []
        self.img_fully_list = []
        self.box_list = []

        # ---- kick off ------------------------------------------------------
        print(f'[CMRxRecon] start loading data in mode "{self.mode}" ...')
        print(f'[CMRxRecon] data root: {self.data_root_dir}')
        print(f'[CMRxRecon] view: {self.view}')
        print(f'[CMRxRecon] R_list: {self.R_list}')
        print(f'[CMRxRecon] max loaded data amount is {self.data_amount}')
        self.create_list_data()
        print(f'[CMRxRecon] {len(self.list_info)} samples in list_info '
              f'({len(self.ksp_us_list)} volumes loaded)')

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------
    def _scan_subjects(self):
        """Return {subject_id: {acc:int|'full' -> {view: mat_path}}}."""
        train_root = os.path.join(self.data_root_dir, 'TrainingSet')
        if not os.path.isdir(train_root):
            train_root = self.data_root_dir   # assume data_dir *is* TrainingSet

        subjects = {}

        def _register(subject_id, acc, view, mat_path):
            subjects.setdefault(subject_id, {}).setdefault(acc, {})[view] = mat_path

        def _scan_folder(folder_path, acc):
            if not os.path.isdir(folder_path):
                print(f'[CMRxRecon] WARNING: missing folder {folder_path}')
                return
            for subj in sorted(os.listdir(folder_path)):
                subj_path = os.path.join(folder_path, subj)
                if not os.path.isdir(subj_path):
                    continue
                for view in self._views_to_load():
                    mat_path = os.path.join(subj_path, f'cine_{view}.mat')
                    if os.path.isfile(mat_path):
                        _register(subj, acc, view, mat_path)

        for acc, folder in self.ACC_DIR.items():
            _scan_folder(os.path.join(train_root, folder), acc)

        _scan_folder(os.path.join(train_root, self.FULL_DIR), 'full')

        if self.subject_list is not None:
            subjects = {k: v for k, v in subjects.items()
                        if k in self.subject_list}

        return subjects

    def _views_to_load(self):
        return ['sax', 'lax'] if self.view == 'both' else [self.view]

    # ------------------------------------------------------------------
    # Main data list construction
    # ------------------------------------------------------------------
    def create_list_data(self):
        subjects = self._scan_subjects()
        subj_ids = sorted(subjects.keys())
        shuffle(subj_ids)

        print(f'[CMRxRecon] found {len(subj_ids)} subjects')
        if len(subj_ids) == 0:
            raise RuntimeError(
                f'No CMRxRecon subjects found under {self.data_root_dir}.'
            )

        n = 0
        n_skipped = 0
        for subj_id in subj_ids:
            if n > self.data_amount - 1:
                break

            acc_to_views = subjects[subj_id]   # {4: {'sax': path}, 'full': {...}}
            full_views = acc_to_views.get('full', {})

            # Which accelerations are available for this subject?
            available_accs = [a for a in self.R_list
                              if a in acc_to_views and a != 'full']
            if 1 in self.R_list and full_views:
                available_accs.append('full')
            available_accs = list(dict.fromkeys(available_accs))

            if not available_accs:
                print(f'[CMRxRecon] skip {subj_id}: no matching acceleration '
                      f'(have {list(acc_to_views.keys())})')
                continue

            accs_this_subj = (available_accs if self.use_all_accelerations
                              else [random.choice(available_accs)])

            for acc in accs_this_subj:
                if n > self.data_amount - 1:
                    break

                # Pick one view per (subject, acc). If 'both', try both views.
                views_here = self._views_to_load()
                for view in views_here:
                    if n > self.data_amount - 1:
                        break
                    try:
                        ok = self._load_one(subj_id, acc, view,
                                            acc_to_views, full_views)
                    except Exception as e:
                        # Never let one bad file kill the whole loader
                        print(f'[CMRxRecon] ERROR loading subj={subj_id} '
                              f'acc={acc} view={view}: {e}')
                        traceback.print_exc()
                        ok = False

                    if ok:
                        n += 1
                    else:
                        n_skipped += 1

        print(f'[CMRxRecon] loaded {n} volumes, skipped {n_skipped}')

    # ------------------------------------------------------------------
    # Single-volume loader
    # ------------------------------------------------------------------
    def _load_one(self, subj_id, acc, view, acc_to_views, full_views):
        """Load one (subject, acc, view) volume. Returns True on success."""
        # ---- figure out which files to read --------------------------------
        if acc == 'full':
            sub_path = full_views.get(view, None)
        else:
            sub_path = acc_to_views.get(acc, {}).get(view, None)

        full_path = full_views.get(view, None)

        if sub_path is None:
            print(f'[CMRxRecon] SKIP {subj_id} acc={acc} view={view}: '
                  f'no sub-sampled file')
            return False
        if full_path is None:
            print(f'[CMRxRecon] SKIP {subj_id} acc={acc} view={view}: '
                  f'no FullSample reference')
            return False

        print(f'[CMRxRecon] loading subj={subj_id} acc={acc} view={view}')

        # ---- read sub-sampled k-space -------------------------------------
        if acc == 'full':
            sub_candidates = self.FULL_KEY_CANDIDATES
        else:
            sub_candidates = self.ACC_KEY.get(acc, []) + self.FULL_KEY_CANDIDATES

        masked_kspace = _safe_load_kspace(
            sub_path, sub_candidates,
            required=True, tag=f'sub({subj_id},{acc},{view})',
        )
        if masked_kspace is None:
            return False

        # ---- read fully-sampled k-space -----------------------------------
        k_space_fully = _safe_load_kspace(
            full_path, self.FULL_KEY_CANDIDATES,
            required=True, tag=f'full({subj_id},{view})',
        )
        if k_space_fully is None:
            return False

        # ---- normalise to (S, C, T, H, W) ---------------------------------
        try:
            k_space_fully = self._to_5d(k_space_fully)
            masked_kspace = self._to_5d(masked_kspace)
        except ValueError as e:
            print(f'[CMRxRecon] SKIP {subj_id} acc={acc} view={view}: '
                  f'shape mismatch ({e})')
            return False

        # ---- reconstruct fully-sampled coil images ------------------------
        img_fully_coil = ifft2c(k_space_fully, axes=[-2, -1])   # (S,C,T,H,W)
        imgccoil = np.abs(img_fully_coil).astype(np.float32)

        # ---- optional bounding box ----------------------------------------
        n_slices = imgccoil.shape[0]
        if self.bounding_box_dir is not None:
            box_path = os.path.join(self.bounding_box_dir, f'{subj_id}.npy')
        else:
            box_path = None

        if box_path is not None and os.path.isfile(box_path):
            try:
                box_info = np.load(box_path).astype(np.uint8)
                box = self.get_bounding_box(box_info, offset=15)
            except Exception as e:
                print(f'[CMRxRecon]   WARNING: bad bbox for {subj_id} ({e}); '
                      f'using full-FOV box')
                box = np.ones((n_slices, *self.out_shape), dtype=np.uint8)
        else:
            box = np.ones((n_slices, *self.out_shape), dtype=np.uint8)

        # ---- append -------------------------------------------------------
        idx = len(self.ksp_us_list)
        self.img_fully_list.append(imgccoil)
        self.ksp_us_list.append(masked_kspace)
        self.box_list.append(box)

        self.fill_lists(idx)
        return True

    # ------------------------------------------------------------------
    # Shape normalisation
    # ------------------------------------------------------------------
    def _to_5d(self, ksp):
        """Normalise a k-space array to (S, C, T, H, W).

        Accepts:
          4D: (C, T, H, W)         -> (1, C, T, H, W)
          5D: (S, C, T, H, W)      -> unchanged
          5D: (C, S, T, H, W)      -> swap S and C
        Raises ValueError if the spatial dims don't match ``out_shape``.
        """
        ksp = np.transpose(ksp, (1, 2, 0, 3, 4))

        # Pad / crop spatial dims to out_shape
        target = self.out_shape
        if ksp.shape[-2] != target[-2] or ksp.shape[-1] != target[-1]:
            # use the existing crop/zpad utils
            S, C, T = ksp.shape[:3]
            flat = ksp.reshape(S, C * T, ksp.shape[-2], ksp.shape[-1])
            if flat.shape[-2] < target[-2] or flat.shape[-1] < target[-1]:
                flat = zpad(flat, (flat.shape[0], flat.shape[1], *target))
            else:
                flat = crop(flat, (flat.shape[0], flat.shape[1], *target))
            ksp = flat.reshape(S, C, T, target[-2], target[-1])
        return ksp

    # ------------------------------------------------------------------
    # Sample enumeration
    # ------------------------------------------------------------------
    def fill_lists(self, index_us):
        """Append (z, t1, t2) entries for one loaded volume."""
        n_slices = self.img_fully_list[index_us].shape[0]
        n_frames = self.img_fully_list[index_us].shape[2]

        for z in range(n_slices):
            for t1 in range(n_frames):
                for t2 in range(n_frames):
                    if len(self.list_info) >= self.data_amount:
                        return
                    self.list_info.append({
                            'img_us_idx': index_us,
                            'img_idx': index_us,
                            'z': z, 't1': t1, 't2': t2,
                        })

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        subject_dict = self.list_info[index]
        data = self.ksp_us_list[subject_dict['img_us_idx']]
        data_fully = self.img_fully_list[subject_dict['img_idx']]
        box = self.box_list[subject_dict['img_idx']]

        z, t1, t2 = subject_dict['z'], subject_dict['t1'], subject_dict['t2']

        img_ref_fully = data_fully[z, :, t1]
        img_mov_fully = data_fully[z, :, t2]
        img_ref_fully = torch.from_numpy(
            self.img_preprocessing(img_ref_fully)).float()
        img_mov_fully = torch.from_numpy(
            self.img_preprocessing(img_mov_fully)).float()

        k_ref = data[z, :, t1]
        k_mov = data[z, :, t2]
        k_ref = torch.from_numpy(self.ksp_preprocessing(k_ref))
        k_mov = torch.from_numpy(self.ksp_preprocessing(k_mov))

        box_z = self.img_preprocessing(box[z], scale=False)
        box_z = torch.from_numpy(box_z[None]).float()

        return {'k_ref_us': k_ref, 'k_mov_us': k_mov,
                'img_ref': img_ref_fully, 'img_mov': img_mov_fully,
                'box': box_z}

    # ------------------------------------------------------------------
    # Preprocessing (unchanged)
    # ------------------------------------------------------------------
    def ksp_preprocessing(self, ksp):
        img = ifft2c(ksp, axes=[-2, -1])
        shape = (ksp.shape[0], *self.out_shape)
        if img.shape[-2] < self.out_shape[-2] or img.shape[-1] < self.out_shape[-1]:
            img = zpad(img, shape)
        if img.shape[-2] > self.out_shape[-2] or img.shape[-1] > self.out_shape[-1]:
            img = crop(img, shape)
        mx = np.abs(img).max()
        if mx == 0:
            mx = 1.0
        ksp_pre = fft2c(img / mx, axes=[-2, -1])
        return ksp_pre

    def get_bounding_box(self, box_info, offset=5):
        n_slices = box_info.shape[0]
        H, W = self.out_shape[-2], self.out_shape[-1]
        box = np.zeros((n_slices, H, W), dtype=np.uint8)
        for z in range(n_slices):
            xmax = min(box_info[z, 1] + offset, H)
            xmin = max(box_info[z, 0] - offset, 0)
            ymax = min(box_info[z, 3] + offset, W)
            ymin = max(box_info[z, 2] - offset, 0)
            box[z, xmin:xmax, ymin:ymax] = 1
        return box

    def img_preprocessing(self, img, scale=True):
        shape_in = self.out_shape
        if len(img.shape) == 3:
            shape_in = (img.shape[0], *self.out_shape)
        if img.shape[-2] < self.out_shape[-2] or img.shape[-1] < self.out_shape[-1]:
            img = zpad(img, shape_in)
        if img.shape[-2] > self.out_shape[-2] or img.shape[-1] > self.out_shape[-1]:
            img = crop(img, shape_in)
        if scale:
            mn, mx = img.min(), img.max()
            if mx - mn > 1e-8:
                img = (img - mn) / (mx - mn)
        return img

    def __rmul__(self, v):
        self.list_info = v * self.list_info
        return self

    def __len__(self):
        return len(self.list_info)


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------
def fetch_cine_loader(args, loader_type='training'):
    train_dataset = CMRxReconCine2DDataset(config=args)
    shuffle_bool = False if loader_type == 'validation' else True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=shuffle_bool,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print('[CMRxRecon] Loader has %d image pairs' % len(train_dataset))
    return train_loader