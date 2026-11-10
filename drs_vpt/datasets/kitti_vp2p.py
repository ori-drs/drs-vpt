# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import math
import os
import random

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch
import torch.utils.data as data


class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]


class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int64)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i + 1], pts))
        return farthest_pts, farthest_pts_idx


class kitti_pc_img_dataset(data.Dataset):
    def __init__(self, root_path, mode, num_pc,
                 P_tx_amplitude=10, P_ty_amplitude=0, P_tz_amplitude=10,
                 P_Rx_amplitude=0, P_Ry_amplitude=2.0 * math.pi, P_Rz_amplitude=0, num_kpt=512, is_front=False):
        super(kitti_pc_img_dataset, self).__init__()
        self.root_path = root_path
        self.mode = mode
        self.dataset = self.make_kitti_dataset(root_path, mode)
        self.calibhelper = KittiCalibHelper(root_path)
        self.num_pc = num_pc
        self.img_H = 160
        self.img_W = 512

        self.P_tx_amplitude = P_tx_amplitude
        self.P_ty_amplitude = P_ty_amplitude
        self.P_tz_amplitude = P_tz_amplitude
        self.P_Rx_amplitude = P_Rx_amplitude
        self.P_Ry_amplitude = P_Ry_amplitude
        self.P_Rz_amplitude = P_Rz_amplitude
        self.num_kpt = num_kpt
        self.farthest_sampler = FarthestSampler(dim=3)

        self.node_a_num = 128
        self.node_b_num = 128
        self.is_front = is_front
        print('load data complete')

    def make_kitti_dataset(self, root_path, mode):
        dataset = []

        if mode == 'train':
            seq_list = list(range(9))
        elif 'val' == mode:
            seq_list = [9, 10]
        elif 'tiny_val' == mode:
            seq_list = [9]
        else:
            raise Exception('Invalid mode.')

        skip_start_end = 0
        for seq in seq_list:
            img2_folder = os.path.join(root_path, 'sequences', '%02d' % seq, 'image_2')
            img3_folder = os.path.join(root_path, 'sequences', '%02d' % seq, 'image_3')
            pc_folder = os.path.join(root_path, 'sequences', '%02d' % seq, 'voxel0.1-SNr0.6')

            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                seq, i, 'P3', sample_num))
        return dataset

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def generate_random_transform(self):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def __len__(self):
        return len(self.dataset)

    def load_raw_velo(self, pc_folder, seq_i):
        scan_path = os.path.join(pc_folder, '%06d.bin' % seq_i)

        scan = np.fromfile(scan_path,
                           dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan[:, :3].T

    def __getitem__(self, index):
        img_folder, pc_folder, seq, seq_i, key, seq_sample_num = self.dataset[index]
        raw_cloud_folder = pc_folder.replace("voxel0.1-SNr0.6", "velodyne")

        # ----------------------load image and points----------------
        img_path = os.path.join(img_folder, '%06d.png' % seq_i)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # HxWx3

        # load point cloud of seq_i
        P_Tr = np.dot(self.calibhelper.get_matrix(seq, key),
                      self.calibhelper.get_matrix(seq, 'Tr'))
        pc = self.load_raw_velo(raw_cloud_folder, seq_i)
        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]

        # -----------------get in_matrix --------------------------
        K = self.calibhelper.get_matrix(seq, key + '_K')

        P = self.generate_random_transform()

        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]

        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc': torch.from_numpy(pc.astype(np.float32)),
                'K': torch.from_numpy(K.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),
                'img_path': img_path
                }
