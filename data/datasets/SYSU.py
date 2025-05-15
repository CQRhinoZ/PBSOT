import glob
import pdb
import pickle
import os.path as osp
import re
import warnings
import json
import random
import numpy as np
import pandas as pd
import xlrd
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 尝试加载截断的图像
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class SYSUGroup(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'SYSUGroup'
    dataset_name = 'SYSUGroup'

    def __init__(self, root='dataset', **kwargs):
        self.root = '/home/wzg/wusl/UMSOT-main/datasets'

        self.dataset_dir = self.root

        self.train_dir = osp.join(self.dataset_dir, 'SYSU', 'bounding_box_train')
        self.test_dir = osp.join(self.dataset_dir, 'SYSU', 'bounding_box_test')
        self.data = self.read_xls(self.root + '/SYSU/Person_ID&Group_ID_Released_v20210530_1.xls')

        train, query, gallery = self.process_data()

        super(SYSUGroup, self).__init__(train, query, gallery, **kwargs)

    def read_xls(self, file_path):
        workbook = xlrd.open_workbook(file_path)
        sheet_names = workbook.sheet_names()

        data = {}
        for sheet_name in sheet_names:
            sheet = workbook.sheet_by_name(sheet_name)
            df = pd.DataFrame(
                [sheet.row_values(row) for row in range(0, sheet.nrows)],
                columns=[f'col_{col}' for col in range(sheet.ncols)]
            )
            data[sheet_name] = df
        return data

    def process_data(self):
        train_person_df = self.data['train']
        test_person_df = self.data['test']
        train_group_df = self.data['train-groupID']
        test_group_df = self.data['test-groupID']

        train_data = self.process_dir(train_person_df, train_group_df, self.train_dir, camid=0, with_prefix=True)
        test_data = self.process_dir(test_person_df, test_group_df, self.test_dir, camid=1, with_prefix=False)

        query_data = [self.set_camid(entry, 1, with_prefix=False) for i, entry in enumerate(test_data) if i % 2 == 0]
        gallery_data = [self.set_camid(entry, 2, with_prefix=False) for i, entry in enumerate(test_data) if i % 2 != 0]

        print("train_data", train_data[100])
        print("query_data", query_data[100])
        print("gallery_data", gallery_data[100])

        return train_data, tuple(query_data), tuple(gallery_data)

    def set_camid(self, entry, camid, with_prefix=True):
        entry = list(entry)
        entry[3] = camid  # 保存为整数

        if not with_prefix and isinstance(entry[1], str):
            entry[1] = int(entry[1].replace(f'{self.dataset_name}_', ''))  # 移除 groupid 前缀
            entry[2] = [pid.replace(f'{self.dataset_name}_', '') for pid in eval(entry[2])]

        return tuple(entry)

    def process_dir(self, person_df, group_df, dir_path, camid, with_prefix):
        data = []

        # 创建一个映射以便快速查找group_df中的groupID
        group_map = {row['col_0']: int(row['col_1']) for _, row in group_df.iterrows()}

        for _, row in person_df.iterrows():
            img_name = row['col_0']
            pid = f"p{int(row['col_1'])}" if with_prefix else f"{int(row['col_1'])}"
            gid = group_map.get(img_name, None)

            if gid is None:
                warnings.warn(f"Group ID for image {img_name} not found in group_df. Skipping this image.")
                continue
            gid_int = gid  # 保存整数形式的 gid
            gid = f'{self.dataset_name}_{gid}' if with_prefix else gid_int

            # 解析图像名称以获取 camid 和 bbox
            parts = img_name.split('_')
            bbox_str = '_'.join(parts[3:]).replace('.jpg', '')
            bbox = list(map(int, bbox_str.split('_')))

            # 获取组图像路径
            group_img_name = '_'.join(parts[:3])
            group_img_path = osp.join(self.root, 'SYSUDB', f'{gid_int:03d}', group_img_name + '.png')

            try:
                # 检查图片是否损坏或不存在
                img = Image.open(group_img_path)
                img.verify()  # 验证图像文件
            except (FileNotFoundError, OSError, UnidentifiedImageError) as e:
                warnings.warn(f"Group image {group_img_path} not found or is corrupted. Skipping this image.")
                continue

            # 查找已有的 group_img_path 条目
            existing_entry_index = next((index for index, item in enumerate(data) if item[0] == group_img_path), None)
            if existing_entry_index is not None:
                # 更新已有条目
                existing_entry = list(data[existing_entry_index])
                pid_list = self.parse_pid_list(existing_entry[2], with_prefix)
                pid_list.append(pid)
                pid_str = self.format_pid_list(pid_list, with_prefix)
                existing_entry[2] = pid_str
                existing_entry[4].append(np.array(bbox, dtype=int))
                data[existing_entry_index] = tuple(existing_entry)
            else:
                # 创建新条目
                pid_str = self.format_pid_list([pid], with_prefix)
                data.append((group_img_path, gid, pid_str, camid, [np.array(bbox, dtype=int)]))

        return data

    def parse_pid_list(self, pid_str, with_prefix):
        if isinstance(pid_str, str):
            if with_prefix:
                pid_str = pid_str.replace('SYSUGroup_', '')  # 移除前缀
            return eval(pid_str)
        return pid_str

    def format_pid_list(self, pid_list, with_prefix):
        if with_prefix:
            return f"{self.dataset_name}_{str(pid_list).replace(' ', '')}"
        return pid_list
