import glob
import pdb
import pickle
import os.path as osp
import re
import warnings
import json
import random
import numpy

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class roadgroup(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'roadgroup'
    dataset_name = "roadgroup"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = '/home/wzg/dataset/road_group'
        self.dataset_dir = self.root
        # allow alternative directory structure
        self.data_dir = osp.join(self.dataset_dir, 'Road_Group/group')
        self.label_dir = osp.join(self.dataset_dir, 'Road_Group_Annotations')

        train, query, gallery = self.process_dir(self.data_dir, self.label_dir)
        # query = self.process_dir(self.data_dir, self.label_dir, 'query')
        # gallery = self.process_dir(self.data_dir, self.label_dir, 'gallery')

        super(roadgroup, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, label_path):

        g_labels = osp.join(label_path, 'group_id.json')
        with open(g_labels, 'r', encoding='utf-8') as file:
            g_labels = json.load(file)

        p_labels = osp.join(label_path, 'person_bounding_box.json')
        with open(p_labels, 'r', encoding='utf-8') as file:
            p_labels = json.load(file)

        t = ["fig" + str(x) + "_" for x in range(324)]


        random_idx = random.sample(range(0,162), 81)  #存储随机索引
        test_idx = list(set(range(0, 162)) - set(random_idx))


        img_path = []
        gid = []
        pid = []
        bbox = []
        for idx in random_idx:
            pdx = idx * 2    #找到组图中的第一张图
            li = g_labels[idx]
            image_name = li['image names']
            id = li['id']
            for i in image_name:
                fig = t[pdx]   # 区分图里的单个人
                p = p_labels[pdx]
                pid_list = []
                bbox_list = []
                img_path.append(i)
                temp = p['pedestrian']
                for j in temp:
                    pid_list.append(fig + str(j['person id']))
                    bbox_list.append(numpy.array(j['bbox'], dtype=int))
                pid.append(pid_list)
                bbox.append(bbox_list)
                gid.append(id)
                pdx += 1

        train = []
        train.append((img_path, gid, pid, bbox))


        query = []
        gallery = []
        img_path = []
        gid = []
        pid = []
        bbox = []
        for idx in test_idx:
            pdx = idx * 2  # 找到组图中的第一张图
            li = g_labels[idx]
            image_name = li['image names']
            id = li['id']
            for i in image_name:
                fig = t[pdx]  # 区分图里的单个人
                p = p_labels[pdx]
                pid_list = []
                bbox_list = []
                img_path.append(i)
                temp = p['pedestrian']
                for j in temp:
                    pid_list.append(fig + str(j['person id']))
                    bbox_list.append(numpy.array(j['bbox'], dtype=int))
                pid.append(pid_list)
                bbox.append(bbox_list)
                gid.append(id)
                pdx += 1

        img_path_query = []
        gid_c_query = []
        pid_query = []
        bbox_query = []

        img_path_gallery = []
        gid_c_gallery = []
        pid_gallery = []
        bbox_gallery = []
        for i in range(len(gid)):
            if(i%2!=0):
                img_path_gallery.append(img_path[i])
                gid_c_gallery.append((gid[i]))
                pid_gallery.append((pid[i]))
                bbox_gallery.append(bbox[i])
            else:
                img_path_query.append(img_path[i])
                gid_c_query.append((gid[i]))
                pid_query.append((pid[i]))
                bbox_query.append(bbox[i])

        query.append((img_path_query, gid_c_query, pid_query,  bbox_query))
        gallery.append((img_path_gallery, gid_c_gallery, pid_gallery,  bbox_gallery))

        train = train[0]
        query = query[0]
        gallery = gallery[0]

        img_paths = [osp.join(dir_path, x) for x in train[0]]
        train_data = []
        for img_path in img_paths:
            index = img_paths.index(img_path)
            # in CUHK_SYSU_Group dataset, person and group id start from 0.
            # we force it start from 1
            gid = train[1][index] + 1
            pid = train[2][index]
            bbox = train[3][index]
            camid = 0
            assert gid >= 0
            # assert 1 <= camid <= 6
            gid = self.dataset_name + "_" + str(gid)
            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_" + str(camid)
            # spceial operator for this dataset
            pid = pid.replace(' ', '')
            train_data.append((img_path, gid, pid, camid, bbox))

        img_paths = [osp.join(dir_path, x) for x in query[0]]
        query_data = []
        for img_path in img_paths:
            index = img_paths.index(img_path)
            # in CUHK_SYSU_Group dataset, person and group id start from 0.
            # we force it start from 1
            gid = query[1][index] + 1
            pid = query[2][index]
            bbox = query[3][index]
            camid = 1
            assert gid >= 0
            query_data.append((img_path, gid, pid, camid, bbox))

        img_paths = [osp.join(dir_path, x) for x in gallery[0]]
        gallery_data = []
        for img_path in img_paths:
            index = img_paths.index(img_path)
            # in CUHK_SYSU_Group dataset, person and group id start from 0.
            # we force it start from 1
            gid = gallery[1][index] + 1
            pid = gallery[2][index]
            bbox = gallery[3][index]
            camid = 2
            assert gid >= 0
            gallery_data.append((img_path, gid, pid, camid, bbox))


        return train_data, query_data, gallery_data