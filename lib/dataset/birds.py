# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
import pickle

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)


class BirdsDataset(JointsDataset):
    """
    Class for birds dataset.
    Body parts legend: ['back',  'beak',  'belly',  'breast',  'crown', 
          'forehead',  'left eye',  'left leg',  'left wing', 'nape',  
          'right eye',  'right leg',  'right wing', 'tail',  'throat']
    """
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        
        self.num_joints = 15
        self.dataset_joints = np.array([['back',  'beak',  'belly',  'breast',  'crown',
                               'forehead',  'left eye',  'left leg',  'left wing',
                               'nape',  'right eye',  'right leg',  'right wing',
                               'tail',  'throat']])
        self.flip_pairs = [[6, 10], [7, 11], [8, 12]]
        
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)
            
        file_name = os.path.join(self.root, 'annot', self.image_set+'.json')
        
        with open(file_name) as file:
            gt_file = json.load(file)
            
        self.jnt_visible = []
        self.pos_gt_src = []
        self.bbox_size = []
        
        for elem in gt_file:
            self.jnt_visible.append(elem['joints_vis'])
            self.pos_gt_src.append(elem['joints'])
            self.bbox_size.append(elem['bbox_max_side'])
            
        self.jnt_visible = np.array(self.jnt_visible)
        self.jnt_visible = self.jnt_visible.swapaxes(0,1)
        
        self.pos_gt_src = np.array(self.pos_gt_src)
        self.pos_gt_src = self.pos_gt_src.transpose((1,2,0))
        self.bbox_size = np.array(self.bbox_size)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.root, 'annot', self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        id_container = set()
        for a in anno:
            image_name = a['image']
            
            #Read identity labels = folder names
            id_name = os.path.split(os.path.split(image_name)[0])[-1]
            id_container.add(id_name)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)

            if 'test' not in self.image_set:
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': (0,0),
                    'scale': 1.,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0
                })

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # get 2D preds
        preds = preds[:, :, 0:2]

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.pkl')
            with open(pred_file, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(preds, output, pickle.HIGHEST_PROTOCOL)

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        threshold = 0.1
        logger.info('Computing accuracy of landmark detection for threshold {} pixels'.format(threshold))

        pos_pred_src = np.transpose(preds, [1, 2, 0])   
    
        back = np.where(self.dataset_joints == 'back')[1][0]
        beak = np.where(self.dataset_joints == 'beak')[1][0]
        belly = np.where(self.dataset_joints == 'belly')[1][0]
        breast = np.where(self.dataset_joints == 'breast')[1][0]
        crown = np.where(self.dataset_joints == 'crown')[1][0]
        forehead = np.where(self.dataset_joints == 'forehead')[1][0]
        
        leye = np.where(self.dataset_joints == 'left eye')[1][0]
        lleg = np.where(self.dataset_joints == 'left leg')[1][0]
        lwing = np.where(self.dataset_joints == 'left wing')[1][0]
        nape = np.where(self.dataset_joints == 'nape')[1][0]

        reye = np.where(self.dataset_joints == 'right eye')[1][0]
        rleg = np.where(self.dataset_joints == 'right leg')[1][0]
        rwing = np.where(self.dataset_joints == 'right wing')[1][0]
        tail = np.where(self.dataset_joints == 'tail')[1][0]
        throat = np.where(self.dataset_joints == 'throat')[1][0]

        uv_error = pos_pred_src - self.pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        scale = np.multiply(self.bbox_size, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, self.jnt_visible)
        jnt_count = np.sum(self.jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          self.jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), self.num_joints))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              self.jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Back', PCKh[back]),
            ('Beak', PCKh[beak]),
            ('Belly', PCKh[belly]),
            ('Breast', PCKh[breast]),
            ('Crown', PCKh[crown]),
            ('Forehead', PCKh[forehead]),
            ('Left eye', PCKh[leye]),
            ('Left leg', PCKh[lleg]),
            ('Left wing', PCKh[lwing]),
            ('Nape', PCKh[nape]),
            ('Right eye', PCKh[reye]),
            ('Right leg', PCKh[rleg]),
            ('Right wing', PCKh[rwing]),
            ('Tail', PCKh[tail]),
            ('Throat', PCKh[throat]),
            
            ('Mean', np.sum(PCKh * jnt_ratio)),
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

