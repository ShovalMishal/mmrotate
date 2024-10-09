# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List
import torch
from mmengine.dataset import BaseDataset
from mmengine.structures import InstanceData
from mmrotate.registry import DATASETS
import re


@DATASETS.register_module()
class DOTADataset(BaseDataset):
    """DOTA-v1.0 dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 ood_labels=None,
                 ignore_ood_labels=False,
                 extracted_patches_folder=None,
                 **kwargs) -> None:
        self.extracted_patches_folder = extracted_patches_folder
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        self.ood_labels = ood_labels if ood_labels is not None else []
        self.ignore_ood_labels = ignore_ood_labels
        super().__init__(**kwargs)

    def get_index_img_id(self, img_id):
        for idx, sample in enumerate(self.load_data_list()):
            if sample['img_id'] == img_id:
                return idx
        return None


    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []

        def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
            return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            txt_files = sorted(txt_files, key=natural_sort_key)
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)
                if self.extracted_patches_folder:
                    patches_path = osp.join(self.extracted_patches_folder, img_id + '.pt')
                    if osp.exists(patches_path):
                        cache_dict = torch.load(patches_path, map_location=torch.device('cpu'))
                        predicted_patches = cache_dict['predicted_patches']
                        preds = InstanceData(predicted_patches=predicted_patches)
                        data_info["predicted_patches"] = preds
                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = bbox_info[8]
                        if cls_name in self.ood_labels and self.ignore_ood_labels:
                            continue
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)
            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]


@DATASETS.register_module()
class DOTAv15Dataset(DOTADataset):
    """DOTA-v1.5 dataset for detection.

    Note: ``ann_file`` in DOTAv15Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv15Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60)]
    }


@DATASETS.register_module()
class DOTAv2Dataset(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport',
         'helipad'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60), (119, 11, 32),
                    (0, 0, 142)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD1(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('small-vehicle',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD11(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('small-vehicle', 'large-vehicle'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD2(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('ship',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD3(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('harbor',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD4(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD5(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('tennis-court',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD6(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('swimming-pool',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD7(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('storage-tank',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetOOD8(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('bridge',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetMSOOD1(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """
    METAINFO = {
        'classes':
            ('small-vehicle',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }

@DATASETS.register_module()
class DOTAv2DatasetMSOOD2(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """
    METAINFO = {
        'classes':
        ('small-vehicle', 'large-vehicle', 'swimming-pool'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0)]
    }

@DATASETS.register_module()
class DOTAv2DatasetMSOOD3(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """
    METAINFO = {
        'classes':
        ('plane', 'tennis-court', 'ship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0)]
    }

@DATASETS.register_module()
class DOTAv2DatasetMSOOD4(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """
    METAINFO = {
        'classes':
        ('baseball-diamond', 'bridge',
         'small-vehicle', 'large-vehicle',
         'storage-tank',
         'swimming-pool'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0)]
    }

@DATASETS.register_module()
class DOTAv2DatasetMSOOD5(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """
    METAINFO = {
        'classes':
        ('bridge',
         'harbor',
         'plane'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0)]
    }

@DATASETS.register_module()
class DOTAv2DatasetMSOOD6(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """
    METAINFO = {
        'classes':
        ('baseball-diamond',
         'small-vehicle', 'large-vehicle',
         'storage-tank',
         'swimming-pool'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0)]
    }


