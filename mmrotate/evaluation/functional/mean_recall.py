from multiprocessing import get_context
from mmrotate.datasets import DOTAv2Dataset
import numpy as np
import torch
from mmcv.ops import box_iou_quadri, box_iou_rotated
from mmdet.evaluation.functional import average_precision
from mmengine.logging import print_log
from terminaltables import AsciiTable
from tqdm import tqdm
from .mean_ap import tpfp_default


def print_mrecall_summary_bbox_regressor(mean_recall,
                                         results,
                                         dataset=None,
                                         scale_ranges=None,
                                         logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str, optional): Dataset name or dataset classes.
        scale_ranges (list[tuple], optional): Range of scales to be evaluated.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details.
            Defaults to None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['precision'], np.ndarray):
        num_scales = len(results[0]['precision'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        recalls[:, i] = cls_result['recall']
        precisions[:, i] = cls_result['precision']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    # if not isinstance(mean_ap, list):
    #     mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'precision']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{precisions[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mean_recall', '', '', '', f'{mean_recall[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)


def print_recall_summary(mean_recall,
                      results,
                      ood_classes,
                      scale_ranges=None,
                      logger=None):
    """Print recall and results of each class."""

    if logger == 'silent':
        return

    if isinstance(results[0]['recall_for_ood_class'], np.ndarray):
        num_scales = len(results[0]['recall_for_ood_class'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_ood_classes = len(results)

    recalls = np.zeros((num_scales, num_ood_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_ood_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall_for_ood_class'].size > 0:
            recalls[:, i] = np.array(cls_result['recall_for_ood_class'], ndmin=2)[:, -1]
        num_gts[:, i] = cls_result['num_gts']

    if ood_classes is None:
        label_names = [str(i) for i in range(num_ood_classes)]
    else:
        label_names = ood_classes

    if not isinstance(mean_recall, list):
        mean_recall = [mean_recall]

    header = ['class', 'gts', 'recall']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(len(ood_classes)):
            row_data = [
                label_names[j], num_gts[i, j],
                f'{recalls[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mean recall', '', '', '', f'{mean_recall[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

def eval_ood_rbbox_recalls(det_results,
                           annotations,
                           ood_classes,
                           scale_ranges=None,
                           iou_thr=0.5,
                           use_07_metric=True,
                           box_type='rbox',
                           dataset=None,
                           logger=None,
                           nproc=4):
    """Evaluate recall of a rotated dataset, for ood classes."""

    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_ood_classes = len(ood_classes)  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    ood_labels_numbers = [DOTAv2Dataset.METAINFO['classes'].index(ood_class) for ood_class in ood_classes]
    for i, class_name in zip(ood_labels_numbers, ood_classes):
        # get gt of this class and all det bboxes
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results_for_ood(
            det_results, annotations, i, box_type)

        # compute tp and fp for each image with multiple processes
        # should change sort in tpfp_default
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [box_type for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                if box_type == 'rbox':
                    gt_areas = bbox[:, 2] * bbox[:, 3]
                elif box_type == 'qbox':
                    pts = bbox.reshape(*bbox.shape[:-1], 4, 2)
                    roll_pts = torch.roll(pts, 1, dims=-2)
                    xyxy = torch.sum(
                        pts[..., 0] * roll_pts[..., 1] -
                        roll_pts[..., 0] * pts[..., 1],
                        dim=-1)
                    gt_areas = 0.5 * torch.abs(xyxy)
                else:
                    raise NotImplementedError
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        tp = np.hstack(tp)
        # calculate recall and precision with tp and fp
        tp = np.sum(tp, axis=1)
        eps = np.finfo(np.float32).eps
        recall = tp / np.maximum(num_gts[:, np.newaxis], eps)
        # calculate and save recall
        if scale_ranges is None:
            recall = recall[0, :]
            num_gts = num_gts.item()
        eval_results.append({
            'recall_for_ood_class': recall,
            'num_gts': num_gts,
            'class_name': class_name
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_recall = np.vstack([cls_result['recall_for_ood_class'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        recalls = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                recalls.append(all_recall[all_num_gts[:, i] > 0, i].mean())
            else:
                recalls.append(0.0)
    else:
        recalls = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                recalls.append(cls_result['recall_for_ood_class'])
    mean_recall = np.array(recalls).mean().item() if recalls else 0.0
    print_recall_summary(
        mean_recall, eval_results, ood_classes, area_ranges, logger=logger)

    return mean_recall, eval_results

def eval_rbbox_mrecall_for_regressor(det_results,
                                     annotations,
                                     scale_ranges=None,
                                     iou_thr=0.5,
                                     use_07_metric=True,
                                     box_type='rbox',
                                     dataset=None,
                                     logger=None,
                                     nproc=4):
    """Evaluate mean recall of a rotated dataset."""

    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(dataset)  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    # pool = get_context('fork').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_results_for_regressor(
            det_results, annotations, i, box_type)

        # compute tp and fp for each image with multiple processes
        tpfp = []
        for det, gt, gt_ignore, iou_thr, box_type, area_range in tqdm(zip(cls_dets, cls_gts, cls_gts_ignore,
                                                                     [iou_thr for _ in range(num_imgs)],
                                                                     [box_type for _ in range(num_imgs)],
                                                                     [area_ranges for _ in range(num_imgs)])):
            tpfp.append(tpfp_default(det, gt, gt_ignore, iou_thr, box_type, area_range))

        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None or  bbox.shape[0] == 0:
                num_gts[0] += bbox.shape[0]
            else:
                if box_type == 'rbox':
                    gt_areas = bbox[:, 2] * bbox[:, 3]
                elif box_type == 'qbox':
                    pts = bbox.reshape(*bbox.shape[:-1], 4, 2)
                    roll_pts = torch.roll(pts, 1, dims=-2)
                    xyxy = torch.sum(
                        pts[..., 0] * roll_pts[..., 1] -
                        roll_pts[..., 0] * pts[..., 1],
                        dim=-1)
                    gt_areas = 0.5 * torch.abs(xyxy)
                else:
                    raise NotImplementedError
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        tp = np.hstack(tp)
        fp = np.hstack(fp)
        # calculate recall and precision with tp and fp
        tp = np.sum(tp, axis=1)
        fp = np.sum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recall = tp / np.maximum(num_gts, eps)
        precision = tp / np.maximum((tp + fp), eps)

        recall = recall.item()
        precision = precision.item()
        num_gts = num_gts.item()
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recall,
            'precision': precision
        })

    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_recall = np.vstack([cls_result['recall'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_recall = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_recall.append(all_recall[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_recall.append(0.0)
    else:
        recalls = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                recalls.append(cls_result['recall'])
        mean_recall = np.array(aps).mean().item() if recalls else 0.0

    print_mrecall_summary_bbox_regressor(mean_recall, eval_results, dataset, area_ranges, logger=logger)

    return mean_recall[0], eval_results

def get_results_for_regressor(det_results, annotations, class_id, box_type):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
        box_type (str): Box type. If the QuadriBoxes is used, you need to
            specify 'qbox'. Defaults to 'rbox'.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = det_results

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        if len(ann['bboxes']) != 0:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            if box_type == 'rbox':
                cls_gts.append(torch.zeros((0, 5), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))
            elif box_type == 'qbox':
                cls_gts.append(torch.zeros((0, 8), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 8), dtype=torch.float64))
            else:
                raise NotImplementedError

    return cls_dets, cls_gts, cls_gts_ignore