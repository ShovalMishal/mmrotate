# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmengine.config import Config
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.runner import Runner
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules
from mmrotate.datasets import DOTAv2Dataset
import json
import os
import torch
import time

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('imgs', help='Image addresses file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-folder', default="/home/shoval/Documents/Repositories/mmrotate/results/ood_examples/", help='Path to output folder')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    # parser.add_argument(
    #     '--ood_labels_names', default=['container-crane', 'airport', 'helipad'], help='Labels to check for')
    parser.add_argument(
        '--score-thr', type=float, default=0.05, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()
    id_mapping={0:4, 1:5}

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = DOTAv2Dataset.METAINFO
    config = Config.fromfile(args.config)
    ood_labels_names = config.ood_labels
    ood_labels = [visualizer.dataset_meta['classes'].index(label) for label in ood_labels_names]

    test_dataloader = Runner.build_dataloader(config.test_dataloader, seed=123456)
    for batch in test_dataloader:
        if any([x in batch['data_samples'][0].gt_instances.labels for x in ood_labels]) and \
            batch['data_samples'][0].img_id == "P2745__1024__824___824":
            t = time.time()
            model.eval()
            with torch.no_grad():
                result = model.test_step(batch)[0]
                for original_value, new_value in id_mapping.items():
                    result.pred_instances.labels[result.pred_instances.labels == original_value] = new_value

            # run inference
            elapsed = time.time() - t
            print("time for a single inference is " + str(elapsed))

            # show the results
            img = batch['inputs'][0].cpu().numpy().transpose((1, 2, 0))
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            file_name = "_".join([DOTAv2Dataset.METAINFO['classes'][x] for x in ood_labels if x in batch['data_samples'][0].gt_instances.labels])
            visualizer.add_datasample(
                'result',
                img,
                data_sample=result,
                draw_gt=True,
                show=False,
                wait_time=0,
                out_file=os.path.join(args.out_folder, batch['data_samples'][0].img_id+f"_{file_name}.png"),
                pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
