# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

import json
import os
from Anomaly_Detection_in_aerial_images.DOTA_devkit.dota_inference import DOTAInference
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('imgs', help='Image addresses file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-folder', default=None, help='Path to output folder')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    with open(args.imgs) as f:
        adresses_file = json.load(f)

    new_labels_addresses = adresses_file["contain_labels"]
    # register all modules in mmrotate into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    for address in new_labels_addresses:
        # show ground true
        inference = DOTAInference(address, os.path.join(model.cfg["val_dataloader"]["dataset"].data_root,
                                                        model.cfg["val_dataloader"]["dataset"].ann_file,
                                                        os.path.splitext(os.path.basename(address))[0] + ".txt"))
        anns = inference.loadAnns()
        inference.showAnns(anns, os.path.join(args.out_folder,
                                              os.path.splitext(os.path.basename(address))[0] + "_new_gt.png"))
        t = time.time()
        # test a single image
        result = inference_detector(model, address)
        elapsed = time.time() - t
        print("time for a single inference is " + str(elapsed))

        # show the results
        img = mmcv.imread(address)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(args.out_folder, os.path.splitext(os.path.basename(address))[0]+"_new_result.png"),
            pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
