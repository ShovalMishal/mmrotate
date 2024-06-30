import os
import json
from mmengine.config import Config
from mmengine.runner import Runner

def save_images_idx_with_certain_labels(data_loader, labels=[15,16,17],
                                        out_path="/home/shoval/Documents/Repositories/mmrotate/results/",
                                        examples_number=20):
    contains_labels = []
    not_contains = []
    file_path = os.path.join(out_path, "file_names_to_inference.json")
    for i, data in enumerate(data_loader):
        is_contains = any([x in data['data_samples'][0].gt_instances.labels for x in labels])
        if is_contains:
            contains_labels.append(data['data_samples'][0].img_path)
            print("new label is found!\n")
        elif len(not_contains)<examples_number:
            not_contains.append(data['data_samples'][0].img_path)
        if len(not_contains)==len(contains_labels)==20:
            break
    json_data = {'contain_labels' : contains_labels,
                 'not_contains': not_contains}
    with open(file_path, 'w') as file:
        json.dump(json_data, file)

if __name__ == '__main__':
    cfg = Config.fromfile("/home/shoval/Documents/Repositories/mmrotate/oriented-rcnn-le90_r50_fpn_1x_dota.py")
    dataloader_cfg = cfg.get('test_dataloader')
    dataloader_cfg["dataset"]["_scope_"] = "mmrotate"
    data_loader = Runner.build_dataloader(dataloader_cfg, seed=123456)
    save_images_idx_with_certain_labels(data_loader, labels=[15,16,17],
                                        out_path="/home/shoval/Documents/Repositories/mmrotate/results/",
                                        examples_number=20)
