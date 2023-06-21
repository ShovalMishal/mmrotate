import os
import json


def save_images_idx_with_certain_labels(data_loader, labels=[15,16,17], out_path="../out", examples_number=10):
    contains_labels = []
    not_contains = []
    file_path = os.path.join(out_path, "file_names_to_inference.json")
    for i, data in enumerate(data_loader):
        is_contains = any([x in data['gt_labels'][0].data[0][0] for x in labels])
        if is_contains:
            contains_labels.append(data["img_metas"][0].data[0][0]["filename"])
            print("new label is found!\n")
        elif len(not_contains)<examples_number:
            not_contains.append(data["img_metas"][0].data[0][0]["filename"])
        if len(not_contains)==len(contains_labels)==10:
            break
    json_data = {'contain_labels' : contains_labels,
                 'not_contains': not_contains}
    with open(file_path, 'w') as file:
        json.dump(json_data, file)
