import torch


def collate_fn(batch):
    output = {}
    tensor_keys = ["lensless", "psf", "lensed", "lensed_roi"]
    for key in tensor_keys:
        if key in batch[0] and batch[0][key] is not None:
            output[key] = torch.stack([item[key] for item in batch], dim=0)
    output["ids"] = [item["id"] for item in batch]
    return output
