import numpy as np
import random

import torch

from util.voxelize import voxelize
import os


def collate_fn_limit_fs(batch, include_scene_names=False):
    """
    :param batch: for episode training, batch size = 1
    :param include_scene_names: boolean flag to indicate if scene names are included in the batch
    :return: various concatenated tensors and optionally scene names
    """
    # Unpack batch according to the expected structure
    if include_scene_names:
        (
            support_feat,
            support_label,
            query_feat,
            query_label,
            sampled_classes,
            scene_names,
        ) = batch[0]
    else:
        (
            support_feat,
            support_label,
            query_feat,
            query_label,
            sampled_classes,
        ) = batch[0]

    # Compute support offset
    support_offset, count = [], 0
    for item in support_feat:
        count += item.shape[0]
        support_offset.append(count)

    # Compute query offset
    query_offset, count = [], 0
    for item in query_feat:
        count += item.shape[0]
        query_offset.append(count)

    # Prepare the return tuple
    return_tuple = (
        torch.cat(support_feat),
        torch.cat(support_label),
        torch.IntTensor(support_offset),
        torch.cat(query_feat),
        torch.cat(query_label),
        torch.IntTensor(query_offset),
        sampled_classes,
    )

    if include_scene_names:
        return_tuple += (scene_names,)

    return return_tuple


def collate_fn_limit_fs_train(batch):
    """
    :param batch: for episode training, batch size = 1
    :param logger:
    :return: sampled_classes: numpy
    """
    # [(n,c),...] [(n),..]
    (
        support_feat,
        support_base_masks,
        support_test_masks,
        query_feat,
        query_base_label,
        query_test_label,
        sampled_classes,
    ) = batch[0]
    support_offset, count = [], 0
    for item in support_feat:
        count += item.shape[0]
        support_offset.append(count)

    query_offset, count = [], 0
    for item in query_feat:
        count += item.shape[0]
        query_offset.append(count)

    return (
        torch.cat(support_feat),
        torch.cat(support_base_masks),
        torch.cat(support_test_masks),
        torch.IntTensor(support_offset),
        torch.cat(query_feat),
        torch.cat(query_base_label),
        torch.cat(query_test_label),
        torch.IntTensor(query_offset),
        sampled_classes,
    )


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0

    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return (
        torch.cat(coord),
        torch.cat(feat),
        torch.cat(label),
        torch.IntTensor(offset),
    )


def collate_fn_limit(batch, max_batch_points, logger):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0

    k = 0
    for item in coord:
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning(
            "batch_size shortened from {} to {}, points from {} to {}".format(
                len(batch), k, s, s_now
            )
        )

    return (
        torch.cat(coord[:k]),
        torch.cat(feat[:k]),
        torch.cat(label[:k]),
        torch.IntTensor(offset[:k]),
    )


def area_crop(coord, area_rate, split="train"):
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min
    coord_max -= coord_min
    x_max, y_max = coord_max[0:2]
    x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
    if split == "train" or split == "trainval":
        x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(
            0, y_max - y_size
        )
    else:
        x_s, y_s = (x_max - x_size) / 2, (y_max - y_size) / 2
    x_e, y_e = x_s + x_size, y_s + y_size
    crop_idx = np.where(
        (coord[:, 0] >= x_s)
        & (coord[:, 0] <= x_e)
        & (coord[:, 1] >= y_s)
        & (coord[:, 1] <= y_e)
    )[0]
    return crop_idx


def load_kitti_data(data_path):
    data = np.fromfile(data_path, dtype=np.float32)
    data = data.reshape((-1, 4))  # xyz+remission
    return data


def load_kitti_label(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert (sem_label + (inst_label << 16) == label).all()
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def data_prepare(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = (
            np.random.randint(label.shape[0])
            if "train" in split
            else label.shape[0] // 2
        )
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[
            :voxel_max
        ]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= (coord_min + coord_max) / 2.0
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v101(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
    sampled_class=None,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)

    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]

    if voxel_max and label.shape[0] > voxel_max:
        crop_idx = np.random.choice(
            np.arange(label.shape[0]), voxel_max, replace=False
        )
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_vis(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
    sampled_class=None,
    filename=None,
    support=False,
    query_source_file=None,
    target_save_file=None,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)

    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]

    if voxel_max and label.shape[0] > voxel_max:
        crop_idx = np.random.choice(
            np.arange(label.shape[0]), voxel_max, replace=False
        )
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    if not support:
        smapled = np.concatenate(
            [coord + coord_min, feat, label[:, None]], axis=1
        )
        filename = filename.replace("query", "input").replace(
            query_source_file, target_save_file
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(filename)
        np.save(filename, smapled)

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_scannet(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        # init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        # crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        # coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
        crop_idx = np.random.choice(
            np.arange(label.shape[0]), voxel_max, replace=False
        )
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v102(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    while voxel_max and label.shape[0] > voxel_max * 1.1:
        area_rate = voxel_max / float(label.shape[0])
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min
        coord_max -= coord_min
        x_max, y_max = coord_max[0:2]
        x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
        if split == "train":
            x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(
                0, y_max - y_size
            )
        else:
            x_s, y_s = 0, 0
        x_e, y_e = x_s + x_size, y_s + y_size
        crop_idx = np.where(
            (coord[:, 0] >= x_s)
            & (coord[:, 0] <= x_e)
            & (coord[:, 1] >= y_s)
            & (coord[:, 1] <= y_e)
        )[0]
        if crop_idx.shape[0] < voxel_max // 8:
            continue
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v103(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min
        coord_max -= coord_min
        xy_area = 7
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(
                xy_area
            )
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[
                1
            ] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[
                1
            ] * (y_area + 1) / float(xy_area)
            crop_idx = np.where(
                (coord[:, 0] >= x_s)
                & (coord[:, 0] <= x_e)
                & (coord[:, 1] >= y_s)
                & (coord[:, 1] <= y_e)
            )[0]
            if crop_idx.shape[0] > 0:
                init_idx = (
                    crop_idx[np.random.randint(crop_idx.shape[0])]
                    if "train" in split
                    else label.shape[0] // 2
                )
                crop_idx = np.argsort(
                    np.sum(np.square(coord - coord[init_idx]), 1)
                )[:voxel_max]
                coord, feat, label = (
                    coord[crop_idx],
                    feat[crop_idx],
                    label[crop_idx],
                )
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v104(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min
        coord_max -= coord_min
        xy_area = 10
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(
                xy_area
            )
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[
                1
            ] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[
                1
            ] * (y_area + 1) / float(xy_area)
            crop_idx = np.where(
                (coord[:, 0] >= x_s)
                & (coord[:, 0] <= x_e)
                & (coord[:, 1] >= y_s)
                & (coord[:, 1] <= y_e)
            )[0]
            if crop_idx.shape[0] > 0:
                init_idx = (
                    crop_idx[np.random.randint(crop_idx.shape[0])]
                    if "train" in split
                    else label.shape[0] // 2
                )
                crop_idx = np.argsort(
                    np.sum(np.square(coord - coord[init_idx]), 1)
                )[:voxel_max]
                coord, feat, label = (
                    coord[crop_idx],
                    feat[crop_idx],
                    label[crop_idx],
                )
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v105(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = (
            np.random.randint(label.shape[0])
            if "train" in split
            else label.shape[0] // 2
        )
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[
            :voxel_max
        ]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord[:, 0:2] -= coord_min[0:2]
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.0
    label = torch.LongTensor(label)
    return coord, feat, label
