from .configs import DataConfig, ScaleConfig

import gunpowder as gp
import daisy
from funlib.geometry import Coordinate, Roi

import numpy as np

import neuroglancer
import neuroglancer.cli

from pathlib import Path

from .gp.resample import Resample
from .gp.pad import Pad
from .gp.relabel import Relabel


def build_pipeline(
    data_config: DataConfig, scale_config: ScaleConfig, gt_voxel_size: Coordinate
) -> gp.Pipeline:

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    training_crops = [
        crop_num
        for crop_num in data_config.training_crops
        if (
            data_config.dataset_container
            / data_config.gt_dataset.format(crop_num=crop_num)
        ).exists()
        or (
            data_config.dataset_container
            / data_config.gt_dataset.format(crop_num=crop_num)
        ).exists()
    ]

    # open training crops as daisy arrays
    # read from fallback location if main container does not contain
    # expected dataset
    training_datasets: list[daisy.Array] = [
        daisy.open_ds(
            f"{data_config.dataset_container}",
            data_config.gt_dataset.format(crop_num=crop_num),
        )
        if Path(
            f"{data_config.dataset_container}",
            data_config.gt_dataset.format(crop_num=crop_num),
        ).exists()
        else daisy.open_ds(
            f"{data_config.fallback_dataset_container}",
            data_config.gt_dataset.format(crop_num=crop_num),
        )
        for crop_num in training_crops
    ]

    # filter training crops by size. Since we need multiple scale
    # levels, we only want volumes large enough to train on
    training_crops = [
        crop
        for crop, dataset in zip(training_crops, training_datasets)
        if min(dataset.roi.shape) >= data_config.min_volume_size and dataset.voxel_size == gt_voxel_size
    ]
    training_datasets = [
        dataset
        for dataset in training_datasets
        if min(dataset.roi.shape) >= data_config.min_volume_size and dataset.voxel_size == gt_voxel_size
    ]

    # get raw container
    raw_datasets: list[daisy.Array] = [
        daisy.open_ds(
            f"{data_config.dataset_container}",
            f"{data_config.raw_dataset}/s{scale_level}",
        )
        if Path(
            f"{data_config.dataset_container}",
            f"{data_config.raw_dataset}/s{scale_level}",
        ).exists()
        else daisy.open_ds(
            f"{data_config.fallback_dataset_container}",
            f"{data_config.raw_dataset}/s{scale_level}",
        )
        for scale_level in raw_scale_levels
    ]
    print(raw_datasets[0].data.store.path, raw_datasets[0].data.name)

    raw_scale_keys = [gp.ArrayKey(f"RAW_S{scale}") for scale in raw_scale_levels]
    labels_key = gp.ArrayKey("LABELS")
    gt_key = gp.ArrayKey("GT")
    gt_scale_keys = [gp.ArrayKey(f"GT_S{scale}") for scale in gt_scale_levels]
    weight_scale_keys = [gp.ArrayKey(f"WEIGHT_S{scale}") for scale in gt_scale_levels]

    pipeline = (
        tuple(
            (
                gp.ZarrSource(
                    dataset.data.store.path,
                    {labels_key: dataset.data.name},
                    {
                        labels_key: gp.ArraySpec(
                            roi=dataset.roi,
                            voxel_size=dataset.voxel_size,
                            interpolatable=False,
                        )
                    },
                )
                + Relabel(
                    labels_key,
                    gt_key,
                    [
                        [3, 4, 5],
                        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37],
                        [16, 17, 18, 19],
                    ],
                ),
                gp.ZarrSource(
                    raw_datasets[0].data.store.path,
                    {
                        raw_scale_keys[scale_level]: raw_dataset.data.name
                        for scale_level, raw_dataset in enumerate(raw_datasets)
                    },
                    {
                        raw_scale_keys[scale_level]: gp.ArraySpec(
                            voxel_size=raw_dataset.voxel_size,
                            interpolatable=True,
                        )
                        for scale_level, raw_dataset in enumerate(raw_datasets)
                    },
                )
                + Pad(raw_scale_keys, None),
            )
            + gp.MergeProvider()
            + gp.RandomLocation()
            for dataset in training_datasets
        )
        + gp.RandomProvider()
    )
    for i in gt_scale_levels:
        pipeline += Resample(
            gt_key, gt_voxel_size * (scale_config.scale_factor**i), gt_scale_keys[i]
        )
        pipeline += gp.BalanceLabels(
            gt_scale_keys[i], weight_scale_keys[i], num_classes=4
        )
    for i in raw_scale_levels:
        pipeline += gp.Normalize(raw_scale_keys[i])

    pipeline += gp.SimpleAugment()
    return pipeline


def get_request(
    base_shape: Coordinate,
    scale_config: ScaleConfig,
) -> gp.BatchRequest:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    raw_shapes = [
        base_shape * (scale_config.scale_factor**scale_level)
        for scale_level in raw_scale_levels
    ]
    gt_shapes = [
        base_shape * (scale_config.scale_factor**scale_level)
        for scale_level in gt_scale_levels
    ]

    raw_scale_keys = [gp.ArrayKey(f"RAW_S{scale}") for scale in range(len(raw_shapes))]
    gt_scale_keys = [gp.ArrayKey(f"GT_S{scale}") for scale in range(len(gt_shapes))]
    weight_scale_keys = [
        gp.ArrayKey(f"WEIGHT_S{scale}") for scale in range(len(gt_shapes))
    ]

    request = gp.BatchRequest()
    for raw_key, input_shape in zip(raw_scale_keys, raw_shapes):
        request.add(
            raw_key,
            input_shape,
        )
    for gt_key, weight_key, output_shape in zip(
        gt_scale_keys, weight_scale_keys, gt_shapes
    ):
        request.add(
            gt_key,
            output_shape,
        )
        request.add(
            weight_key,
            output_shape,
        )
    return request


def split_batch(
    batch: gp.BatchRequest,
    scale_config: ScaleConfig,
) -> tuple[list[gp.Array], list[gp.Array], list[gp.Array]]:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    raw_scale_keys = [gp.ArrayKey(f"RAW_S{scale}") for scale in raw_scale_levels]
    gt_scale_keys = [gp.ArrayKey(f"GT_S{scale}") for scale in gt_scale_levels]
    weight_scale_keys = [gp.ArrayKey(f"WEIGHT_S{scale}") for scale in gt_scale_levels]

    raw_arrays = [batch[key] for key in raw_scale_keys]
    gt_arrays = [batch[key] for key in gt_scale_keys]
    weight_arrays = [batch[key] for key in weight_scale_keys]
    return raw_arrays, gt_arrays, weight_arrays
