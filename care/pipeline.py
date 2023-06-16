from .configs import DataConfig
from .gp.scale_provider import ScaleProvider
from .gp.context import Context
from .gp.histogram_equalization import HistogramEqualization

import gunpowder as gp
import daisy

from funlib.geometry import Coordinate

import yaml

from collections import defaultdict
import string

RAW_CONTAINER = "/groups/cellmap/cellmap/data/{dataset}/{dataset}.n5"
RAW_DATASET = "volumes/raw/s{level}"


def build_pipeline(
    data_config: DataConfig,
) -> tuple[gp.Pipeline, list[Coordinate]]:

    raw_input_key = gp.ArrayKey("RAW_INPUT")
    raw_context_key = gp.ArrayKey("RAW_CONTEXT")

    resolution_pipelines = defaultdict(lambda: list())
    for dataset_yaml in data_config.datasets:

        dataset_info = yaml.safe_load(dataset_yaml.open().read())

        raw_info = dataset_info.get("raw", dict())
        dataset_container: str = raw_info.get("container", RAW_CONTAINER).format(
            dataset=dataset_yaml.name[:-5]
        )
        raw_dataset_name: str = raw_info.get("dataset", RAW_DATASET)
        has_scale_levels = "scale_level" in [
            tup[1]
            for tup in string.Formatter().parse(raw_dataset_name)
            if tup[1] is not None
        ]

        for scale_level in range(5):

            if not has_scale_levels and scale_level > 0:
                continue

            # get raw container
            raw_dataset_level = raw_dataset_name.format(level=scale_level)
            raw_dataset: daisy.Array = daisy.open_ds(
                f"{dataset_container}",
                raw_dataset_level,
            )

            pipeline = (
                gp.ZarrSource(
                    str(dataset_container),
                    {raw_input_key: raw_dataset_level},
                    {raw_input_key: gp.ArraySpec(voxel_size=raw_dataset.voxel_size)},
                )
                + gp.Normalize(raw_input_key)
                + gp.IntensityAugment(raw_input_key, 0.9, 1.1, -0.1, 0.1)
                + Context(raw_input_key, raw_context_key, data_config.neighborhood)
                + HistogramEqualization(raw_context_key, raw_context_key)
                + gp.RandomLocation()
            )
            resolution_pipelines[raw_dataset.voxel_size].append(pipeline)

    scale_pipelines = tuple(
        tuple(dataset_pipelines) + gp.RandomProvider()
        for _, dataset_pipelines in resolution_pipelines.items()
    )
    pipeline = tuple(scale_pipelines) + ScaleProvider(scale_key=raw_input_key)

    return pipeline, list(resolution_pipelines.keys())


def get_request(shape: Coordinate, out_shape) -> gp.BatchRequest:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_input_key = gp.ArrayKey("RAW_INPUT")
    raw_context_key = gp.ArrayKey("RAW_CONTEXT")

    request = gp.BatchRequest()
    request.add(raw_input_key, shape)
    request.add(raw_context_key, out_shape)
    return request


def split_batch(batch: gp.BatchRequest) -> tuple[gp.Array, gp.Array]:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_input_key = gp.ArrayKey("RAW_INPUT")
    raw_context_key = gp.ArrayKey("RAW_CONTEXT")

    raw_input_array = batch[raw_input_key]
    raw_context_array = batch[raw_context_key]
    return (raw_input_array, raw_context_array)
