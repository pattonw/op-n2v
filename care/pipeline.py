from .configs import DataConfig
from .gp.scale_provider import ScaleProvider
from .gp.context import Context
from .gp.histogram_equalization import HistogramEqualization
from .gp.tiff_stack_source import TiffStackSource

import gunpowder as gp
import daisy

from funlib.persistence import open_ds, Array
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

    dataset_pipelines = []
    for dataset_path in data_config.datasets:

        pipeline = (
            TiffStackSource(raw_input_key, dataset_path)
            # + gp.Normalize(raw_input_key)
            + gp.IntensityAugment(raw_input_key, 0.9, 1.1, -0.1, 0.1, clip=False)
            + Context(raw_input_key, raw_context_key, data_config.neighborhood)
            + HistogramEqualization(raw_context_key, raw_context_key)
            + gp.RandomLocation()
        )
        dataset_pipelines.append(pipeline)

    pipeline = tuple(dataset_pipelines) + gp.RandomProvider()

    return pipeline


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
