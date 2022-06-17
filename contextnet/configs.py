from pydantic import BaseModel
from pathlib import Path

from funlib.geometry import Coordinate


class PydanticCoordinate(Coordinate):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Coordinate(*v)


class BackboneConfig(BaseModel):
    n_input_channels: int
    n_output_channels: int
    num_init_features: int
    growth_rate: int
    block_config: list[int]
    padding: str


class TrainConfig(BaseModel):
    input_shape_voxels: PydanticCoordinate
    num_iterations: int
    checkpoint_interval: int
    snapshot_interval: int
    batch_size: int
    num_workers: int
    checkpoint_dir: Path
    snapshot_container: Path
    loss_file: Path
    learning_rate: float


class ScaleConfig(BaseModel):

    scale_factor: PydanticCoordinate
    num_raw_scale_levels: int
    num_gt_scale_levels: int


class DataConfig(BaseModel):
    dataset_container: Path
    fallback_dataset_container: Path
    raw_dataset: str
    gt_dataset: str
    training_crops: list[int]
    validation_crops: list[int]
    min_volume_size: int
    gt_voxel_size: PydanticCoordinate  # assumed to be half the raw voxel size
