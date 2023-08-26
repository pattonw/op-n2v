from funlib.geometry import Coordinate

from pydantic import BaseModel
import yaml

from typing import Optional, Union
from pathlib import Path


class PydanticCoordinate(Coordinate):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, validation_info):
        return Coordinate(*v)


class BackboneConfig(BaseModel):
    raw_input_channels: int
    n_output_channels: int
    num_init_features: int
    num_embeddings: int
    growth_rate: int
    block_config: list[int]
    padding: str
    embeddings: bool


class DataConfig(BaseModel):
    datasets: list[Path]
    neighborhood: list[PydanticCoordinate]


class StorageConfig(BaseModel):
    dataset: str = "volumes/raw/s{level}"
    crop: str = "volumes/groundtruth/crop{crop_num}/labels/{organelle}"
    container: str = "/groups/cellmap/cellmap/data/{dataset}/{dataset}.n5"
    fallback: str = "/nrs/cellmap/pattonw/data/tmp_data/{dataset}/{dataset}.n5"


class TrainConfig(BaseModel):
    input_shape_voxels: PydanticCoordinate
    eval_input_shape_voxels: PydanticCoordinate
    num_iterations: int
    checkpoint_interval: int
    snapshot_interval: int
    warmup: int
    batch_size: int
    num_workers: int
    checkpoint_dir: Path
    snapshot_container: Path
    validation_container: Path
    loss_file: Path
    val_file: Path
    learning_rate: float
    data_config_file: Path
    architecture_config_file: Path
    start: Optional[Path] = None

    @property
    def data_config(self) -> DataConfig:
        return DataConfig(**yaml.safe_load(self.data_config_file.open("r").read()))

    @property
    def architecture_config(self) -> BackboneConfig:
        return BackboneConfig(
            **yaml.safe_load(self.architecture_config_file.open("r").read())
        )
