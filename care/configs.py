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


class DatasetConfig(BaseModel):
    path: Path


class DataConfig(BaseModel):
    datasets: list[DatasetConfig]
    neighborhood: list[PydanticCoordinate]


class TrainConfig(BaseModel):
    input_shape_voxels: PydanticCoordinate
    eval_input_shape_voxels: PydanticCoordinate
    num_iterations: int
    checkpoint_interval: int
    snapshot_interval: int
    warmup: int
    batch_size: int
    num_workers: int
    base_dir: Path
    learning_rate: float
    data_config: DataConfig
    architecture_config: BackboneConfig
    start: Optional[Path] = None

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def snapshot_container(self) -> Path:
        return self.base_dir / "snapshots"

    @property
    def validation_container(self) -> Path:
        return self.base_dir / "validation"

    @property
    def loss_file(self) -> Path:
        return self.base_dir / "loss.csv"

    @property
    def val_file(self) -> Path:
        return self.base_dir / "val.csv"
