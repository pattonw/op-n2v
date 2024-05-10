from funlib.geometry import Coordinate

from pydantic import BaseModel, Field
import yaml

from typing import Optional, Union, Annotated, Literal
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from .backbones.dense import DenseNet
from funlib.learn.torch.models import UNet


class PydanticCoordinate(Coordinate):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, validation_info):
        return Coordinate(*v)


class BackboneConfig(BaseModel, ABC):
    @abstractmethod
    def model(self) -> torch.nn.Module:
        pass


class DenseBackboneConfig(BackboneConfig):
    model_type: Literal["dense"] = "dense"
    raw_input_channels: int
    n_output_channels: int
    num_init_features: int
    num_embeddings: int
    growth_rate: int
    block_config: list[int]
    padding: str
    embeddings: bool

    def model(self) -> torch.nn.Module:
        return DenseNet(
            n_input_channels=self.raw_input_channels,
            n_output_channels=self.n_output_channels,
            num_init_features=self.num_init_features,
            num_embeddings=self.num_embeddings,
            growth_rate=self.growth_rate,
            block_config=self.block_config,
            padding=self.padding,
        )


class UNetBackboneConfig(BackboneConfig):
    model_type: Literal["unet"] = "unet"
    in_channels: int
    num_fmaps: int
    num_fmaps_out: int
    fmap_inc_factor: int
    down_sample_factors: list[PydanticCoordinate]

    def model(self):
        module = UNet(
            self.in_channels,
            self.num_fmaps,
            self.fmap_inc_factor,
            self.down_sample_factors,
            num_fmaps_out=self.num_fmaps_out,
            kernel_size_down=[[(3, 3), (3, 3)]] * (len(self.down_sample_factors) + 1),
            kernel_size_up=[[(3, 3), (3, 3)]] * len(self.down_sample_factors),
            constant_upsample=True,
        )

        for _name, layer in module.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        module.context = sum(
            [2**(i+1) for i in range(len(self.down_sample_factors) + 1)]
            + [2**(i+1) for i in range(len(self.down_sample_factors))]
        )

        return module


class DataConfig(BaseModel):
    datasets: list[Path]
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
    pred_delta: bool = False
    architecture_config: Annotated[
        Union[DenseBackboneConfig, UNetBackboneConfig],
        Field(discriminator="model_type"),
    ]

    start: Optional[Path] = None

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def snapshot_container(self) -> Path:
        return self.base_dir / "snapshots.zarr"

    @property
    def validation_container(self) -> Path:
        return self.base_dir / "validation"

    @property
    def loss_file(self) -> Path:
        return self.base_dir / "loss.csv"

    @property
    def val_file(self) -> Path:
        return self.base_dir / "val.csv"
