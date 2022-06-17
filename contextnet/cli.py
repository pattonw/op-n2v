import click
import yaml

import sys
from pathlib import Path


@click.group()
def main(args=None):
    """Console script for contextnet."""
    return None


@main.command()
@click.option("-s", "--scale-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-d", "--data-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--num-voxels", type=int, default=32)
def visualize_pipeline(scale_config, data_config, num_voxels):
    from contextnet.pipeline import build_pipeline, get_request, split_batch
    from contextnet.configs import ScaleConfig, DataConfig

    from funlib.geometry import Coordinate

    import gunpowder as gp

    import neuroglancer

    scale_config = ScaleConfig(**yaml.safe_load(open(scale_config, "r").read()))
    data_config = DataConfig(**yaml.safe_load(open(data_config, "r").read()))
    pipeline = build_pipeline(
        data_config, scale_config, gt_voxel_size=data_config.gt_voxel_size
    )

    volume_shape = Coordinate((num_voxels,) * 3) * 4

    def load_batch(event):
        with gp.build(pipeline):
            batch = pipeline.request_batch(get_request(volume_shape, scale_config))
        raw, gt, _weights = split_batch(batch, scale_config)

        with viewer.txn() as s:
            while len(s.layers) > 0:
                del s.layers[0]

            # reverse order for raw so we can set opacity to 1, this
            # way higher res raw replaces low res when available
            for scale_level, raw_scale_array in list(enumerate(raw))[::-1]:

                dims = neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=raw_scale_array.spec.voxel_size,
                )

                raw_vol = neuroglancer.LocalVolume(
                    data=raw_scale_array.data,
                    voxel_offset=raw_scale_array.spec.roi.offset
                    / raw_scale_array.spec.voxel_size,
                    dimensions=dims,
                )

                s.layers[f"raw_s{scale_level}"] = neuroglancer.ImageLayer(
                    source=raw_vol, opacity=1.0
                )
            for scale_level, gt_scale_array in enumerate(gt):
                dims = neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=gt_scale_array.spec.voxel_size,
                )
                gt_vol = neuroglancer.LocalVolume(
                    data=gt_scale_array.data,
                    voxel_offset=gt_scale_array.spec.roi.offset
                    / gt_scale_array.spec.voxel_size,
                    dimensions=dims,
                )

                s.layers[f"gt_s{scale_level}"] = neuroglancer.SegmentationLayer(
                    source=gt_vol,
                )
            s.layout = "yz"

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    viewer.actions.add("load_batch", load_batch)

    with viewer.config_state.txn() as s:
        s.input_event_bindings.data_view["keyt"] = "load_batch"

    print(viewer)
    load_batch(None)

    input("Enter to quit!")


@main.command()
@click.option("-m", "--model-config", type=click.Path(exists=True, dir_okay=False))
def model_summary(model_config):
    from contextnet.backbones.dense import DenseNet
    from contextnet.configs import BackboneConfig

    from torchsummary import summary

    model_config = BackboneConfig(**yaml.safe_load(open(model_config, "r").read()))

    model = DenseNet(
        n_input_channels=model_config.n_input_channels,
        num_init_features=model_config.n_output_channels,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
    ).to("cpu")

    print(summary(model, (model_config.n_input_channels, 26, 26, 26), device="cpu"))
    print(model)


@main.command()
@click.option("-s", "--scale-config", type=click.Path(exists=True, dir_okay=False))
@click.option("--weights/--no-weights", type=bool, default=False)
@click.option("--loss/--no-loss", type=bool, default=False)
def view_snapshots(scale_config, weights, loss):
    from contextnet.configs import ScaleConfig

    import daisy

    import neuroglancer

    from scipy.special import softmax
    import numpy as np

    scale_config = ScaleConfig(**yaml.safe_load(open(scale_config, "r").read()))

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    raw_datasets = [
        f"raw_s{scale_level}"
        for scale_level in range(scale_config.num_raw_scale_levels)
    ]
    target_datasets = [
        f"target_s{scale_level}"
        for scale_level in range(scale_config.num_gt_scale_levels)
    ]
    pred_datasets = [
        f"pred_s{scale_level}"
        for scale_level in range(scale_config.num_raw_scale_levels)
    ]
    weight_datasets = [
        f"weight_s{scale_level}"
        for scale_level in range(scale_config.num_gt_scale_levels)
    ]
    loss_datasets = [
        f"loss_s{scale_level}"
        for scale_level in range(scale_config.num_gt_scale_levels)
    ]

    with viewer.txn() as s:
        while len(s.layers) > 0:
            del s.layers[0]
        for raw_dataset in raw_datasets[::-1]:
            daisy_array = daisy.open_ds(
                "/nrs/cellmap/pattonw/scale_net/snapshots.zarr",
                f"volumes/{raw_dataset}",
            )

            dims = neuroglancer.CoordinateSpace(
                names=["iterations", "z", "y", "x"],
                units="nm",
                scales=(1, *daisy_array.voxel_size),
            )

            raw_vol = neuroglancer.LocalVolume(
                data=daisy_array.data,
                voxel_offset=(
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                ),
                dimensions=dims,
            )

            s.layers[raw_dataset] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)
        for target_dataset in target_datasets:
            daisy_array = daisy.open_ds(
                "/nrs/cellmap/pattonw/scale_net/snapshots.zarr",
                f"volumes/{target_dataset}",
            )

            dims = neuroglancer.CoordinateSpace(
                names=["iterations", "z", "y", "x"],
                units="nm",
                scales=(1, *daisy_array.voxel_size),
            )

            target_vol = neuroglancer.LocalVolume(
                data=daisy_array.data,
                voxel_offset=(
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                ),
                dimensions=dims,
            )

            s.layers[target_dataset] = neuroglancer.SegmentationLayer(source=target_vol)
        for pred_dataset in pred_datasets:
            daisy_array = daisy.open_ds(
                "/nrs/cellmap/pattonw/scale_net/snapshots.zarr",
                f"volumes/{pred_dataset}",
            )
            data = softmax(daisy_array.data, axis=1)
            data = np.argmax(data, axis=1).astype(np.uint32)

            dims = neuroglancer.CoordinateSpace(
                names=["iterations", "z", "y", "x"],
                units="nm",
                scales=(1, *daisy_array.voxel_size),
            )

            pred_vol = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=(
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                ),
                dimensions=dims,
            )

            s.layers[pred_dataset] = neuroglancer.SegmentationLayer(
                source=pred_vol,
            )

        if weights:
            for weight_dataset in weight_datasets:
                daisy_array = daisy.open_ds(
                    "/nrs/cellmap/pattonw/scale_net/snapshots.zarr",
                    f"volumes/{weight_dataset}",
                )

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "z", "y", "x"],
                    units="nm",
                    scales=(1, *daisy_array.voxel_size),
                )

                target_vol = neuroglancer.LocalVolume(
                    data=daisy_array.data,
                    voxel_offset=(
                        0,
                        *(daisy_array.roi.offset / daisy_array.voxel_size),
                    ),
                    dimensions=dims,
                )

                s.layers[weight_dataset] = neuroglancer.ImageLayer(source=target_vol)
        if loss:
            for loss_dataset in loss_datasets:
                daisy_array = daisy.open_ds(
                    "/nrs/cellmap/pattonw/scale_net/snapshots.zarr",
                    f"volumes/{loss_dataset}",
                )

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "z", "y", "x"],
                    units="nm",
                    scales=(1, *daisy_array.voxel_size),
                )

                target_vol = neuroglancer.LocalVolume(
                    data=daisy_array.data,
                    voxel_offset=(
                        0,
                        *(daisy_array.roi.offset / daisy_array.voxel_size),
                    ),
                    dimensions=dims,
                )

                s.layers[loss_dataset] = neuroglancer.ImageLayer(source=target_vol)

        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=raw_datasets + target_datasets
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=raw_datasets + pred_datasets
                        ),
                    ]
                ),
            ]
        )

    print(viewer)

    input("Enter to quit!")


@main.command()
@click.option("-m", "--model-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-s", "--scale-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-d", "--data-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
def train(model_config, scale_config, data_config, train_config):
    from contextnet.configs import ScaleConfig, DataConfig, TrainConfig, BackboneConfig
    from contextnet.backbones.dense import DenseNet
    from contextnet.pipeline import build_pipeline, get_request, split_batch

    from funlib.geometry import Coordinate

    import gunpowder as gp

    from tqdm import tqdm
    import zarr
    import torch
    import numpy as np

    scale_config = ScaleConfig(**yaml.safe_load(open(scale_config, "r").read()))
    model_config = BackboneConfig(**yaml.safe_load(open(model_config, "r").read()))
    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    data_config = DataConfig(**yaml.safe_load(open(data_config, "r").read()))

    model = DenseNet(
        n_input_channels=model_config.n_input_channels,
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
    ).cuda()

    if train_config.loss_file.exists():
        loss_stats = [float(x) for x in train_config.loss_file.open("r").readlines()]
    else:
        loss_stats = []

    snapshot_zarr = zarr.open(f"{train_config.snapshot_container}")

    upsample = torch.nn.Upsample(
        scale_factor=2,
    )
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.RAdam(model.parameters(), lr=train_config.learning_rate)

    torch.backends.cudnn.benchmark = True

    if not train_config.checkpoint_dir.exists():
        train_config.checkpoint_dir.mkdir(parents=True)
    checkpoints = [int(f.name) for f in train_config.checkpoint_dir.iterdir()]
    if len(checkpoints) > 0:
        most_recent = sorted(checkpoints)[-1]
        weights = torch.load(train_config.checkpoint_dir / f"{most_recent}")
        model.load_state_dict(weights)
        print(f"Starting from: {most_recent}")

    # get pipeline. Stack to create appropriate batch size, add precache
    pipeline = build_pipeline(data_config, scale_config, data_config.gt_voxel_size)
    pipeline += gp.Stack(train_config.batch_size)
    pipeline += gp.PreCache(num_workers=train_config.num_workers)

    with gp.build(pipeline):

        for i in tqdm(range(train_config.num_iterations)):
            batch_request = get_request(
                train_config.input_shape_voxels
                * data_config.gt_voxel_size
                * scale_config.scale_factor,
                scale_config,
            )
            raws, targets, weights = split_batch(
                pipeline.request_batch(batch_request), scale_config
            )

            optimizer.zero_grad()

            # forward pass
            predictions = {}
            for scale_level, raw in list(enumerate(raws))[::-1]:
                raw_shape = raw.spec.roi.shape / raw.spec.voxel_size
                previous_pred = predictions.get(
                    scale_level + 1,
                    torch.zeros(
                        (
                            train_config.batch_size,
                            model_config.n_output_channels,
                            *raw_shape / 2,
                        )
                    )
                    .cuda()
                    .float(),
                )
                previous_pred = upsample(previous_pred)
                # convert raw to tensor and add channel dim
                torch_raw = torch.unsqueeze(
                    torch.from_numpy(raw.data).cuda().float(), 1
                )
                if Coordinate(previous_pred.shape[2:]) - Coordinate(
                    torch_raw.shape[2:]
                ) != Coordinate(0, 0, 0):
                    upsampled_shape = Coordinate(previous_pred.shape[2:])
                    context = (upsampled_shape - raw_shape) / 2
                    previous_pred = previous_pred[
                        (slice(None), slice(None))
                        + tuple(slice(c, c + r) for c, r in zip(context, raw_shape))
                    ]

                concatenated = torch.concat(
                    [
                        torch_raw,
                        previous_pred,
                    ],
                    dim=1,
                )
                pred = model.forward(concatenated.cuda().float())
                print(
                    "concatenated",
                    scale_level,
                    concatenated.min().item(),
                    concatenated.max().item(),
                    pred.min().item(),
                    pred.max().item(),
                )
                predictions[scale_level] = pred

            losses = []
            weighted_losses = []
            for scale_level, (target, weight) in enumerate(zip(targets, weights)):
                # convert raw to tensor and add batch dim
                torch_target = (
                    torch.from_numpy(target.data.astype(np.int8)).cuda().long()
                )
                torch_weight = torch.from_numpy(weight.data).cuda().float()
                pred = upsample(predictions[scale_level])
                element_loss = loss_func(pred, torch_target)
                weighted_loss = element_loss * torch_weight
                weighted_losses.append(weighted_loss)
                loss = weighted_loss.mean()
                losses.append(loss)

            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            loss_stats.append(loss.item())
            var_mean = torch.var_mean(predictions[0])
            for scale_level, prediction in predictions.items():
                print(scale_level, prediction.min().item(), prediction.max().item())
            print(
                f"Losses: {[loss.item() for loss in losses]}, "
                f"Var: {var_mean[0].item():.2f}, "
                f"Mean: {var_mean[1].item():.2f}"
            )

            if i % train_config.checkpoint_interval == 0:
                torch.save(model.state_dict(), train_config.checkpoint_dir / f"{i}")

            if i % train_config.snapshot_interval == 0:
                with train_config.loss_file.open("w") as f:
                    f.write("\n".join([str(x) for x in loss_stats]))

                snapshot_zarr.attrs["iterations"] = snapshot_zarr.attrs.get(
                    "iterations", list()
                ) + [i]
                for scale_level, raw in enumerate(raws):
                    dataset_name = f"raw_s{scale_level}"
                    sample = raw.data[0]  # select a sample from batch
                    if dataset_name not in snapshot_zarr:
                        snapshot_raw = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=raw.data.dtype,
                        )
                        snapshot_raw.attrs["resolution"] = raw.spec.voxel_size
                        snapshot_raw.attrs["offset"] = raw.spec.roi.offset
                        snapshot_raw.attrs["axes"] = ["iteration^", "z", "y", "x"]
                    else:
                        snapshot_raw = snapshot_zarr[dataset_name]
                    snapshot_raw.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, target in enumerate(targets):
                    dataset_name = f"target_s{scale_level}"
                    sample = target.data[0]
                    if dataset_name not in snapshot_zarr:
                        snapshot_target = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_target.attrs["resolution"] = target.spec.voxel_size
                        snapshot_target.attrs["offset"] = target.spec.roi.offset
                        snapshot_target.attrs["axes"] = [
                            "iteration^",
                            "c^",
                            "z",
                            "y",
                            "x",
                        ]
                    else:
                        snapshot_target = snapshot_zarr[dataset_name]
                    snapshot_target.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, prediction in predictions.items():
                    sample = prediction.detach().cpu().numpy()[0]
                    dataset_name = f"pred_s{scale_level}"
                    if dataset_name not in snapshot_zarr:
                        snapshot_pred = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_pred.attrs["resolution"] = raws[
                            scale_level
                        ].spec.voxel_size
                        snapshot_pred.attrs["offset"] = raws[
                            scale_level
                        ].spec.roi.offset
                        snapshot_pred.attrs["axes"] = [
                            "iteration^",
                            "c^",
                            "z",
                            "y",
                            "x",
                        ]
                    else:
                        snapshot_pred = snapshot_zarr[dataset_name]
                    snapshot_pred.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, weight in enumerate(weights):
                    dataset_name = f"weight_s{scale_level}"
                    sample = weight.data[0]
                    if dataset_name not in snapshot_zarr:
                        snapshot_pred = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_pred.attrs["resolution"] = weight.spec.voxel_size
                        snapshot_pred.attrs["offset"] = weight.spec.roi.offset
                        snapshot_pred.attrs["axes"] = ["iteration^", "z", "y", "x"]
                    else:
                        snapshot_pred = snapshot_zarr[dataset_name]
                    snapshot_pred.append(sample.reshape((1, *sample.shape)), axis=0)
                for scale_level, weighted_loss in enumerate(weighted_losses):
                    sample = weighted_loss.detach().cpu().numpy()[0]
                    dataset_name = f"loss_s{scale_level}"
                    if dataset_name not in snapshot_zarr:
                        snapshot_pred = snapshot_zarr.create_dataset(
                            dataset_name,
                            shape=(0, *sample.shape),
                            dtype=sample.dtype,
                        )
                        snapshot_pred.attrs["resolution"] = targets[
                            scale_level
                        ].spec.voxel_size
                        snapshot_pred.attrs["offset"] = targets[
                            scale_level
                        ].spec.roi.offset
                        snapshot_pred.attrs["axes"] = ["iteration^", "z", "y", "x"]
                    else:
                        snapshot_pred = snapshot_zarr[dataset_name]
                    snapshot_pred.append(sample.reshape((1, *sample.shape)), axis=0)


if __name__ == "__main__":
    sys.exit(main())
