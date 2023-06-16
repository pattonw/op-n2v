from gunpowder.torch import Predict

import click
import yaml

import random


@click.group()
def v1():
    pass


@v1.command()
@click.option("-s", "--scale-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-d", "--data-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--num-voxels", type=int, default=32)
def visualize_pipeline(data_config, num_voxels):
    from care.pipeline import build_pipeline, get_request, split_batch
    from care.configs import DataConfig

    from funlib.geometry import Coordinate

    import gunpowder as gp

    import neuroglancer

    data_config = DataConfig(**yaml.safe_load(open(data_config, "r").read()))
    pipeline = build_pipeline(data_config, gt_voxel_size=data_config.gt_voxel_size)

    volume_shape = Coordinate((num_voxels,) * 3) * 4

    def load_batch(event):
        with gp.build(pipeline):
            batch = pipeline.request_batch(get_request(volume_shape, lsds=lsds))
        if lsds:
            raw, gt, _weights, _masks, _lsds, _lsd_mask = split_batch(batch, lsds=lsds)
        else:
            raw, gt, _weights, _masks = split_batch(batch, lsds=lsds)

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
            if lsds:
                for scale_level, lsd_scale_array in enumerate(_lsds):
                    dims = neuroglancer.CoordinateSpace(
                        names=["c^", "z", "y", "x"],
                        units="nm",
                        scales=(1,) + tuple(lsd_scale_array.spec.voxel_size),
                    )
                    lsd_vol = neuroglancer.LocalVolume(
                        data=lsd_scale_array.data,
                        voxel_offset=(0,)
                        + tuple(
                            lsd_scale_array.spec.roi.offset
                            / lsd_scale_array.spec.voxel_size
                        ),
                        dimensions=dims,
                    )

                    s.layers[f"lsd_s{scale_level}"] = neuroglancer.ImageLayer(
                        source=lsd_vol,
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


@v1.command()
@click.option("-m", "--model-config", type=click.Path(exists=True, dir_okay=False))
def model_summary(model_config):
    from care.backbones.dense import DenseNet
    from care.configs import BackboneConfig

    from torchsummary import summary

    model_config = BackboneConfig(**yaml.safe_load(open(model_config, "r").read()))

    in_channels = model_config.raw_input_channels + (
        model_config.n_output_channels
        if not model_config.embeddings
        else model_config.num_embeddings
    )
    print(in_channels, model_config)

    model = DenseNet(
        n_input_channels=in_channels,
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        num_embeddings=model_config.num_embeddings,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
        upsample_mode=model_config.upsample_mode,
    ).to("cpu")

    print(summary(model, (model_config.raw_input_channels, 26, 26, 26), device="cpu"))
    print(model)


@v1.command()
@click.option("-s", "--scale-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("--weights/--no-weights", type=bool, default=False)
@click.option("--loss/--no-loss", type=bool, default=False)
@click.option("--argmax/--no-argmax", type=bool, default=False)
def view_snapshots(train_config, weights, loss, argmax):
    from care.configs import TrainConfig

    import daisy

    import neuroglancer

    from scipy.special import softmax
    import numpy as np

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    raw_datasets = [
        f"raw_s{scale_level}" for scale_level in range(num_raw_scale_levels)
    ]
    target_datasets = [
        f"target_s{scale_level}" for scale_level in range(num_gt_scale_levels)
    ]
    pred_datasets = [
        f"pred_s{scale_level}" for scale_level in range(num_raw_scale_levels)
    ]
    weight_datasets = [
        f"weight_s{scale_level}" for scale_level in range(num_gt_scale_levels)
    ]
    loss_datasets = [
        f"loss_s{scale_level}" for scale_level in range(num_gt_scale_levels)
    ]

    with viewer.txn() as s:
        while len(s.layers) > 0:
            del s.layers[0]
        for raw_dataset in raw_datasets:
            daisy_array = daisy.open_ds(
                train_config.snapshot_container,
                f"{raw_dataset}",
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
                train_config.snapshot_container,
                f"{target_dataset}",
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
                train_config.snapshot_container,
                f"{pred_dataset}",
            )
            data = softmax(daisy_array.data, axis=1)
            if argmax:
                data = np.argmax(data, axis=1).astype(np.uint32)

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "z", "y", "x"],
                    units="nm",
                    scales=(1, *daisy_array.voxel_size),
                )
                voxel_offset = (
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                )
            else:

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "c^", "z", "y", "x"],
                    units="nm",
                    scales=(1, 1, *daisy_array.voxel_size),
                )
                voxel_offset = (
                    0,
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                )

            pred_vol = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )
            if argmax:
                s.layers[pred_dataset] = neuroglancer.SegmentationLayer(
                    source=pred_vol,
                )
            else:
                s.layers[pred_dataset] = neuroglancer.ImageLayer(
                    source=pred_vol,
                )

        if weights:
            for weight_dataset in weight_datasets:
                daisy_array = daisy.open_ds(
                    train_config.snapshot_container,
                    f"{weight_dataset}",
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
                    train_config.snapshot_container,
                    f"{loss_dataset}",
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
                            layers=raw_datasets[::-1] + target_datasets
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=raw_datasets[::-1]
                            + pred_datasets
                            + (loss_datasets if loss else [])
                        ),
                    ]
                ),
            ]
        )

    print(viewer)

    input("Enter to quit!")


@v1.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--num-iter", type=int, default=1000000000)
@click.option("-o", "--output", type=click.Path(exists=False))
@click.option("-s", "--smooth", type=float, default=0)
def plot_loss(train_config, num_iter, output, smooth):
    from care.configs import TrainConfig

    import numpy as np
    import matplotlib.pyplot as plt

    def smooth_func(scalars, weight):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val

        return smoothed

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))

    losses = [
        tuple(float(loss) for loss in line.strip("[]()\n").split(",") if len(loss) > 0)
        for line in list(train_config.loss_file.open().readlines())[-num_iter:]
    ]
    loss_resolutions = [np.array(loss_resolution) for loss_resolution in zip(*losses)]

    for loss_resolution in loss_resolutions:
        plt.plot(smooth_func(loss_resolution, smooth))
    plt.savefig(f"{output}.png")


@v1.command()
@click.option("-m", "--model-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-s", "--scale-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-d", "--data-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--iteration", type=int)
def validate(model_config, data_config, train_config, iteration):
    from care.configs import DataConfig, TrainConfig, BackboneConfig
    from care.backbones.dense import DenseNet
    from care.pipeline import build_pipeline, get_request, split_batch

    from funlib.geometry import Coordinate

    import gunpowder as gp
    import daisy

    from sklearn.metrics import f1_score
    from tqdm import tqdm
    import zarr
    import torch
    import numpy as np

    assert torch.cuda.is_available(), "Cannot validate reasonably without cuda!"

    model_config = BackboneConfig(**yaml.safe_load(open(model_config, "r").read()))
    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    data_config = DataConfig(**yaml.safe_load(open(data_config, "r").read()))

    model = DenseNet(
        n_input_channels=model_config.raw_input_channels
        + (
            model_config.n_output_channels
            if not model_config.embeddings
            else model_config.num_embeddings
        ),
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        num_embeddings=model_config.num_embeddings,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
        upsample_mode=model_config.upsample_mode,
    ).cuda()

    torch.backends.cudnn.benchmark = True

    checkpoint = train_config.checkpoint_dir / f"{iteration}"
    assert checkpoint.exists()
    weights = torch.load(checkpoint)
    model.load_state_dict(weights)

    # paths
    validation_pred_dataset = "volumes/val/{crop}/{i}/pred/scale_{scale}"
    validation_emb_dataset = "volumes/val/{crop}/{i}/emb/scale_{scale}"
    validation_raw_dataset = "volumes/val/{crop}/raw/scale_{scale}"

    model = model.eval()
    # validate
    with torch.no_grad():
        for dataset_config in data_config.datasets:
            # TODO: What happens when s0 has different resolutions for different datasets?
            try:
                raw_s0 = daisy.open_ds(
                    dataset_config.dataset_container,
                    dataset_config.raw_dataset + f"/s0",
                )
            except FileExistsError:
                raw_s0 = daisy.open_ds(
                    dataset_config.fallback_dataset_container,
                    dataset_config.raw_dataset + f"/s0",
                )
            assert (
                raw_s0.voxel_size == data_config.gt_voxel_size * 2
            ), f"gt resolution is not double the raw s0 resolution: raw({raw_s0.voxel_size}):gt({data_config.gt_voxel_size})"
            for validation_crop in dataset_config.validation_crops:
                try:
                    gt_ds = daisy.open_ds(
                        dataset_config.dataset_container,
                        dataset_config.gt_group.format(crop_num=validation_crop)
                        + "/all",
                    )
                except FileExistsError:
                    gt_ds = daisy.open_ds(
                        dataset_config.fallback_dataset_container,
                        dataset_config.gt_group.format(crop_num=validation_crop)
                        + "/all",
                    )
                gt_voxel_size = gt_ds.voxel_size

                # prepare an empty dataset from which we can pull 0's
                # in a consistent manner
                daisy.prepare_ds(
                    str(train_config.validation_container),
                    validation_pred_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=num_raw_scale_levels,
                    ),
                    total_roi=gt_ds.roi.snap_to_grid(
                        gt_voxel_size * 2 ** (num_raw_scale_levels),
                        mode="grow",
                    ),
                    voxel_size=gt_voxel_size * 2 ** (num_raw_scale_levels),
                    dtype=np.float32,
                    num_channels=model_config.n_output_channels,
                    delete=True,
                )
                daisy.prepare_ds(
                    str(train_config.validation_container),
                    validation_emb_dataset.format(
                        i=iteration,
                        crop=validation_crop,
                        scale=num_raw_scale_levels,
                    ),
                    total_roi=gt_ds.roi.snap_to_grid(
                        gt_voxel_size * 2 ** (num_raw_scale_levels),
                        mode="grow",
                    ),
                    voxel_size=gt_voxel_size * 2 ** (num_raw_scale_levels),
                    dtype=np.float32,
                    num_channels=model_config.num_embeddings,
                    delete=True,
                )
                for scale_level in range(num_raw_scale_levels - 1, -1, -1):
                    # assumptions:
                    # 1) raw data is provided as a scale pyramid
                    # 2) gt data is provided in labels/all

                    try:
                        raw_ds = daisy.open_ds(
                            dataset_config.dataset_container,
                            dataset_config.raw_dataset + f"/s{scale_level}",
                        )
                    except FileExistsError:
                        raw_ds = daisy.open_ds(
                            dataset_config.fallback_dataset_container,
                            dataset_config.raw_dataset + f"/s{scale_level}",
                        )
                    raw_voxel_size = raw_ds.voxel_size

                    raw_key, upsampled_key, pred_key, emb_key = (
                        gp.ArrayKey("RAW"),
                        gp.ArrayKey("UPSAMPLED"),
                        gp.ArrayKey("PRED"),
                        gp.ArrayKey("EMB"),
                    )
                    input_size = train_config.eval_input_shape_voxels * raw_voxel_size
                    output_size = input_size
                    reference_request = gp.BatchRequest()
                    reference_request.add(
                        raw_key,
                        input_size,
                    )
                    reference_request.add(
                        upsampled_key,
                        input_size,
                    )
                    reference_request.add(
                        pred_key,
                        output_size,
                    )
                    if model_config.embeddings:
                        reference_request.add(emb_key, output_size)
                    out_voxel_size = raw_voxel_size / 2
                    out_roi = gt_ds.roi.snap_to_grid(raw_voxel_size, mode="grow")
                    if any([a < b for a, b in zip(out_roi.shape, input_size)]):
                        context = (
                            gp.Coordinate(
                                *(
                                    max(out_shape - gt_shape, 0)
                                    for gt_shape, out_shape in zip(
                                        out_roi.shape, output_size
                                    )
                                )
                            )
                            + 1
                        ) / 2
                        out_roi = out_roi.grow(context, context).snap_to_grid(
                            raw_voxel_size, mode="grow"
                        )

                    out_offset = Coordinate(
                        max(a, b) for a, b in zip(out_roi.offset, raw_ds.roi.offset)
                    )
                    out_offset += (-out_offset) % out_voxel_size
                    out_roi.offset = out_offset

                    val_pipeline = (
                        (
                            gp.ZarrSource(
                                str(dataset_config.dataset_container),
                                {raw_key: f"volumes/raw/s{scale_level}"},
                                array_specs={
                                    raw_key: gp.ArraySpec(
                                        roi=raw_ds.roi,
                                        voxel_size=raw_ds.voxel_size,
                                        interpolatable=True,
                                    )
                                },
                            )
                            + gp.Normalize(raw_key),
                            gp.ZarrSource(
                                str(train_config.validation_container),
                                {
                                    upsampled_key: (
                                        validation_pred_dataset
                                        if not model_config.embeddings
                                        else validation_emb_dataset
                                    ).format(
                                        i=iteration,
                                        crop=validation_crop,
                                        scale=scale_level + 1,
                                    )
                                },
                            )
                            + gp.Pad(upsampled_key, None),
                        )
                        + gp.MergeProvider()
                        + gp.Unsqueeze([raw_key])
                        + gp.Unsqueeze([raw_key, upsampled_key])
                        + Predict(
                            model=model,
                            inputs={
                                "raw": raw_key,
                                "upsampled": upsampled_key,
                            },
                            outputs={0: emb_key, 1: pred_key},
                            array_specs={
                                pred_key: gp.ArraySpec(
                                    roi=out_roi,
                                    voxel_size=out_voxel_size,
                                    dtype=np.float32,
                                ),
                                emb_key: gp.ArraySpec(
                                    roi=out_roi,
                                    voxel_size=out_voxel_size,
                                    dtype=np.float32,
                                ),
                            },
                        )
                        + gp.Squeeze([raw_key, emb_key, pred_key])
                        + gp.Squeeze([raw_key])
                        + gp.ZarrWrite(
                            dataset_names={
                                pred_key: validation_pred_dataset.format(
                                    i=iteration,
                                    crop=validation_crop,
                                    scale=scale_level,
                                ),
                                emb_key: validation_emb_dataset.format(
                                    i=iteration,
                                    crop=validation_crop,
                                    scale=scale_level,
                                ),
                                raw_key: validation_raw_dataset.format(
                                    crop=validation_crop,
                                    scale=scale_level,
                                ),
                            },
                            output_dir=str(train_config.validation_container.parent),
                            output_filename=train_config.validation_container.name,
                        )
                        + gp.Scan(reference=reference_request)
                    )

                    # prepare the dataset to be written to
                    pred_ds = daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_pred_dataset.format(
                            i=iteration,
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                        total_roi=out_roi,
                        voxel_size=out_voxel_size,
                        dtype=np.float32,
                        write_size=output_size,
                        num_channels=model_config.n_output_channels,
                        delete=True,
                    )

                    # prepare emb ds
                    daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_emb_dataset.format(
                            i=iteration,
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                        total_roi=out_roi,
                        voxel_size=out_voxel_size,
                        dtype=np.float32,
                        write_size=output_size,
                        num_channels=model_config.num_embeddings,
                        delete=True,
                    )
                    # prepare raw ds
                    daisy.prepare_ds(
                        str(train_config.validation_container),
                        validation_raw_dataset.format(
                            crop=validation_crop,
                            scale=scale_level,
                        ),
                        total_roi=out_roi,
                        voxel_size=raw_voxel_size,
                        dtype=np.float32,
                        write_size=output_size,
                        num_channels=None,
                        delete=True,
                    )

                    with gp.build(val_pipeline):
                        val_pipeline.request_batch(gp.BatchRequest())

                # compare prediction s0 to gt
                gt_data = gt_ds.to_ndarray(gt_ds.roi)
                label_data = np.zeros_like(gt_data)
                for label, label_ids in enumerate(data_config.categories):
                    label_data[np.isin(gt_data, label_ids)] = label + 1
                pred_data = pred_ds.to_ndarray(pred_ds.roi)
                pred_data = np.argmax(pred_data, axis=0)

                val_score = f1_score(
                    label_data.flatten(),
                    pred_data.flatten(),
                    average=None,
                )
                print(
                    f"Iteration: {iteration}, crop: {validation_crop}, f1_score: {val_score}"
                )
