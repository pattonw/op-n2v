import click
import yaml


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("--workers/--no-workers", type=bool, default=True)
def train(train_config, workers):
    from care.configs import DataConfig, TrainConfig, BackboneConfig
    from care.backbones.dense import DenseNet
    from care.pipeline import build_pipeline, get_request, split_batch

    from funlib.geometry import Coordinate

    import gunpowder as gp

    from tqdm import tqdm
    import zarr
    import torch
    import numpy as np

    import wandb


    wandb.init(project="op-n2v", name=str(train_config).split("/")[-1])

    def save_snapshot(
        name, dataset: np.ndarray, offset: Coordinate, voxel_size: Coordinate
    ):
        sample = dataset[0]  # select a sample from batch
        if name not in snapshot_zarr:
            snapshot_dataset = snapshot_zarr.create_dataset(
                name,
                shape=(0, *sample.shape),
                dtype=dataset.dtype,
            )
            snapshot_dataset.attrs["resolution"] = voxel_size
            snapshot_dataset.attrs["offset"] = offset
            snapshot_dataset.attrs["axes"] = ["iteration^"] + ["c^", "z", "y", "x"][
                -len(sample.shape) :
            ]
        else:
            snapshot_dataset = snapshot_zarr[name]
        snapshot_dataset.append(sample.reshape((1, *sample.shape)), axis=0)

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    model_config = train_config.architecture_config
    data_config = train_config.data_config

    model = DenseNet(
        n_input_channels=model_config.raw_input_channels,
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        num_embeddings=model_config.num_embeddings,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
    )

    if train_config.loss_file.exists():
        loss_stats = [
            tuple(float(x) for x in line.strip("[]()\n").split(",") if len(x) > 0)
            for line in train_config.loss_file.open("r").readlines()
        ]
    else:
        loss_stats = []

    snapshot_zarr = zarr.open(f"{train_config.snapshot_container}")

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=train_config.warmup
    )

    torch.backends.cudnn.benchmark = True

    if train_config.start is not None:
        weights = torch.load(train_config.start)
        try:
            model.load_state_dict(weights)
        except RuntimeError as e:
            print(e)
            pass

    if not train_config.checkpoint_dir.exists():
        train_config.checkpoint_dir.mkdir(parents=True)
    checkpoints = sorted([int(f.name) for f in train_config.checkpoint_dir.iterdir()])
    most_recent = 0 if len(checkpoints) == 0 else checkpoints[-1]
    if most_recent > 0:
        weights = torch.load(train_config.checkpoint_dir / f"{most_recent}")
        try:
            model.load_state_dict(weights)
        except RuntimeError as e:
            # Couldn't load the weights. We will just continue with most of
            # the weights loaded
            pass
        print(f"Starting from: {most_recent}")
        loss_stats = loss_stats[:most_recent]
    else:
        print(f"Starting from scratch!")
        loss_stats = []

    # get pipeline. Stack to create appropriate batch size, add precache
    pipeline = build_pipeline(data_config)
    pipeline += gp.Stack(train_config.batch_size)
    if workers:
        pipeline += gp.PreCache(num_workers=train_config.num_workers)

    model = model
    n_dims = train_config.input_shape_voxels.dims
    context = Coordinate((model.context,) * n_dims)

    with gp.build(pipeline):

        for i in tqdm(range(most_recent, train_config.num_iterations)):
            batch_request = get_request(
                train_config.input_shape_voxels,
                train_config.input_shape_voxels - (context * 2),
            )
            (raw_input, raw_context) = split_batch(
                pipeline.request_batch(batch_request),
            )

            # convert raw, target and weight to tensor
            if len(raw_input.data.shape) == train_config.input_shape_voxels.dims + 1:
                torch_raw_input = torch.unsqueeze(
                    torch.from_numpy(raw_input.data).float(), 1
                )
            else:
                torch_raw_input = torch.from_numpy(raw_input.data).float()

            torch_raw_context = torch.from_numpy(raw_context.data).float()

            _, pred = model.forward(torch_raw_input)

            loss = loss_func(pred, torch_raw_context)

            # standard training steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({"loss": loss.item()})

            if i % train_config.checkpoint_interval == 0:
                torch.save(model.state_dict(), train_config.checkpoint_dir / f"{i}")

            if i % train_config.snapshot_interval == 0:

                pred_data = pred.detach().cpu().numpy()

                for name, dataset, offset, voxel_size in zip(
                    [
                        "raw",
                        "raw_context",
                        "pred",
                    ],
                    [raw_input.data, raw_context.data, pred_data],
                    [
                        raw_input.spec.roi.offset,
                        raw_context.spec.roi.offset,
                        raw_context.spec.roi.offset,
                    ],
                    [
                        raw_input.spec.voxel_size,
                        raw_context.spec.voxel_size,
                        raw_context.spec.voxel_size,
                    ],
                ):
                    save_snapshot(name, dataset, offset, voxel_size)

                # keep track in an attribute which iterations have been stored
                snapshot_zarr.attrs["iterations"] = snapshot_zarr.attrs.get(
                    "iterations", list()
                ) + [i]


        wandb.finish()