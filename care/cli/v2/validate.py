import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--iteration", type=int)
def validate(train_config, iteration):
    from care.configs import TrainConfig, StorageConfig
    from care.backbones.dense import DenseNet
    from care.gp.context import DeContext, Context


    from funlib.persistence import open_ds

    import gunpowder as gp
    from funlib.persistence import prepare_ds

    import yaml
    import torch
    import numpy as np

    assert torch.cuda.is_available(), "Cannot validate reasonably without cuda!"

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
    ).cuda()

    torch.backends.cudnn.benchmark = True

    checkpoint = train_config.checkpoint_dir / f"{iteration}"
    assert checkpoint.exists()
    weights = torch.load(checkpoint)
    model.load_state_dict(weights)

    # paths
    validation_pred_dataset = "volumes/val/{crop}/{i}/pred"
    validation_target_dataset = "volumes/val/{crop}/{i}/target"
    validation_care_dataset = "volumes/val/{crop}/{i}/care"
    validation_emb_dataset = "volumes/val/{crop}/{i}/emb"
    validation_raw_dataset = "volumes/val/{crop}/raw"

    model = model.eval()
    # validate
    with torch.no_grad():
        for dataset_config_path in data_config.datasets:
            # TODO: What happens when s0 has different resolutions for different datasets?
            dataset_config = StorageConfig(
                **yaml.safe_load(dataset_config_path.open("r").read())
            )
            dataset_name = dataset_config_path.name[:-5]
            raw_s0 = open_ds(
                dataset_config.container,
                dataset_config.dataset,
            )
            raw_container = dataset_config.container
            voxel_size = raw_s0.voxel_size

            raw_key, target_key, care_raw_key, pred_key, emb_key = (
                gp.ArrayKey("RAW"),
                gp.ArrayKey("TARGET"),
                gp.ArrayKey("CARE_RAW"),
                gp.ArrayKey("PRED"),
                gp.ArrayKey("EMB"),
            )
            input_size = train_config.eval_input_shape_voxels * voxel_size
            context = voxel_size * model.context
            output_size = input_size - context * 2
            care_size = output_size - (context + voxel_size) * 2
            reference_request = gp.BatchRequest()
            reference_request.add(
                raw_key,
                input_size,
            )
            reference_request.add(
                pred_key,
                output_size,
            )
            reference_request.add(
                target_key,
                output_size,
            )
            reference_request.add(
                care_raw_key,
                care_size,
            )
            if model_config.embeddings:
                reference_request.add(emb_key, output_size)
            in_roi = raw_s0.roi
            pred_roi = in_roi.grow(-context - voxel_size, -context - voxel_size)
            care_roi = pred_roi.grow(-context, -context)

            val_pipeline = (
                (
                    gp.ZarrSource(
                        raw_container,
                        {raw_key: dataset_config.dataset.format(level=0)},
                        array_specs={
                            raw_key: gp.ArraySpec(
                                roi=raw_s0.roi,
                                voxel_size=raw_s0.voxel_size,
                                interpolatable=True,
                            )
                        },
                    )
                    + gp.Normalize(raw_key)
                    + Context(raw_key, target_key, data_config.neighborhood)
                )
                + (gp.Unsqueeze([raw_key]) + gp.Unsqueeze([raw_key]) if len(raw_s0.data.shape) == raw_s0.roi.dims else gp.Unsqueeze([raw_key]))
                + gp.torch.nodes.Predict(
                    model=model,
                    inputs={
                        "raw": raw_key,
                    },
                    outputs={0: emb_key, 1: pred_key},
                    array_specs={
                        pred_key: gp.ArraySpec(
                            roi=pred_roi,
                            voxel_size=voxel_size,
                            dtype=np.float32,
                        ),
                        emb_key: gp.ArraySpec(
                            roi=pred_roi,
                            voxel_size=voxel_size,
                            dtype=np.float32,
                        ),
                    },
                    checkpoint=f"{checkpoint}",
                )
                + ((gp.Squeeze([raw_key, emb_key, pred_key])
                + gp.Squeeze([raw_key])) if len(raw_s0.data.shape) == raw_s0.roi.dims else gp.Squeeze([raw_key, emb_key, pred_key]))
                + DeContext(pred_key, care_raw_key, data_config.neighborhood)
                + gp.ZarrWrite(
                    dataset_names={
                        care_raw_key: validation_care_dataset.format(
                            i=iteration,
                            crop=dataset_name,
                        ),
                        pred_key: validation_pred_dataset.format(
                            i=iteration,
                            crop=dataset_name,
                        ),
                        target_key: validation_target_dataset.format(
                            i=iteration,
                            crop=dataset_name,
                        ),
                        emb_key: validation_emb_dataset.format(
                            i=iteration,
                            crop=dataset_name,
                        ),
                        raw_key: validation_raw_dataset.format(
                            crop=dataset_name,
                        ),
                    },
                    output_dir=str(train_config.validation_container.parent),
                    output_filename=train_config.validation_container.name,
                )
                + gp.Scan(reference=reference_request)
            )

            # prepare the datasets to be written to
            prepare_ds(
                str(train_config.validation_container),
                validation_care_dataset.format(
                    i=iteration,
                    crop=dataset_name,
                ),
                total_roi=care_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
                write_size=output_size,
                num_channels=model_config.raw_input_channels,
                delete=True,
            )

            prepare_ds(
                str(train_config.validation_container),
                validation_pred_dataset.format(
                    i=iteration,
                    crop=dataset_name,
                ),
                total_roi=pred_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
                write_size=output_size,
                num_channels=model_config.n_output_channels,
                delete=True,
            )

            prepare_ds(
                str(train_config.validation_container),
                validation_target_dataset.format(
                    i=iteration,
                    crop=dataset_name,
                ),
                total_roi=pred_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
                write_size=output_size,
                num_channels=model_config.n_output_channels,
                delete=True,
            )

            # prepare emb ds
            prepare_ds(
                str(train_config.validation_container),
                validation_emb_dataset.format(
                    i=iteration,
                    crop=dataset_name,
                ),
                total_roi=pred_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
                write_size=output_size,
                num_channels=model_config.num_embeddings,
                delete=True,
            )
            # prepare raw ds
            prepare_ds(
                str(train_config.validation_container),
                validation_raw_dataset.format(
                    crop=dataset_name,
                ),
                total_roi=in_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
                write_size=output_size,
                num_channels=model_config.raw_input_channels,
                delete=True,
            )

            with gp.build(val_pipeline):
                val_pipeline.request_batch(gp.BatchRequest())

            print(f"Iteration: {iteration}")
