import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--iteration", type=int, default=0)
@click.option("-c", "--crop", type=int)
def view_validations(train_config, iteration, crop):
    from care.configs import TrainConfig, DataConfig, DataSetConfig

    import daisy

    import neuroglancer
    import numpy as np
    import yaml

    import itertools

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    data_config = train_config.data_config

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    validation_pred_dataset = "volumes/val/{crop}/{i}/pred"
    validation_emb_dataset = "volumes/val/{crop}/{i}/emb"
    validation_raw_dataset = "volumes/val/{crop}/raw"

    datasets: list[tuple[list[str], list[str], list[str], str]] = []

    def add_layers(s, crop):
        raw = f"{crop}_raw"
        try:
            dataset = raw
            daisy_array = daisy.open_ds(
                f"{train_config.validation_container}",
                validation_raw_dataset.format(i=iteration, crop=crop),
            )

            dims = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=daisy_array.voxel_size,
            )

            raw_vol = neuroglancer.LocalVolume(
                data=daisy_array.data,
                voxel_offset=(*(daisy_array.roi.offset / daisy_array.voxel_size),),
                dimensions=dims,
            )

            s.layers[dataset] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)
        except KeyError:
            pass

        # pred
        pred = f"{crop}_pred"
        try:
            dataset = pred
            daisy_array = daisy.open_ds(
                f"{train_config.validation_container}",
                validation_pred_dataset.format(i=iteration, crop=crop),
            )

            dims = neuroglancer.CoordinateSpace(
                names=["c^", "z", "y", "x"],
                units="nm",
                scales=(1, *daisy_array.voxel_size),
            )
            voxel_offset = (
                0,
                *((daisy_array.roi.offset / daisy_array.voxel_size) + 13),
            )

            pred_data = daisy_array.to_ndarray(daisy_array.roi)
            sub_arrays = [
                pred_data[i][tuple(slice(13 - a, (-13 - a) or None, 1) for a in n)]
                for i, n in enumerate(data_config.neighborhood)
            ]
            care_data = np.stack(
                sub_arrays,
                axis=0,
            )

            pred_vol = neuroglancer.LocalVolume(
                data=care_data,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )

            s.layers[dataset] = neuroglancer.ImageLayer(source=pred_vol, opacity=1.0)

            # add debug layers:
            dims2 = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=(*daisy_array.voxel_size,),
            )
            voxel_offset2 = (
                *((daisy_array.roi.offset / daisy_array.voxel_size) + 13),
            )

            means = care_data.mean(axis=0)
            debug_vol = neuroglancer.LocalVolume(
                data=means,
                voxel_offset=voxel_offset2,
                dimensions=dims2,
            )

            s.layers["mean"] = neuroglancer.ImageLayer(source=debug_vol, opacity=1.0)

            variances = care_data.var(axis=0)
            debug_vol = neuroglancer.LocalVolume(
                data=variances,
                voxel_offset=voxel_offset2,
                dimensions=dims2,
            )

            s.layers["var"] = neuroglancer.ImageLayer(source=debug_vol, opacity=1.0)

            normed = (care_data - means) / variances
            debug_vol = neuroglancer.LocalVolume(
                data=normed,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )

            s.layers["normed"] = neuroglancer.ImageLayer(source=debug_vol, opacity=1.0)

            debug_vol = neuroglancer.LocalVolume(
                data=normed.mean(axis=0),
                voxel_offset=voxel_offset2,
                dimensions=dims2,
            )

            s.layers["normed2"] = neuroglancer.ImageLayer(source=debug_vol, opacity=1.0)

            

        except KeyError:
            pass

        return (raw, pred, ["mean", "var"])

    with viewer.txn() as s:
        for dataset_yaml in data_config.datasets:
            dataset_config = DataSetConfig(**yaml.safe_load(dataset_yaml.open().read()))
            for validation_crop in dataset_config.validation:
                if validation_crop == crop:
                    print(f"Adding layers for {validation_crop}")
                    raw, pred, debug = add_layers(s, validation_crop)
                    datasets.append((raw, pred, debug))

        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=[dataset[0] for dataset in datasets]
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=[dataset[1] for dataset in datasets]
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=list(itertools.chain([dataset[2] for dataset in datasets]))
                        ),
                    ]
                ),
            ]
        )

    # viewer.actions.add("switch_dataset", set_layout)

    # with viewer.config_state.txn() as s:
    #     s.input_event_bindings.data_view["keyt"] = "swith_dataset"

    print(viewer)

    input("Enter to quit!")
