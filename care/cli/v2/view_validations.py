import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--iteration", type=int, default=0)
@click.option("-c", "--crop", type=int)
def view_validations(train_config, iteration, crop):
    from care.configs import TrainConfig, DataConfig, StorageConfig

    from funlib.persistence import open_ds

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
    validation_care_dataset = "volumes/val/{crop}/{i}/care"

    datasets: list[tuple[list[str], list[str], list[str], str]] = []

    def add_layers(s, crop):
        raw = f"{crop}_raw"
        try:
            dataset = raw
            daisy_array = open_ds(
                f"{train_config.validation_container}",
                validation_raw_dataset.format(i=iteration, crop=crop),
            )

            dims = neuroglancer.CoordinateSpace(
                names=["c^", "z", "y", "x"],
                units="nm",
                scales=(1, *daisy_array.voxel_size),
            )
            voxel_offset = (
                0,
                *((daisy_array.roi.offset / daisy_array.voxel_size)),
            )

            raw_vol = neuroglancer.LocalVolume(
                data=daisy_array.data,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )

            s.layers[dataset] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)
        except KeyError:
            pass


        care = f"{crop}_care"
        try:
            dataset = care
            daisy_array = open_ds(
                f"{train_config.validation_container}",
                validation_care_dataset.format(i=iteration, crop=crop),
            )

            dims = neuroglancer.CoordinateSpace(
                names=["c^", "z", "y", "x"],
                units="nm",
                scales=(1, *daisy_array.voxel_size),
            )
            voxel_offset = (
                0,
                *((daisy_array.roi.offset / daisy_array.voxel_size)),
            )

            raw_vol = neuroglancer.LocalVolume(
                data=daisy_array.data,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )

            s.layers[dataset] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)
        except KeyError:
            pass

        return (raw, care)

    with viewer.txn() as s:
        for dataset_yaml in data_config.datasets:
            raw, pred = add_layers(s, dataset_yaml.name[:-5])
            datasets.append((raw, pred))

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
            ]
        )

    # viewer.actions.add("switch_dataset", set_layout)

    # with viewer.config_state.txn() as s:
    #     s.input_event_bindings.data_view["keyt"] = "swith_dataset"

    print(viewer)

    input("Enter to quit!")
