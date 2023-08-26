import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
def view_snapshots(train_config):
    from care.configs import TrainConfig

    import daisy

    from funlib.persistence import open_ds, Array

    import neuroglancer
    import yaml
    import numpy as np

    from itertools import chain

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    raw_datasets = [
        "raw",
        "raw_context",
        "pred",
    ]

    with viewer.txn() as s:
        while len(s.layers) > 0:
            del s.layers[0]
        for dataset in raw_datasets:
            daisy_array = open_ds(
                f"{train_config.snapshot_container}",
                f"{dataset}",
            )
            ndims = len(daisy_array.data.shape)
            if ndims == 4:
                axis_names = ["iterations", "z", "y", "x"]
            elif ndims == 5:
                axis_names = ["iterations", "c^", "z", "y", "x"]

            data = daisy_array.data

            dims = neuroglancer.CoordinateSpace(
                names=axis_names,
                units="nm",
                scales=(1,) * (ndims - 3) + tuple(daisy_array.voxel_size),
            )
            voxel_offset = (0,) * (ndims - 3) + tuple(
                daisy_array.roi.offset / daisy_array.voxel_size
            )

            vol = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )

            s.layers[dataset] = neuroglancer.ImageLayer(source=vol, opacity=1.0)

        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(layers=[raw_datasets[1]]),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(layers=[raw_datasets[2]]),
                    ]
                ),
            ]
        )

    print(viewer)

    input("Enter to quit!")
