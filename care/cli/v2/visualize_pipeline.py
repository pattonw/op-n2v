import click
import yaml


@click.command()
@click.option("-d", "--data-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--num-voxels", type=int, default=32)
@click.option("-c", "--context", type=int, default=13)
def visualize_pipeline(data_config, num_voxels, context):
    from care.pipeline import build_pipeline, get_request, split_batch
    from care.configs import DataConfig

    from funlib.geometry import Coordinate
    import gunpowder as gp

    import neuroglancer

    import numpy as np

    data_config = DataConfig(**yaml.safe_load(open(data_config, "r").read()))
    pipeline, resolutions = build_pipeline(data_config)

    volume_shape = Coordinate((num_voxels,) * 3)

    def load_batch(event):
        with gp.build(pipeline):
            input_size = gp.Coordinate((num_voxels,) * 3)
            model_context = gp.Coordinate((context,) * 3)
            batch_request = get_request(
                input_size,
                input_size - model_context * 2,
            )
            raw_input, raw_context = split_batch(
                pipeline.request_batch(batch_request),
            )

        with viewer.txn() as s:
            while len(s.layers) > 0:
                del s.layers[0]

            # reverse order for raw so we can set opacity to 1, this
            # way higher res raw replaces low res when available

            # RAW INPUT
            dims = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=raw_input.spec.voxel_size,
            )

            raw_vol = neuroglancer.LocalVolume(
                data=raw_input.data,
                voxel_offset=(
                    (-raw_input.spec.roi.shape / 2) / raw_input.spec.voxel_size
                ),
                dimensions=dims,
            )

            s.layers["input"] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)

            # RAW CONTEXT
            dims = neuroglancer.CoordinateSpace(
                names=["c^", "z", "y", "x"],
                units="nm",
                scales=(1,) + tuple(raw_context.spec.voxel_size),
            )

            raw_vol = neuroglancer.LocalVolume(
                data=raw_context.data,
                voxel_offset=(0,) + tuple(
                    (-raw_context.spec.roi.shape / 2) / raw_context.spec.voxel_size
                ),
                dimensions=dims,
            )

            s.layers["context"] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)

            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(layers=["input"]),
                        ]
                    ),
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(layers=["context"]),
                        ]
                    ),
                ]
            )

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    viewer.actions.add("load_batch", load_batch)

    with viewer.config_state.txn() as s:
        s.input_event_bindings.data_view["keyt"] = "load_batch"

    print(viewer)
    load_batch(None)

    input("Enter to quit!")
