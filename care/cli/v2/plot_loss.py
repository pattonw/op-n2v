import click

@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--num-iter", type=int, default=1000000000)
@click.option("-o", "--output", type=click.Path(exists=False))
@click.option("-s", "--smooth", type=float, default=0)
def plot_loss(train_config, num_iter, output, smooth):
    from care.configs import TrainConfig

    import numpy as np
    import matplotlib.pyplot as plt

    import yaml

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