import gunpowder as gp
import tifffile

class TiffStackSource(gp.BatchProvider):
    def __init__(self, array_key, path):
        self.array_key = array_key
        self.path = path

    def setup(self):
        data = tifffile.imread(self.path)
        sample, channel, x_axis, y_axis = data.shape
        data = data.reshape(sample * channel, x_axis, y_axis)
        array_spec = gp.ArraySpec(gp.Roi((0, 0), (x_axis, y_axis)), dtype=data.dtype, voxel_size=(1, 1))
        self.array = gp.Array(data, array_spec)

        self.provides(self.array_key, array_spec)

    def provide(self, request):
        batch = gp.Batch()
        batch[self.array_key] = self.array.crop(request[self.array_key].roi)
        return batch
