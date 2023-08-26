from funlib.geometry import Coordinate
import gunpowder as gp

import numpy as np


def hist_eq(array, number_bins=256):
    # adapted from https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy

    # get array histogram
    eq_array = array + np.random.randn(*array.shape) * 1e-4
    image_histogram, bins = np.histogram(eq_array.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(array.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(array.shape)


class HistogramEqualization(gp.BatchFilter):
    def __init__(self, in_array: gp.ArrayKey, out_array: gp.ArrayKey):
        self.in_array = in_array
        self.out_array = out_array

    def setup(self):
        spec = self.spec[self.in_array].copy()
        if self.out_array == self.in_array:
            self.updates(self.out_array, spec)
        else:
            self.provides(self.out_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.in_array] = request[self.out_array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        in_array = batch[self.in_array]
        if in_array.data.ndim == in_array.spec.roi.dims + 1:
            for channel in range(in_array.data.shape[0]):
                in_array.data[channel] = hist_eq(in_array.data[channel], number_bins=256).astype(
                    in_array.spec.dtype
                )
        else:
            in_array.data = hist_eq(in_array.data).astype(in_array.spec.dtype)
        outputs[self.out_array] = in_array

        return outputs
