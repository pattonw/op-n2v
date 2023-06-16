from multiprocessing import context
from funlib.geometry import Coordinate
import gunpowder as gp

import numpy as np


class Context(gp.BatchFilter):
    def __init__(self, in_array, out_array, neighborhood):
        self.in_array = in_array
        self.out_array = out_array
        self.neighborhood = neighborhood
        self.context_pos = Coordinate(
            [max([n[i] for n in neighborhood]) for i in range(len(neighborhood[0]))]
        )
        self.context_neg = Coordinate(
            [min([n[i] for n in neighborhood]) for i in range(len(neighborhood[0]))]
        )

    def setup(self):

        context_spec = self.spec[self.in_array].copy()
        vs = self.spec[self.in_array].voxel_size
        context_spec.roi = context_spec.roi.grow(
            -abs(self.context_neg) * vs, -self.context_pos * vs
        )
        self.provides(self.out_array, context_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        context_spec = request[self.out_array].copy()
        vs = self.spec[self.in_array].voxel_size
        context_spec.roi = context_spec.roi.grow(
            abs(self.context_neg) * vs, self.context_pos * vs
        )
        deps[self.in_array] = context_spec
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        out_roi = request[self.out_array].roi
        in_array = batch[self.in_array]
        vs = self.spec[self.in_array].voxel_size
        context_data = np.stack(
            [in_array.crop(out_roi + n * vs).data for n in self.neighborhood], axis=0
        )
        out_spec = in_array.spec.copy()
        out_spec.roi = out_roi
        outputs[self.out_array] = gp.Array(context_data, out_spec)

        return outputs


class DeContext(gp.BatchFilter):
    def __init__(self, in_array, out_array, neighborhood):
        self.in_array = in_array
        self.out_array = out_array
        self.neighborhood = neighborhood
        self.context_pos = Coordinate(
            [max([n[i] for n in neighborhood]) for i in range(len(neighborhood[0]))]
        )
        self.context_neg = Coordinate(
            [min([n[i] for n in neighborhood]) for i in range(len(neighborhood[0]))]
        )

    def setup(self):

        context_spec = self.spec[self.in_array].copy()
        vs = self.spec[self.in_array].voxel_size
        context_spec.roi = context_spec.roi.grow(
            self.context_neg * vs, -self.context_pos * vs
        )
        self.provides(self.out_array, context_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        context_spec = request[self.out_array].copy()
        vs = self.spec[self.in_array].voxel_size
        context_spec.roi = context_spec.roi.grow(
            -self.context_neg * vs, self.context_pos * vs
        )
        deps[self.in_array] = context_spec
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        out_roi = request[self.out_array].roi
        in_array = batch[self.in_array]
        vs = self.spec[self.in_array].voxel_size
        care_data = np.mean(
            [
                in_array.crop(out_roi - n * vs).data[i]
                for i, n in enumerate(self.neighborhood)
            ],
            axis=0,
        )
        out_spec = in_array.spec.copy()
        out_spec.roi = out_roi
        outputs[self.out_array] = gp.Array(care_data, out_spec)

        return outputs
