import logging
import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.array import ArrayKey
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.batch_request import BatchRequest

logger = logging.getLogger(__name__)

class Pad(BatchFilter):
    '''Add a constant intensity padding around arrays of another batch 
    provider. This is useful if your requested batches can be larger than what 
    your source provides.

    Args:

        keys (:class:`ArrayKeys` or :class:`GraphKeys`):

            The array or points set to pad.

        size (:class:`Coordinate` or ``None``):

            The padding to be added. If None, an infinite padding is added. If
            a coordinate, this amount will be added to the ROI in the positive
            and negative direction.

        value (scalar or ``None``):

            The value to report inside the padding. If not given, 0 is used.
            Only used for :class:`Array<Arrays>`.
    '''

    def __init__(self, keys, size, value=None):

        self.keys = keys
        self.size = size
        self.value = value

    def setup(self):
        self.enable_autoskip()

        assert all(key in self.spec for key in self.keys), (
            "Asked to pad %s, but is not provided upstream."%self.keys)
        assert all(self.spec[key].roi is not None for key in self.keys), (
            "Asked to pad %s, but upstream provider doesn't have a ROI for "
            "it."%self.keys)

        for key in self.keys:
            spec = self.spec[key].copy()
            if self.size is not None:
                spec.roi = spec.roi.grow(self.size, self.size)
            else:
                spec.roi.shape = Coordinate((None,) * spec.roi.dims)
            self.updates(key, spec)

    def prepare(self, request):

        upstream_spec = self.get_upstream_provider().spec

        logger.debug("request: %s"%request)
        logger.debug("upstream spec: %s"%upstream_spec)

        deps = BatchRequest()
        for key in self.keys:
            roi = request[key].roi.copy()

            # change request to fit into upstream spec
            request[key].roi = roi.intersect(upstream_spec[key].roi)

            if request[key].roi.empty:

                logger.warning(
                    "Requested %s ROI %s lies entirely outside of upstream "
                    "ROI %s.", key, roi, upstream_spec[key].roi)

                # ensure a valid request by asking for empty ROI
                request[key].roi = Roi(
                        upstream_spec[key].roi.offset,
                        (0,)*upstream_spec[key].roi.dims
                )

            logger.debug("new request: %s"%request)

            deps[key] = request[key]
        return deps

    def process(self, batch, request):

        for key in self.keys:
            # restore requested batch size and ROI
            if isinstance(key, ArrayKey):

                array = batch.arrays[key]
                array.data = self.__expand(
                        array.data,
                        array.spec.roi/array.spec.voxel_size,
                        request[key].roi/array.spec.voxel_size,
                        self.value if self.value else 0
                )
                array.spec.roi = request[key].roi

            else:

                points = batch.graphs[key]
                points.spec.roi = request[key].roi

    def __expand(self, a, from_roi, to_roi, value):
        '''from_roi and to_roi should be in voxels.'''

        logger.debug(
            "expanding array of shape %s from %s to %s",
            str(a.shape), from_roi, to_roi)

        num_channels = len(a.shape) - from_roi.dims
        channel_shapes = a.shape[:num_channels]

        b = np.zeros(channel_shapes + to_roi.shape, dtype=a.dtype)
        if value != 0:
            b[:] = value

        shift = -to_roi.offset
        logger.debug("shifting 'from' by " + str(shift))
        a_in_b = from_roi.shift(shift).to_slices()

        logger.debug("target shape is " + str(b.shape))
        logger.debug("target slice is " + str(a_in_b))

        b[(slice(None),)*num_channels + a_in_b] = a

        return b
