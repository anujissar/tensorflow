
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colorsys
import math
import os
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

class CropToBoundingBoxTest(test_util.TensorFlowTestCase):

  def _CropToBoundingBox(self, x, offset_height, offset_width, target_height,
                         target_width, use_tensor_inputs):
    if use_tensor_inputs:
      offset_height = ops.convert_to_tensor(offset_height)
      offset_width = ops.convert_to_tensor(offset_width)
      target_height = ops.convert_to_tensor(target_height)
      target_width = ops.convert_to_tensor(target_width)
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      x_tensor = x
      feed_dict = {}

    y = image_ops.crop_to_bounding_box(x_tensor, offset_height, offset_width,
                                       target_height, target_width)
    if not use_tensor_inputs:
      self.assertTrue(y.get_shape().is_fully_defined())

    with self.test_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertReturns(self,
                     x,
                     x_shape,
                     offset_height,
                     offset_width,
                     y,
                     y_shape,
                     use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._CropToBoundingBox(x, offset_height, offset_width,
                                     target_height, target_width,
                                     use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertRaises(self,
                    x,
                    x_shape,
                    offset_height,
                    offset_width,
                    target_height,
                    target_width,
                    err_msg,
                    use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    x = np.array(x).reshape(x_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      try:
        self._CropToBoundingBox(x, offset_height, offset_width, target_height,
                                target_width, use_tensor_inputs)
      except Exception as e:
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError('Exception not raised: %s' % err_msg)

  def _assertShapeInference(self, pre_shape, height, width, post_shape):
    image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = Image_ops.crop_to_bounding_box(image, 0, 0, height, width)
    self.assertEqual(y.get_shape().as_list(), post_shape)

  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)
    self._assertReturns(x, x_shape, 0, 0, x, x_shape)

  def testCrop(self):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_shape = [3, 3, 1]

    offset_height, offset_width = [1, 0]
    y_shape = [2, 3, 1]
    y = [4, 5, 6, 7, 8, 9]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 1]
    y_shape = [3, 2, 1]
    y = [2, 3, 5, 6, 8, 9]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 0]
    y_shape = [2, 3, 1]
    y = [1, 2, 3, 4, 5, 6]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 0]
    y_shape = [3, 2, 1]
    y = [1, 2, 4, 5, 7, 8]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

  def testNon3DInput(self):
    # Input image is not 3D
    x = [0] * 15
    offset_height, offset_width = [0, 0]
    target_height, target_width = [2, 2]

    for x_shape in ([1, 3, 5, 1], [3, 5]):
      self._assertRaises(x, x_shape, offset_height, offset_width, target_height,
                         target_width, "'image' must be three-dimensional")

  def testZeroLengthInput(self):
    # Input image has 0-length dimension(s).
    # Each line is a test configuration:
    #   x_shape, target_height, target_width
    test_config = (([0, 2, 2], 1, 1), ([2, 0, 2], 1, 1), ([2, 2, 0], 1, 1),
                   ([0, 2, 2], 0, 1), ([2, 0, 2], 1, 0))
    offset_height, offset_width = [0, 0]
    x = []

    for x_shape, target_height, target_width in test_config:
      self._assertRaises(
          x,
          x_shape,
          offset_height,
          offset_width,
          target_height,
          target_width,
          "all dims of 'image.shape' must be > 0",
          use_tensor_inputs_options=[False])
      # Multiple assertion could fail, but the evaluation order is arbitrary.
      # Match gainst generic pattern.
      self._assertRaises(
          x,
          x_shape,
          offset_height,
          offset_width,
          target_height,
          target_width,
          'assertion failed:',
          use_tensor_inputs_options=[True])

#  def testBadParams(self):
#    x_shape = [4, 4, 1]
#    x = np.zeros(x_shape)
#
#    # Each line is a test configuration:
#    #   (offset_height, offset_width, target_height, target_width), err_msg
#    test_config = (([-1, 0, 3, 3], 'offset_height must be >= 0'),
#                   ([0, -1, 3, 3], 'offset_width must be >= 0'),
#                   ([0,  0, 0, 3], 'target_height must be > 0'),
#                   ([0,  0, 3, 0], 'target_width must be > 0'),
#                   ([2,  0, 3, 3], 'height must be >= target + offset'),
#                   ([0,  2, 3, 3], 'width must be >= target + offset'))
#
#    for params, err_msg in test_config:
#      self._assertRaises(x, x_shape, *params, err_msg=err_msg)
#


if __name__ == '__main__':
  googletest.main()
