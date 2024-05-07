CROP_PROPORTION = 0.875  # Standard for ImageNet.

# This file is adapted from SimCLRv2.
# https://github.com/google-research/simclr/blob/master/data_util.py

import numpy as np
import torch
import random
from PIL import Image, ImageFilter
from torchvision import transforms

random_flip = transforms.RandomHorizontalFlip()
to_tensor = transforms.ToTensor()

def center_crop(image, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.
  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.
  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = np.shape(np.asarray(image))
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  # image = tf.image.crop_to_bounding_box(
  #     image, offset_height, offset_width, crop_height, crop_width)
  l = offset_width
  t = offset_height
  r = l + crop_width
  b = t + crop_height
  image = image.crop((l, t, r, b))
  # image = tf.image.resize_bicubic([image], [height, width])[0]
  image = image.resize((width, height), Image.BICUBIC)

  return image

def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""

  if np.random.uniform() < p:
    return func(x)
  return x

  # return tf.cond(
  #     tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
  #             tf.cast(p, tf.float32)),
  #     lambda: func(x),
  #     lambda: x)


def crop_and_resize(image, height, width):
  """Make a random crop and resize it to height `height` and width `width`.
  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.
  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
  raise NotImplementedError()

def random_color_jitter(image, p=1.0, impl='simclrv2'):
  raise NotImplementedError()

def random_crop_with_resize(image, height, width, p=1.0):
  """Randomly crop and resize an image.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: Probability of applying this transformation.
  Returns:
    A preprocessed image `Tensor`.
  """
  def _transform(image):  # pylint: disable=missing-docstring
    image = crop_and_resize(image, height, width)
    return image
  return random_apply(_transform, p=p, x=image)

def preprocess_for_train(image,
                         height,
                         width,
                         color_distort=True,
                         crop=True,
                         flip=True,
                         impl='simclrv2'):
  """Preprocesses the given image for training.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    color_distort: Whether to apply the color distortion.
    crop: Whether to crop the image.
    flip: Whether or not to flip left and right of an image.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.
  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = random_crop_with_resize(image, height, width)
  if flip:
    image = random_flip(image)
    #image = tf.image.random_flip_left_right(image)
  if color_distort:
    image = random_color_jitter(image, impl=impl)
  # image = tf.reshape(image, [height, width, 3])
  # image = tf.clip_by_value(image, 0., 1.)
  image = np.clip(image, 0., 1.)
  image = to_tensor(image)
  return image

def preprocess_for_eval(image, height=224, width=224, crop=True):
  """Preprocesses the given image for evaluation.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.
  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  # image = np.reshape(image, [height, width, 3])
  # image = np.clip(image, 0., 1.)
  image = to_tensor(image)
  return image

def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion):
    """Compute aspect ratio-preserving shape for central crop.
    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.
    Args:
        image_height: Height of image to be cropped.
        image_width: Width of image to be cropped.
        aspect_ratio: Desired aspect ratio (width / height) of output.
        crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
        crop_height: Height of image after cropping.
        crop_width: Width of image after cropping.
    """
    image_width_float = image_width
    image_height_float = image_height

    def _requested_aspect_ratio_wider_than_image():
        crop_height = int(np.round(
            crop_proportion / aspect_ratio * image_width_float))
        crop_width = int(np.round(
            crop_proportion * image_width_float))
        return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
        crop_height = int(
            np.round(crop_proportion * image_height_float))
        crop_width = int(np.round(
            crop_proportion * aspect_ratio *
            image_height_float))
        return crop_height, crop_width

    if aspect_ratio > image_width_float / image_height_float:
        return _requested_aspect_ratio_wider_than_image()
    else:
        return _image_wider_than_requested_aspect_ratio()

if False:
  class GaussianBlur(object):
      """
          PyTorch version of
          https://github.com/google-research/simclr/blob/244e7128004c5fd3c7805cf3135c79baa6c3bb96/data_util.py#L311
      """
      def gaussian_blur(self, image, sigma):
          image = image.reshape(1, 3, 224, 224)
          radius = np.int(self.kernel_size/2)
          kernel_size = radius * 2 + 1
          x = np.arange(-radius, radius + 1)

          blur_filter = np.exp(
                -np.power(x, 2.0) / (2.0 * np.power(np.float(sigma), 2.0)))
          blur_filter /= np.sum(blur_filter)

          conv1 = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), groups=3, padding=[kernel_size//2, 0], bias=False)
          conv1.weight = torch.nn.Parameter(
              torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 0, 1])))

          conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), groups=3, padding=[0, kernel_size//2], bias=False)
          conv2.weight = torch.nn.Parameter(
              torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 1, 0])))

          res = conv2(conv1(image))
          assert res.shape == image.shape
          return res[0]

      def __init__(self, kernel_size, p=0.5):
          self.kernel_size = kernel_size
          self.p = p

      def __call__(self, img):
          with torch.no_grad():
              assert isinstance(img, torch.Tensor)
              if np.random.uniform() < self.p:
                  return self.gaussian_blur(img, sigma=np.random.uniform(0.2, 2))
              return img

      def __repr__(self):
          return self.__class__.__name__ + '(kernel_size={0}, p={1})'.format(self.kernel_size, self.p)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        # given from https://arxiv.org/pdf/2002.05709.pdf
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

class Clip(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)
