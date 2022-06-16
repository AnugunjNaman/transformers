# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature extractor class for Omnivore."""

from typing import BinaryIO, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.transforms._transforms_video import NormalizeVideo

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ImageFeatureExtractionMixin,
    ImageInput,
    is_torch_tensor,
)
from ...utils import TensorType, logging


try:
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
except ModuleNotFoundError:
    raise ModuleNotFoundError("pytorchvideo is missing. please install pytorchvideo")

logger = logging.get_logger(__name__)


class DepthNorm(nn.Module):
    """
    Normalize the depth channel: in an RGBD input of shape (4, H, W), only the last channel is modified. The depth
    channel is also clamped at 0.0. The Midas depth prediction model outputs inverse depth maps - negative values
    correspond to distances far away so can be clamped at 0.0
    """

    def __init__(
        self,
        max_depth: float,
        clamp_max_before_scale: bool = False,
        min_depth: float = 0.01,
    ):
        """
        Args:
            max_depth (`float`): The max value of depth for the dataset
            clamp_max (`bool`): Whether to clamp to max_depth or to divide by max_depth
        """
        super().__init__()
        if max_depth < 0.0:
            raise ValueError("max_depth must be > 0; got %.2f" % max_depth)
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def forward(self, image: torch.Tensor):
        channels, height, width = image.shape
        if channels != 4:
            err_msg = f"This transform is for 4 channel RGBD input only; got {image.shape}"
            raise ValueError(err_msg)
        color_img = image[:3, ...]  # (3, H, W)
        depth_img = image[3:4, ...]  # (1, H, W)

        # Clamp to 0.0 to prevent negative depth values
        depth_img = depth_img.clamp(min=self.min_depth)

        # divide by max_depth
        if self.clamp_max_before_scale:
            depth_img = depth_img.clamp(max=self.max_depth)

        depth_img /= self.max_depth

        img = torch.cat([color_img, depth_img], dim=0)
        return img


class TemporalCrop(nn.Module):
    """
    Convert the video into smaller clips temporally.
    """

    def __init__(self, frames_per_clip: int = 8, stride: int = 8):
        super().__init__()
        self.frames = frames_per_clip
        self.stride = stride

    def forward(self, video):
        assert video.ndim == 4, "Must be (C, T, H, W)"
        res = []
        for start in range(0, video.size(1) - self.frames + 1, self.stride):
            res.append(video[:, start : start + self.frames, ...])
        return res


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with -2 in the spatial crop at the slowfast
        augmentation stage (so full frames are passed in here). Will return a larger list with the 3x spatial crops as
        well. It's useful for 3x4 testing (eg in SwinT) or 3x10 testing in SlowFast etc.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
        elif num_crops == 1:
            self.crops_to_ext = [1]
        else:
            raise NotImplementedError("Nothing else supported yet, slowfast only takes 0, 1, 2 as arguments")

    def forward(self, videos: Sequence[torch.Tensor]):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
        return res


def crop_boxes(boxes, x_offset, y_offset):
    """
    Args:
    Peform crop on the bounding boxes given the offsets.
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis. y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Args:
    Perform uniform spatial sampling on the images and corresponding boxes.
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images. spatial_idx (int): 0, 1, or 2 for left, center, and
        right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class OmnivoreFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a Omnivore feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the shortest edge of the input to int(256/224 *`size`).
        size (`int` or `Tuple(int)`, *optional*, defaults to 224):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then shorter side of input will be resized to 'size'.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether or not to center crop the input to `size`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BICUBIC,
        do_center_crop=True,
        do_normalize=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
        num_frames=160,
        sampling_rate=2,
        frames_per_second=30,
        frames_per_clip=32,
        video_stride=40,
        video_crop_size=224,
        num_of_crops_in_video=3,
        max_depth=75.0,
        clamp_max_before_scale=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.num_frames = num_frames
        self.sampling_rate = (sampling_rate,)
        self.frames_per_second = frames_per_second
        self.frames_per_clip = frames_per_clip
        self.video_stride = video_stride
        self.video_crop_size = video_crop_size
        self.num_of_crops_in_video = num_of_crops_in_video
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale

    def __call__(self, inputs, input_type, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs):
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
        """
        # Input type checking for clearer error
        if input_type == "image":
            return self.preprocess_images(inputs, return_tensors)
        elif input_type == "video":
            return self.preprocess_videos(inputs, return_tensors)
        elif input_type == "rgbd":
            return self.preprocess_rgbd(inputs["images"], inputs["depths"], return_tensors)
        else:
            raise ValueError("""Input type not supported> Support inputs are "images", "videos" and "rgbd"""")

    def preprocess_images(self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs):
        valid_images = False
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None:
            size_ = int((256 / 224) * self.size)
            images = [self.resize(image=image, size=size_, resample=self.resample) for image in images]
        if self.do_center_crop:
            images = [self.center_crop(image=image, size=self.size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        input_type = ["image" for _ in images]
        # return as BatchFeature
        data = {"pixel_values": images, "pixel_input_type": input_type}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        return encoded_inputs

    def preprocess_videos(self, videos, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs):
        if isinstance(videos, (np.ndarray)) or is_torch_tensor(videos):
            video = self.transform_videos()(videos)
            video = video["video"][None, ...]
            input_type = "video"
            data = {"pixel_values": [video], "pixel_input_type": [input_type]}
            encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
            return encoded_inputs
        elif isinstance(videos, (list, tuple)):
            if len(videos) == 0 or isinstance(videos[0], (np.ndarray)) or is_torch_tensor(videos[0]):
                videos = [self.transform_videos()(video) for video in videos]
                videos = [video["video"] for video in videos]
                videos = [video[0][None, ...] for video in videos]
                input_type = ["video" for _ in videos]
                data = {"pixel_values": videos, "pixel_input_type": input_type}
                encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
                return encoded_inputs

    def transform_videos(self):
        video_transform = ApplyTransformToKey(
            key="video",
            transform=T.Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    T.Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=224),
                    NormalizeVideo(mean=self.image_mean, std=self.image_std),
                    TemporalCrop(frames_per_clip=self.frames_per_clip, stride=self.frames_per_clip),
                    SpatialCrop(crop_size=self.video_crop_size, num_crops=self.num_of_crops_in_video),
                ]
            ),
        )

        return video_transform

    def preprocess_rgbd(
        self, rgbd_images, rgbd_depths, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
    ):
        valid_images, valid_depths = False, False
        if isinstance(rgbd_images, (Image.Image, np.ndarray)) or is_torch_tensor(rgbd_images):
            valid_images = True
        elif isinstance(rgbd_images, (list, tuple)):
            if (
                len(rgbd_images) == 0
                or isinstance(rgbd_images[0], (Image.Image, np.ndarray))
                or is_torch_tensor(rgbd_images[0])
            ):
                valid_images = True

        if is_torch_tensor(rgbd_depths):
            valid_depths = True
        elif isinstance(rgbd_depths, (list, tuple)):
            if len(rgbd_depths) == 0 or is_torch_tensor(rgbd_depths[0]):
                valid_depths = True

        if not valid_images or not valid_depths:
            raise ValueError(
                """
                RGBD Image or Depth File not in supported format, Images must of type `PIL.Image.Image`, `np.ndarray`
                or `torch.Tensor` (single example), `List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]`
                (batch of examples).
                
                Depths must have `torch.Tensor`, `np.ndarry` (single example), `List[np.ndarray]` or
                `List[torch.Tensor]` (batch of examples)
                """
            )

        is_batched_image = bool(
            isinstance(rgbd_images, (list, tuple))
            and (isinstance(rgbd_images[0], (Image.Image, np.ndarray)) or is_torch_tensor(rgbd_images[0]))
        )
        is_batched_depth = bool(isinstance(rgbd_images, (list, tuple)) and (is_torch_tensor(rgbd_depths[0])))

        if not is_batched_image:
            if not is_batched_depth:
                rgbd_images = [rgbd_images]
                rgbd_depths = [rgbd_depths]
            else:
                logger.error("There is a mis-match in number of images and depths")

        rgbd_images = [self.convert_rgb(rgbd_image) for rgbd_image in rgbd_images]
        rgbd_images = [T.ToTensor(rgbd_image) for rgbd_image in rgbd_images]
        rgbds = [torch.cat([rgbd_images[i], rgbd_depths[i]], dim=0) for i in range(len(rgbd_images))]
        depth_norm = DepthNorm(max_depth=self.max_depth, clamp_max_before_scale=self.clamp_max_before_scale)
        rgbds = [depth_norm(rgbd) for rgbd in rgbds]
        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None:
            rgbds = [self.resize(image=rgbd, size=self.size, resample=self.resample) for rgbd in rgbds]
        if self.do_center_crop:
            rgbds = [self.center_crop(image=rgbd, size=self.size) for rgbd in rgbds]
        if self.do_normalize:
            rgbds = [self.normalize(image=rgbd, mean=self.image_mean, std=self.image_std) for rgbd in rgbds]

        input_type = ["rgbd" for _ in rgbds]
        # return as BatchFeature
        data = {"pixel_values": rgbds, "pixel_input_type": input_type}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        return encoded_inputs
