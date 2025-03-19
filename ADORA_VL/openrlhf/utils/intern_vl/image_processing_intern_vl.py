"""Image processor class for Intern-VL."""

import io
import torch
import numpy as np
import torchvision.transforms as T
from typing import Union, Optional
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
)
from transformers.image_utils import (
    ImageInput,
    VideoInput,
    make_list_of_images,
    to_numpy_array,
)
from transformers.utils import TensorType, is_vision_available, logging

logger = logging.get_logger(__name__)

if is_vision_available():
    from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

def transform_numpy(image, input_size, normalize_type):  
    """  
    使用 numpy 实现与 torchvision.transforms.Compose 相同的操作。  
    
    Args:  
        image (PIL.Image.Image): 输入的 PIL 图像。  
        input_size (int): 调整后的图像大小 (input_size, input_size)。  
        mean (list): 每个通道的均值，用于标准化。  
        std (list): 每个通道的标准差，用于标准化。  
    
    Returns:  
        numpy.ndarray: 标准化后的图像数组，形状为 (C, H, W)。  
    """  
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    # 1. 转换为 RGB 模式  
    if image.mode != 'RGB':  
        image = image.convert('RGB')  
    
    # 2. 调整大小  
    image = image.resize((input_size, input_size), resample=Image.BICUBIC)  
    
    # 3. 转换为 NumPy 数组并归一化到 [0, 1]  
    image_array = np.array(image).astype(np.float32) / 255.0  # Shape: (H, W, C)  
    
    # 4. 转换为 (C, H, W) 格式  
    image_array = np.transpose(image_array, (2, 0, 1))  # Shape: (C, H, W)  
    
    # 5. 标准化  
    mean = np.array(MEAN).reshape(-1, 1, 1)  # Reshape to (C, 1, 1)  
    std = np.array(STD).reshape(-1, 1, 1)    # Reshape to (C, 1, 1)  
    image_array = (image_array - mean) / std  
    
    return image_array  

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVLImageProcessor(BaseImageProcessor):
    r"""
    """

    model_input_names = ["pixel_values", "num_tiles"]

    def __init__(
        self,
        is_train: bool = True,
        image_size: int = None,
        pad2square: bool = False,
        normalize_type: str = 'imagenet',
        dynamic_image_size: bool = True,
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 12,
        use_thumbnail: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.is_train = is_train
        self.image_size = image_size
        self.pad2square = pad2square
        self.normalize_type = normalize_type
        self.dynamic_image_size = dynamic_image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
    
    def preprocess(
        self,
        images: Union[ImageInput,],
        videos: VideoInput = None,
        is_train: bool = None,
        image_size: int = None,
        pad2square: bool = None,
        normalize_type: str = None,
        dynamic_image_size: bool = None,
        min_dynamic_patch: int = None,
        max_dynamic_patch: int = None,
        use_thumbnail: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        """
        is_train = is_train if is_train is not None else self.is_train
        image_size = image_size if image_size is not None else self.image_size
        pad2square = pad2square if pad2square is not None else self.pad2square
        normalize_type = normalize_type if normalize_type is not None else self.normalize_type
        dynamic_image_size = dynamic_image_size if dynamic_image_size is not None else self.dynamic_image_size
        min_dynamic_patch = min_dynamic_patch if min_dynamic_patch is not None else self.min_dynamic_patch
        max_dynamic_patch = max_dynamic_patch if max_dynamic_patch is not None else self.max_dynamic_patch
        image_size = image_size if image_size is not None else self.image_size
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail

        images = make_list_of_images(images)

        images_list, num_tiles = [], []
        num_image = len(images)

        for image in images:
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                        #    max_num=max(1, self.max_dynamic_patch // num_image),
                                           max_num=self.max_dynamic_patch,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images_list += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images_list.append(image)
                num_tiles.append(1)
        
        pixel_values = np.array([transform_numpy(image, self.image_size, self.normalize_type) for image in images_list])
        data = {"pixel_values": pixel_values, "num_tiles": num_tiles, "image_flags": torch.tensor([1] * pixel_values.shape[0], dtype=torch.long)}
        return BatchFeature(data=data, tensor_type=return_tensors)
