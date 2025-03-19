import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from PIL import Image

from .experience_maker_vl import ExperienceVL


@dataclass
class BufferItemVL:
    """BufferItemVL is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    pixel_values: (B*H, W)
    image_grid_thws: (B, 3)
    raw_images: Optional[List[Image.Image]]  # raw images before processing
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor

    pixel_values: Optional[torch.Tensor] = None # image pixel processed by HF processor
    image_grid_thws: Optional[torch.Tensor] = None # image grid thw 
    raw_images: Optional[List[Image.Image]] = None  # raw images before processing
    pixel_values_intern: Optional[torch.Tensor] = None # InternVL image_info
    image_flags: Optional[torch.Tensor] = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None


def split_experience_batch(experience: ExperienceVL) -> List[BufferItemVL]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        # "pixel_values", 
        "image_grid_thws", # 3
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    # image data split
    if experience.pixel_values is not None:
        pixel_values = experience.pixel_values
        if isinstance(pixel_values, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                batch_kwargs[i]["pixel_values"] = pixel_values[index: index + torch.prod(batch_kwargs[i]["image_grid_thws"])]
                index += torch.prod(batch_kwargs[i]["image_grid_thws"])
        # TODO pixel_values is not torch.Tensor
    if experience.pixel_values_intern is not None:
        pixel_values_intern = experience.pixel_values_intern
        if isinstance(pixel_values_intern, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                batch_kwargs[i]["pixel_values_intern"] = pixel_values_intern[index: index + torch.prod(batch_kwargs[i]["image_grid_thws"])]
                index += torch.prod(batch_kwargs[i]["image_grid_thws"])
        # TODO pixel_values_intern is not torch.Tensor
    if experience.image_flags is not None:
        image_flags = experience.image_flags
        if isinstance(image_flags, torch.Tensor):
            index = 0
            for i in range(len(batch_kwargs)):
                batch_kwargs[i]["image_flags"] = image_flags[index: index + torch.prod(batch_kwargs[i]["image_grid_thws"])]
                index += torch.prod(batch_kwargs[i]["image_grid_thws"])

    # raw images split
    if experience.raw_images is not None:
        for i in range(len(batch_kwargs)):
            batch_kwargs[i]["raw_images"] = experience.raw_images[i]

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItemVL(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItemVL], packing_samples=False) -> ExperienceVL:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    # image data
    pixel_values_list = [item.pixel_values for item in items]
    kwargs["pixel_values"] = torch.cat(pixel_values_list, dim=0) if pixel_values_list[0] is not None else None
    image_grid_thws_list = [item.image_grid_thws for item in items]
    kwargs["image_grid_thws"] = torch.stack(image_grid_thws_list, dim=0) if image_grid_thws_list[0] is not None else None
    raw_images_list = [item.raw_images for item in items]
    kwargs["raw_images"] = raw_images_list if raw_images_list[0] is not None else None
    pixel_values_intern_list = [item.pixel_values_intern for item in items]
    kwargs["pixel_values_intern"] = torch.cat(pixel_values_intern_list, dim=0) if pixel_values_intern_list[0] is not None else None
    image_flags_list = [item.image_flags for item in items]
    kwargs["image_flags"] = torch.cat(image_flags_list, dim=0) if image_flags_list[0] is not None else None

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return ExperienceVL(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBufferVL(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling. train_micro_batchsize
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItemVL] = []

    @torch.no_grad()
    def append(self, experience: ExperienceVL) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> ExperienceVL:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItemVL:
        return self.items[idx]

    def collate_fn(self, batch) -> ExperienceVL:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
