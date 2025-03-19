from typing import Callable

import math
from io import BytesIO
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import exist_and_not_none, zero_pad_sequences

from PIL import Image
from PIL.Image import Image as ImageObject


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    images_key="images",
    apply_chat_template=None,
    is_dpo=False,
    processor=None
) -> str:

        
    if apply_chat_template:
        if prompt_key:
            conversation = data[prompt_key]  
            # 对conversation进行解析  
            conversation_item = conversation[0]  
            user_text = conversation_item["value"].split("<image>")[-1]

            _prompt = [  
                {  
                    "role": "user",  
                    "content": [  
                        {  
                            "type": "image",  
                        },  
                        {  
                            "type": "text",  
                            "text": user_text,  
                        },  
                    ],  
                }, 
            ]
            _chosen = [  
                {  
                    "role": "assistant",  
                    "content": [  
                        {  
                            "type": "text",  
                            "text": data[chosen_key]["value"],  
                        },  
                    ],  
                }, 
            ]
            _rejected = [  
                {  
                    "role": "assistant",  
                    "content": [  
                        {  
                            "type": "text",  
                            "text": data[rejected_key]["value"],  
                        },  
                    ],  
                }, 
            ]
            
            prompt = apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(_prompt + _chosen, tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(_prompt  + _rejected, tokenize=False)[len(prompt) :]
    # TODO
    #     else:
    #         prompt = ""
    #         chosen = apply_chat_template([data[chosen_key]], tokenize=False)
    #         rejected = apply_chat_template([data[rejected_key]], tokenize=False)

    #         if is_dpo:
    #             prompt = apply_chat_template([data[chosen_key]][:-1], tokenize=False, add_generation_prompt=True)
    #             chosen = chosen[len(prompt) :]
    #             rejected = rejected[len(prompt) :]
    # else:
    #     if prompt_key:
    #         prompt = data[prompt_key]
    #         if input_template:
    #             prompt = input_template.format(prompt)
    #     else:
    #         prompt = ""
    #     chosen = data[chosen_key]
    #     rejected = data[rejected_key]

    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, data[images_key], margin


class RewardDatasetVL(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        processor: Callable, 
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.processor = processor
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.images_key = getattr(self.strategy.args, "images_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            strategy.print(self.apply_chat_template)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image"
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }, 
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "A young man standing on stage wearing a white shirt and black pants."},
                    ],
                }
            ]
            strategy.print(self.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            # strategy.print(tokenizer.decode(self.apply_chat_template(example), skip_special_tokens=False))
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.images = processed_dataset["images"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, reject, images, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.images_key,
            self.apply_chat_template,
            self.is_dpo,
            self.processor
        )

        # if self.is_dpo:
        prompt_token = self.processor(
            images = images,
            text = prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        # print(prompt_token)
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        # Filter the sample whose length is greater than max_length (2 for answer length)
        if prompt_ids_len >= self.max_length - 2:
            prompt = None
            self.strategy.print("Warning: prompt_ids_len >= self.max_length - 2 !!!!", prompt_ids_len)

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "images": images,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, image, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.images[idx], self.extras[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.processor(
            images=image,
            text=chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.processor(
            images=image,
            text=reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True
        if "image_flags" not in chosen_token:
            return (
                chosen_token["input_ids"],
                chosen_token["attention_mask"],
                reject_token["input_ids"],
                reject_token["attention_mask"],
                chosen_token["pixel_values"],
                chosen_token["image_grid_thw"],
                extra,
            )
        else:
            return (
                chosen_token["input_ids"],
                chosen_token["attention_mask"],
                reject_token["input_ids"],
                reject_token["attention_mask"],
                chosen_token["pixel_values"],
                chosen_token["image_flags"],
                extra,
            )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        pixel_values = []
        image_grid_thws = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, pixel_value, image_grid_thw, extra in item_list:
            
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            pixel_values.append(pixel_value)
            image_grid_thws.append(image_grid_thw)
            extras.append(extra)
        # self.strategy.print("chosen_id: ", chosen_id.shape)
        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, torch.cat(pixel_values, dim=0), torch.cat(image_grid_thws, dim=0), extras

# TODO 
    def packing_collate_fn(self, item_list):
        extras = []
        pixel_values = []
        image_grid_thws = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, pixel_value, image_grid_thw, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))

            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

            pixel_values.append(pixel_value)
            image_grid_thws.append(image_grid_thw)

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, torch.cat(pixel_values, dim=0), torch.cat(image_grid_thws, dim=0), extras
