from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import zero_pad_sequences


def preprocess_data(data, input_template=None, input_key="input", output_key=None, images_key="images", apply_chat_template=None):
    if apply_chat_template:
        if output_key:
            conversation = data[input_key]  
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
            _response = [  
                {  
                    "role": "assistant",  
                    "content": [  
                        {  
                            "type": "text",  
                            "text": data[output_key]["value"],  
                        },  
                    ],  
                }, 
            ]

            prompt = apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(_prompt + _response, tokenize=False)[len(prompt) :]
    # TODO
    #     else:
    #         prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
    #         response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    # else:
    #     prompt = data[input_key]
    #     if input_template:
    #         prompt = input_template.format(prompt)
    #     # output_key is None for continue pretrain
    #     response = data[output_key] if output_key else ""
    return prompt, response, data[images_key]


class SFTDatasetVL(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        processor: Callable, 
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiple_of=1,
        multiturn=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.multiturn = multiturn

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.images_key = getattr(self.strategy.args, "images_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
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
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.images = processed_dataset["images"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

    def process_data(self, data):
        #TODO support VLM multiturn
        # if self.multiturn and self.output_key:
        #     data[self.input_key].append(data[self.output_key])
        #     data[self.output_key] = None
        
        # if self.multiturn:
        #     assert not self.output_key or not data[self.output_key], "You should put the whole trajactory into data[input_key] and do not set output_key"
        #     input_key = self.input_key
        #     apply_chat_template = self.apply_chat_template
        #     response_ranges = []
        #     for idx, message in enumerate(data[input_key]):
        #         if message['role'] == 'assistant':
        #             prompt = apply_chat_template(data[input_key][: idx], tokenize=False, add_generation_prompt=True)
        #             response = apply_chat_template(data[input_key][: idx + 1], tokenize=False)[len(prompt):]

        #             start_idx = self.tokenizer(
        #                 prompt,
        #                 max_length=self.max_length,
        #                 padding=False,
        #                 truncation=True,
        #                 return_tensors="pt",
        #                 add_special_tokens=False,
        #             )["attention_mask"].int().sum().item()
                    
        #             end_idx = start_idx + self.tokenizer(
        #                 response,
        #                 max_length=self.max_length,
        #                 padding=False,
        #                 truncation=True,
        #                 return_tensors="pt",
        #                 add_special_tokens=False,
        #             )["attention_mask"].int().sum().item() - 1
        #             response_ranges.append((start_idx, end_idx)) # left close right open
                    
        prompt, response, images = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            self.images_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )

        if not self.pretrain_mode:
            prompt_token = self.processor(
                images = images,
                text = prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {"prompt": prompt, 
                "response": response, 
                "images": images,
                "prompt_ids_len": prompt_ids_len}

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]
        image = self.images[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            text = prompt

        input_token = self.processor(
            images=image,
            text=text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            input_token["attention_mask"][0][-1] = True
        info = {"input": prompt, "output": response, "input_length": input_token["attention_mask"].int().sum().item(), "image": image}

        return (
            prompt_ids_len, 
            input_token["input_ids"], 
            input_token["attention_mask"],
            input_token["pixel_values"],
            input_token["image_grid_thw"],
            info
        )

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        pixel_values = []
        image_grid_thws = []
        infos = {"input": [], "output": [], "image": []}

        for prompt_ids_len, input_id, attention_mask, pixel_value, image_grid_thw, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            pixel_values.append(pixel_value)
            image_grid_thws.append(image_grid_thw)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])
            infos["image"].append(info["image"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, torch.cat(pixel_values, dim=0), torch.cat(image_grid_thws, dim=0), infos

# TODO 
    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        prompt_ids_lens = []
        infos = {"input_length": [], "response_ranges": [] if self.multiturn else None}
        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.full_like(input_id.flatten(), index))
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            if self.multiturn:
                if len(infos["response_ranges"]) >= 1:
                    for i in range(len(info["response_ranges"])):
                        info["response_ranges"][i][0] += infos["response_ranges"][-1][-1][1] # end_index of the last response of the last item
                        info["response_ranges"][i][1] += infos["response_ranges"][-1][-1][1]
                infos["response_ranges"].append(info["response_ranges"])
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)
        
        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0: # not divisible by multiple_of; here we align for grouping
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos
