from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    images_key="images",
    reference_key=None,
    label_key=None,
    apply_chat_template=None,
    processor=None,
    min_size=56,
    system_prompt=None,
) -> tuple:

    if apply_chat_template:
        _sys_prompt = []
        if system_prompt:
            _sys_prompt += [{"role": "system", "content": system_prompt}]
        conversation = data[prompt_key]  
        if conversation and isinstance(conversation[0], dict) and 'value' in conversation[0]:
            user_text = conversation[0]['value'].split("<image>")[-1]
        else:
            user_text = conversation.split("<image>")[-1]
        if input_template:
            user_text = input_template.format(user_text)
        _prompt = _sys_prompt + [{"role": "user", "content": [{"type": "image", "image": ""}, {"type": "text", "text": user_text}]}]
        prompt = apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[prompt_key]
        if input_template:
            prompt = input_template.format(prompt)

    images = data[images_key] if data.get(images_key) else [Image.new("RGB", (min_size, min_size), color=(128, 128, 128))]
    reference = data[reference_key] if reference_key in data else None
    if reference and isinstance(reference, dict) and 'value' in reference:
        reference = reference['value']
    labels = data[label_key] if label_key in data else None
    ids = data["id"] if "id" in data else None

    return prompt, images, reference, labels, ids

class PromptDatasetVL(Dataset):  
    def __init__(  
        self,  
        dataset,  
        tokenizer,  
        processor,   
        max_length,  
        strategy,  
        input_template=None  
    ) -> None:  
        super().__init__()  
        self.strategy = strategy  
        self.tokenizer = tokenizer  
        self.processor = processor  
        self.max_length = max_length  
        self.min_size = int(processor.image_processor.min_pixels ** 0.5)

        # chat_template  
        self.input_template = input_template  
        self.prompt_key = getattr(self.strategy.args, "input_key", None)  
        self.images_key = getattr(self.strategy.args, "images_key", None)  
        self.reference_key = getattr(self.strategy.args, "reference_key", None)  
        self.label_key = getattr(self.strategy.args, "label_key", None)  
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)  
        self.system_prompt = getattr(self.strategy.args, "system_prompt", None)  

        if self.apply_chat_template:  
            # self.apply_chat_template = self.tokenizer.apply_chat_template
            self.apply_chat_template = self.processor.apply_chat_template
            strategy.print(self.apply_chat_template)
            user_text = "Describe this image."
            if input_template:
                user_text = input_template.format(user_text)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", "image": "",
                        },
                        {"type": "text", "text": user_text},
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
            
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)  
            if tokenizer_chat_template:  
                self.tokenizer.chat_template = tokenizer_chat_template  

        self.dataset = dataset  

    def __len__(self):  
        return len(self.dataset)  

    def __getitem__(self, idx):  
        data = self.dataset[idx]  
        prompt, image, reference, label, uid = preprocess_data(  
            data,  
            self.input_template,  
            self.prompt_key,  
            self.images_key,  
            self.reference_key,  
            self.label_key,  
            self.apply_chat_template,  
            self.processor,
            self.min_size,
            self.system_prompt
        )  
        return prompt, image, reference, label, uid
    
    def collate_fn(self, item_list):
        prompts_list = []
        images_list = []
        references_list = []
        labels_list = []
        ids_list = []
        for prompts, images, references, labels, ids in item_list:
            prompts_list.append(prompts)
            images_list.append(images)
            references_list.append(references)
            labels_list.append(labels)
            ids_list.append(ids)
        return prompts_list, images_list, references_list, labels_list, ids_list
