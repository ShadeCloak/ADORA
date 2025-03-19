import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from PIL import Image
import numpy as np

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor_vl import ActorVL
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class ExperienceVL:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    pixel_values: (B * h, w)
    image_grid_thws: (B, 3)
    raw_images: Optional[List[Image.Image]]  # raw images before processing
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    # 图像处理相关
    pixel_values: Optional[torch.Tensor] = None  # image pixel processed by HF processor
    image_grid_thws: Optional[torch.Tensor] = None  # image grid thw
    raw_images: Optional[List[Image.Image]] = None  # raw images before processing
    # InternVL image_info
    pixel_values_intern: Optional[torch.Tensor] = None
    image_flags: Optional[torch.Tensor] = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.pixel_values is not None:
            self.pixel_values = to(self.pixel_values, device)
        if self.image_grid_thws is not None:
            self.image_grid_thws = to(self.image_grid_thws, device)
        if self.pixel_values_intern is not None:
            self.pixel_values_intern = to(self.pixel_values_intern, device)
            self.image_flags = to(self.image_flags, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.pixel_values is not None:
            self.pixel_values = pin_memory(self.pixel_values)
        if self.image_grid_thws is not None:
            self.image_grid_thws = pin_memory(self.image_grid_thws)
        if self.pixel_values_intern is not None:
            self.pixel_values_intern = pin_memory(self.pixel_values_intern)
            self.image_flags = pin_memory(self.image_flags)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class SamplesVL:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None

    pixel_values: Optional[torch.Tensor] = None # image pixel processed by HF processor
    image_grid_thws: Optional[torch.Tensor] = None # image grid thw 
    raw_images: Optional[List[Image.Image]] = None  # 图像数据列表
    
    # InternVL image_info
    pixel_values_intern: Optional[torch.Tensor] = None
    image_flags: Optional[torch.Tensor] = None
    
    num_actions: Union[int, torch.Tensor] = None
    packed_seq_lens: Optional[torch.Tensor] = None
    response_length: torch.Tensor = None
    total_length: torch.Tensor = None
    
    references: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    prompts: list[str] = None
    ids: Optional[List[str]] = None


class NaiveExperienceMakerVL(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: ActorVL,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: ActorVL,
        tokenizer,
        processor, 
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, references)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}
    
    # processor
    def processor_fn(self, texts, images, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.processor(
                text = texts,
                images = images, 
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.processor(
            text = texts,
            images = images, 
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], 
        all_images, 
        all_references, 
        all_labels, 
        all_ids, 
        **generate_kwargs
    ) -> List[ExperienceVL]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_images, all_references, all_labels, all_ids, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            samples_list = self.generate_samples(all_prompts, all_images, all_references, all_labels, all_ids, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.distributed.barrier()
        torch.cuda.synchronize()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences
    
    # TODO support InternVL
    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_images, all_references, all_labels, all_ids, **generate_kwargs) -> List[SamplesVL]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_images = sum([[image] * args.n_samples_per_prompt for image in all_images], [])
        all_references = sum([[reference] * args.n_samples_per_prompt for reference in all_references], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_ids = sum([[uid] * args.n_samples_per_prompt for uid in all_ids], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            images = all_images[i : i + args.micro_rollout_batch_size]
            references = all_references[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            ids = all_ids[i : i + args.micro_rollout_batch_size]
            inputs = self.processor_fn(prompts, images, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            if "internvl" in self.actor.pretrain_or_model.lower():
                samples = SamplesVL(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask, 
                    pixel_values_intern=inputs["pixel_values"], 
                    image_flags=inputs["image_flags"], 
                    raw_images=images,  # 使用分批处理后的图像数据
                    num_actions=action_mask.size(1),
                    packed_seq_lens=None,
                    response_length=action_mask.float().sum(dim=-1),
                    total_length=attention_mask.float().sum(dim=-1),
                    references=references,
                    labels=labels,
                    prompts=prompts,
                    ids=ids,
                )
            else:
                samples = SamplesVL(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask, 
                    pixel_values=inputs["pixel_values"], 
                    image_grid_thws=inputs["image_grid_thw"], 
                    raw_images=images,  # 使用分批处理后的图像数据
                    num_actions=action_mask.size(1),
                    packed_seq_lens=None,
                    response_length=action_mask.float().sum(dim=-1),
                    total_length=attention_mask.float().sum(dim=-1),
                    references=references,
                    labels=labels,
                    ids=ids,
                )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: SamplesVL) -> ExperienceVL:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        pixel_values = samples.pixel_values
        image_grid_thws = samples.image_grid_thws
        raw_images = samples.raw_images
        pixel_values_intern = samples.pixel_values_intern
        image_flags = samples.image_flags
        num_actions = samples.num_actions

        if "internvl" in self.actor.pretrain_or_model.lower():
            # log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask, pixel_values_intern=pixel_values_intern, image_flags=image_flags)

            # init log probs
            if self.initial_model is not None:
                base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask, pixel_values_intern=pixel_values_intern, image_flags=image_flags)
            else:
                base_action_log_probs = None

            # values
            if self.critic is not None:
                value = self.critic(sequences, num_actions, attention_mask, pixel_values_intern=pixel_values_intern, image_flags=image_flags)
            else:
                value = None

            # rewards
            if self.remote_rm_url is not None:
                # remote RM
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                references = samples.references if hasattr(samples, "references") else None
                if self.custom_reward_func:
                    r = self.custom_reward_func(queries, samples.prompts, references).to(
                        device=action_log_probs.device
                    )
                else:
                    r = remote_rm_fn(
                        api_url=self.remote_rm_url, queries=queries, prompts=samples.prompts, references=references, raw_images=raw_images
                    ).to(device=action_log_probs.device)
            else:
                # local RM
                r = self.reward_model(sequences, attention_mask, pixel_values_intern=pixel_values_intern, image_flags=image_flags)
        else:
            # log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask, pixel_values, image_grid_thws)

            # init log probs
            if self.initial_model is not None:
                base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask, pixel_values, image_grid_thws)
            else:
                base_action_log_probs = None

            # values
            if self.critic is not None:
                value = self.critic(sequences, num_actions, attention_mask, pixel_values, image_grid_thws)
            else:
                value = None

            # rewards
            if self.remote_rm_url is not None:
                # remote RM
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                references = samples.references if hasattr(samples, "references") else None
                if self.custom_reward_func:
                    r = self.custom_reward_func(queries, samples.prompts, references).to(
                        device=action_log_probs.device
                    )
                else:
                    r = remote_rm_fn(
                        api_url=self.remote_rm_url, queries=queries, prompts=samples.prompts, references=references, raw_images=raw_images
                    ).to(device=action_log_probs.device)
            else:
                # local RM
                r = self.reward_model(sequences, attention_mask, pixel_values, image_grid_thws)

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        info = {
            "uid": torch.Tensor(samples.ids), 
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return ExperienceVL(
            sequences,
            pixel_values,
            image_grid_thws,
            raw_images,  # 传递原始图像数据
            pixel_values_intern,  # InternVL image_info
            image_flags,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[ExperienceVL]) -> Tuple[List[ExperienceVL], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        rewards = torch.cat([experience.info["reward"] for experience in experiences])
        rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
        if args.use_adora:
            ids = torch.cat([experience.info["uid"] for experience in experiences])
            ids = ids.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            response_lengths = torch.cat([experience.info["response_length"] for experience in experiences])
            response_lengths = response_lengths.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            weights = weight_func(rewards, response_lengths, args.adora_lamda)
            uids_set = set(ids[weights==1.0].cpu().numpy())
            correct_samples_count = count_correct_samples_per_id(ids, rewards, uids_set, args.n_samples_per_prompt)
            logger.info(f"\nuids: {correct_samples_count}")
        else:
            weights = torch.ones_like(rewards, device=rewards.device)  
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards * weights
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards * weights
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards * weights
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        rewards = rewards * weights
        rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
        return experiences, rewards

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMakerVL(NaiveExperienceMakerVL):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_images, all_references, all_labels, all_ids, **generate_kwargs
    ) -> List[ExperienceVL]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_images, all_references, all_labels, all_ids, **generate_kwargs)
        # print(f"=========================experiences: {experiences}")
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_images, all_references, all_labels, all_ids, **generate_kwargs) -> List[SamplesVL]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_images, all_references, all_labels, all_ids, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_images, all_references, all_labels, all_ids, **generate_kwargs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: SamplesVL) -> ExperienceVL:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        pixel_values = samples.pixel_values
        image_grid_thws = samples.image_grid_thws
        raw_images = samples.raw_images
        pixel_values_intern = samples.pixel_values_intern
        image_flags = samples.image_flags
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        labels = samples.labels if hasattr(samples, "labels") else None

        start = time.time()
        num_custom_reward_func = 0
        if "internvl" in self.actor.pretrain_or_model.lower():
            # TODO support other dtype
            if self.strategy.bf16:
                pixel_values_intern = pixel_values_intern.to(dtype=torch.bfloat16)

            sequences_cpu, attention_mask_cpu, pixel_values_intern_cpu, image_flags_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
                pixel_values_intern.to("cpu"),
                image_flags.to("cpu"),
            )

            # init log probs
            if self.initial_model is not None:
                base_action_log_probs_ref = self.initial_model.forward.remote(
                    sequences_cpu, num_actions, attention_mask_cpu, pixel_values_intern=pixel_values_intern_cpu, image_flags=image_flags_cpu, packed_seq_lens=packed_seq_lens
                )
                
                if args.colocate_actor_ref or args.colocate_all_models:
                    ray.get([base_action_log_probs_ref])
                    ray.get([self.initial_model.empty_cache.remote()])
            else:
                base_action_log_probs_ref = ray.put(None)

            # values
            if self.critic is not None:
                value_ref = self.critic.forward.remote(
                    sequences_cpu, num_actions, attention_mask_cpu, pixel_values_intern=pixel_values_intern_cpu, image_flags=image_flags_cpu, packed_seq_lens=packed_seq_lens
                )
                # avoid CUDA OOM when colocate models
                if args.colocate_critic_reward or args.colocate_all_models:
                    ray.get([value_ref])
                    ray.get([self.critic.empty_cache.remote()])
            else:
                value_ref = ray.put(None)

            # rewards
            r_refs = []
            # support remote RM API with ray
            if not self.remote_rm_url:
                for rm in self.reward_model:
                    r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, pixel_values_intern=pixel_values_intern_cpu, image_flags=image_flags_cpu, packed_seq_lens=packed_seq_lens))
            else:
                # remote RM
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

                references = samples.references if hasattr(samples, "references") else None
                if self.custom_reward_func:
                    r = self.custom_reward_func.remote(queries, samples.prompts, references)
                    r_refs.append(r)
                    num_custom_reward_func = 1
                if len(self.remote_rm_url) > num_custom_reward_func:
                    for rm in self.remote_rm_url[num_custom_reward_func:]:
                        # Pass raw images directly from samples
                        r = remote_rm_fn_ray.remote(
                            api_url=rm,
                            queries=queries,
                            prompts=samples.prompts,
                            references=references,
                            raw_images=raw_images
                        )
                        r_refs.append(r)

            if args.colocate_all_models and not self.remote_rm_url:
                ray.get(r_refs)
                ray.get([self.reward_model[0].empty_cache.remote()])

            # log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask, pixel_values_intern=pixel_values_intern, image_flags=image_flags, packed_seq_lens=packed_seq_lens)
            actor_value_rm_time = time.time() - start

            # wait initial/critic/reward model done
            start = time.time()
            ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            wait_time = time.time() - start

            base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
            if base_action_log_probs is not None:
                base_action_log_probs = base_action_log_probs.to(device)
            if value is not None:
                value = value.to(device)
            rewards = [r.to(device) for r in rewards]
            r = self.reward_fn(rewards, labels, num_custom_reward_func) if len(rewards) > 0 else rewards[0]
            # print(f"==========================rewards before reward_fn: {rewards}")
            # print(f"==========================rewards after reward_fn: {r}")
        else:
            # TODO support other dtype
            if self.strategy.bf16:
                pixel_values = pixel_values.to(dtype=torch.bfloat16)
            
            sequences_cpu, attention_mask_cpu, pixel_values_cpu, image_grid_thws_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
                pixel_values.to("cpu"),
                image_grid_thws.to("cpu"),
            )

            # init log probs
            if self.initial_model is not None:
                base_action_log_probs_ref = self.initial_model.forward.remote(
                    sequences_cpu, num_actions, attention_mask_cpu, pixel_values=pixel_values_cpu, image_grid_thw=image_grid_thws_cpu, packed_seq_lens=packed_seq_lens
                )
                if args.colocate_actor_ref or args.colocate_all_models:
                    ray.get([base_action_log_probs_ref])
                    ray.get([self.initial_model.empty_cache.remote()])
            else:
                base_action_log_probs_ref = ray.put(None)

            # values
            if self.critic is not None:
                value_ref = self.critic.forward.remote(
                    sequences_cpu, num_actions, attention_mask_cpu, pixel_values=pixel_values_cpu, image_grid_thw=image_grid_thws_cpu, packed_seq_lens=packed_seq_lens
                )
                # avoid CUDA OOM when colocate models
                if args.colocate_critic_reward or args.colocate_all_models:
                    ray.get([value_ref])
                    ray.get([self.critic.empty_cache.remote()])
            else:
                value_ref = ray.put(None)

            # rewards
            r_refs = []
            # support remote RM API with ray
            if not self.remote_rm_url:
                for rm in self.reward_model:
                    r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, pixel_values=pixel_values_cpu, image_grid_thw=image_grid_thws_cpu, packed_seq_lens=packed_seq_lens))
            else:
                # remote RM
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

                references = samples.references if hasattr(samples, "references") else None
                if self.custom_reward_func:
                    r = self.custom_reward_func.remote(queries, samples.prompts, references)
                    r_refs.append(r)
                    num_custom_reward_func = 1
                if len(self.remote_rm_url) > num_custom_reward_func:
                    for rm in self.remote_rm_url[num_custom_reward_func:]:
                        # Pass raw images directly from samples
                        r = remote_rm_fn_ray.remote(
                            api_url=rm,
                            queries=queries,
                            prompts=samples.prompts,
                            references=references,
                            raw_images=raw_images
                        )
                        r_refs.append(r)

            if args.colocate_all_models and not self.remote_rm_url:
                ray.get(r_refs)
                ray.get([self.reward_model[0].empty_cache.remote()])

            # log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thws, packed_seq_lens=packed_seq_lens)
            actor_value_rm_time = time.time() - start

            # wait initial/critic/reward model done
            start = time.time()
            ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            wait_time = time.time() - start

            base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
            base_action_log_probs = base_action_log_probs.to(device)
            if value is not None:
                value = value.to(device)
            rewards = [r.to(device) for r in rewards]
            r = self.reward_fn(rewards, labels, num_custom_reward_func) if len(rewards) > 0 else rewards[0]
            # print(f"==========================rewards before reward_fn: {rewards}")
            # print(f"==========================rewards after reward_fn: {r}")

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "uid": torch.Tensor(samples.ids), 
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = ExperienceVL(
            sequences,
            pixel_values,
            image_grid_thws,
            raw_images,
            pixel_values_intern,  # InternVL image_info
            image_flags,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_images, all_references, all_labels, all_ids, **kwargs) -> List[SamplesVL]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 4096),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        if isinstance(all_images[0], list): 
            all_images = sum([image * args.n_samples_per_prompt for image in all_images], [])
        else:
            all_images = sum([[image] * args.n_samples_per_prompt for image in all_images], [])
        all_references = sum([[reference] * args.n_samples_per_prompt for reference in all_references], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_ids = sum([[uid] * args.n_samples_per_prompt for uid in all_ids], [])

        # Tokenize prompts to get prompt token IDs
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            # Slice the batch of prompts and images
            prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
            images = all_images[i * batch_size : (i + 1) * batch_size]
            
            # Prepare multi-modal inputs for each prompt in the batch
            llm_inputs = []
            for j in range(len(prompts)):
                llm_inputs.append({
                    "prompt": prompts[j],
                    "multi_modal_data": {"image": images[j]},
                })
            # debug
            # self.strategy.print("prompt: ", prompts[j])
            # Add requests to the LLM engine with both prompt token IDs and multi-modal inputs
            # prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(
                    rank,
                    sampling_params=sampling_params,
                    prompts=llm_inputs  # Pass multi-modal inputs as prompts parameter
                )
            )
        ray.get(refs)

        # Make sure all requests are sent.
        torch.distributed.barrier()

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        inputs = self.processor_fn(all_prompts, all_images, self.prompt_max_len, padding=False)
        all_prompt_token_ids = inputs["input_ids"]
        all_images_pixel_values = inputs["pixel_values"]
        if "internvl" in self.actor.pretrain_or_model.lower():
            all_images_grid_thw = inputs["num_tiles"]
            all_image_flags = inputs["image_flags"]
        else:
            all_images_grid_thw = inputs["image_grid_thw"]
        samples_list = []
        index_pixel_patch = 0
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            prompt_token_ids = all_prompt_token_ids[i : i + self.strategy.args.micro_rollout_batch_size]
            # images_pixel_values = all_images_pixel_values[i : i + self.strategy.args.micro_rollout_batch_size]
            images_grid_thw = all_images_grid_thw[i : i + self.strategy.args.micro_rollout_batch_size]
            references = all_references[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            ids = all_ids[i : i + self.strategy.args.micro_rollout_batch_size]
            raw_images = all_images[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                pixel_values = []
                image_grid_thws = []
                image_flags = []

                for output, prompt_token_id, images_grid in zip(outputs, prompt_token_ids, images_grid_thw):
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
                    
                    # split pixel_patch
                    if "internvl" in self.actor.pretrain_or_model.lower():
                        num_patch = images_grid if isinstance(images_grid, int) else images_grid.sum().item()
                        _image_flags = all_image_flags[index_pixel_patch: index_pixel_patch + num_patch]
                        image_flags.append(_image_flags)
                        image_grid_thws.append(torch.tensor([1,1,num_patch]).unsqueeze(0))
                    else:
                        num_patch = images_grid[0] * images_grid[1] * images_grid[2]
                        image_grid_thws.append(images_grid.unsqueeze(0))
                    images_pixel_value = all_images_pixel_values[index_pixel_patch: index_pixel_patch + num_patch]
                    index_pixel_patch += num_patch

                    # concat input and output
                    sequences.append(input_ids + output_ids)
                    pixel_values.append(images_pixel_value)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                if "internvl" in self.actor.pretrain_or_model.lower():
                    samples_list.append(
                        SamplesVL(
                            sequences=sequences,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            pixel_values=None,
                            image_grid_thws=torch.cat(image_grid_thws, dim=0).to("cuda"),
                            pixel_values_intern=torch.cat(pixel_values, dim=0).to("cuda"), 
                            image_flags=torch.cat(image_flags, dim=0).to("cuda"),
                            raw_images=raw_images,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                            references=references,
                            labels=labels,
                            prompts=prompts,
                            ids=ids,
                        )
                    )
                else:
                    samples_list.append(
                        SamplesVL(
                            sequences=sequences,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            pixel_values=torch.cat(pixel_values, dim=0).to("cuda"), 
                            image_grid_thws=torch.cat(image_grid_thws, dim=0).to("cuda"),
                            raw_images=raw_images,
                            pixel_values_intern=None, 
                            image_flags=None,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                            references=references,
                            labels=labels,
                            prompts=prompts,
                            ids=ids,
                        )
                    )
            # TODO support VLM
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    SamplesVL(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None


def weight_func(rewards, response_length, lamda=0.1):  
    """   
    """  
    weights = torch.ones_like(rewards, device=rewards.device)  
    for i in range(rewards.shape[0]):  
        reward_row = rewards[i]  
        response_row = response_length[i]  
        response_reward_1 = response_row[reward_row > 0.5]  
        response_reward_not_1 = response_row[(reward_row <= 0.5) & (response_row < 4094)]  
        mean_reward_1 = response_reward_1.max() if response_reward_1.numel() > 0 else float('-inf')  
        mean_reward_not_1 = response_reward_not_1.mean() if response_reward_not_1.numel() > 0 else float('inf') 

        if mean_reward_1 > mean_reward_not_1 or response_reward_1.numel() == 0:  
            weights[i] = 1.0
        else: 
            weights[i] = lamda

    return weights

def count_correct_samples_per_id(ids, rewards, uids_set, n_samples_per_prompt):  
    """  
    Count how many correct samples (reward=1.0) each ID in uids_set has  
    
    Args:  
        ids: Tensor of shape [num_prompts, n_samples_per_prompt] containing IDs  
        rewards: Tensor of shape [num_prompts, n_samples_per_prompt] containing rewards (1.0 for correct)  
        uids_set: Set of unique user IDs to analyze  
        n_samples_per_prompt: Number of samples per prompt  
        
    Returns:  
        Dictionary mapping each ID to the count of correct samples  
    """  
    # Convert to numpy for easier processing if they're tensors  
    if torch.is_tensor(ids):  
        ids = ids.cpu().numpy()  
    if torch.is_tensor(rewards):  
        rewards = rewards.cpu().numpy()  
    
    # Initialize results dictionary  
    correct_counts = {}  
    
    # For each unique ID in the set  
    for uid in uids_set:  
        # Find all positions where this ID appears  
        positions = np.where(ids == uid)  
        
        # Count how many of these positions have a reward of 1.0  
        correct_count = np.sum(rewards[positions] > 0.5)  
        
        # Store the count  
        correct_counts[uid] = correct_count  
    
    return correct_counts  