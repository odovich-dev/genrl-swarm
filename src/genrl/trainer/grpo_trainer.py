import contextlib
import gc
import os
from collections import defaultdict
from typing import Any, List
import torch
import torch.utils.data
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from trl.models import create_reference_model
from trl.trainer.grpo_config import GRPOConfig
from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule

class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Unified Trainer for GRPO, supporting both standard training and vLLM inference.
    """
    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the trainer module.
        Can be configured for vLLM inference or standard GRPO training.
        """
        # --- Common Configuration ---
        self.use_vllm = kwargs.get("use_vllm", False)
        self.log_with = kwargs.get("log_with", None)
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.callbacks = kwargs.get("callbacks", [])
        self.global_step = 0
        self.num_generations = kwargs.get("num_generations", 2)
        if not models or len(models) < 1:
            raise ValueError("A model name must be provided.")

        model_name = models[0]
        self.model_name = model_name
        # --- Mode-Specific Initialization ---
        if self.use_vllm:
            # --- VLLM INFERENCE PATH ---
            print("✅ Initializing with vLLM for fast inference.")
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                raise ImportError("vLLM not installed. Please run `pip install vllm`")
            self.vllm_engine = LLM(
                model=model_name,
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.85),
                dtype="float16",
            )
            self.vllm_sampling_params = SamplingParams(
                n=self.num_generations,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
                max_tokens=kwargs.get("max_tokens", 512),
            )
            self.processing_class = self.vllm_engine.get_tokenizer()
            self.model = None
            self.ref_model = None
            self.optimizer = None
            self.device = torch.device("cuda")
            print("✅ vLLM is initialized and active.")
        else:
            # --- STANDARD GRPO TRAINING PATH ---
            print("✅ Initializing with Hugging Face model for GRPO training.")

            quant_config = None
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print("✅ bitsandbytes found - enabling 4-bit quantization")
            except Exception:
                print("⚠️ bitsandbytes not installed - skipping quantization for training.")
            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            }
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

            config = kwargs.get("config", None)
            self.args = (
                config
                if isinstance(config, GRPOConfig)
                else GRPOConfig(config) if config else GRPOConfig()
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
            self.processing_class = kwargs.get("processing_class", None)
            self.epsilon = kwargs.get("epsilon", 0.2)
            self.epsilon_high = kwargs.get("epsilon_high", 0.28)
            self.beta = kwargs.get("beta", 0.0)
            self.enable_gradient_checkpointing = kwargs.get(
                "enable_gradient_checkpointing", True
            )
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.autocast = torch.amp.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.fp16
                )
            else:
                self.device = torch.device("cpu")
                self.autocast = contextlib.nullcontext()
            self._initialize_model(self.enable_gradient_checkpointing)
            self._initialize_tokenizers()
            self._initialize_generation_config()
        # --- Common Initializations for both modes ---
        self._initialize_metrics()
        self.init_tracker(self.save_dir, log_with=self.log_with)

    def generate(
        self, inputs: Any, return_completion_ids: bool = False, stage=0
    ) -> Any:
        input_tokens_or_prompts = self._process_inputs(inputs)
        if self.use_vllm:
            if not hasattr(self, "vllm_engine"):
                raise RuntimeError("Trainer is not initialized for vLLM inference.")

            vllm_outputs = self.vllm_engine.generate(input_tokens_or_prompts, self.vllm_sampling_params)

            rollout = [[completion.text for completion in output.outputs] for output in vllm_outputs]

            if return_completion_ids:
                rollout_ids = [[completion.token_ids for completion in output.outputs] for output in vllm_outputs]
                return rollout, rollout_ids
            return rollout
        else:
            input_tokens = input_tokens_or_prompts
            rollout, rollout_ids = ([], [])

            for _ in range(self.num_generations):
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_tokens.input_ids.to(self.model.device),
                        attention_mask=input_tokens.attention_mask.to(self.model.device),
                        generation_config=self.generation_config,
                    )
                prompt_length = input_tokens.input_ids.size(1)
                completion_ids = outputs[:, prompt_length:]
                completions = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=True
                )
                if len(rollout) == 0:
                    rollout = [[comp] for comp in completions]
                    if return_completion_ids:
                        rollout_ids = [[comp_id] for comp_id in completion_ids]
                else:
                    for idx, comp in enumerate(completions):
                        rollout[idx].append(comp)
                        if return_completion_ids:
                            rollout_ids[idx].append(completion_ids[idx])
            if return_completion_ids:
                return rollout, rollout_ids
            else:
                return rollout

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        if self.use_vllm:
            return
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(
                stage, state, data_manager, reward_manager, global_step
            )
        self.global_step = global_step
        self.model.eval()

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]
        if with_template:
            templated_prompts = []
            for item in inputs:
                text_content = ""
                if isinstance(item, str):
                    text_content = item
                elif isinstance(item, dict):
                    text_content = item.get("prompt") or item.get("content")
                    if text_content is None:
                        raise ValueError(f"Input dictionary must have a 'prompt' or 'content' key. Found: {item.keys()}")
                else:
                    raise TypeError(f"Unsupported input type for chat template processing: {type(item)}")
                if isinstance(text_content, list):
                    text_content = "\n".join(map(str, text_content))

                chat_history = [{"role": "user", "content": text_content}]

                formatted_prompt = self.processing_class.apply_chat_template(
                    chat_history, tokenize=False, add_generation_prompt=True
                )

                count = self.num_generations if for_training else 1
                for _ in range(count):
                    templated_prompts.append(formatted_prompt)
            if self.use_vllm and not for_training:
                return templated_prompts
        else:
            if for_training:
                templated_prompts = []
                for generations in inputs:
                    for output in generations:
                        templated_prompts.append(output)
            else:
                templated_prompts = [item[0] for item in inputs]
        return self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )

    def _initialize_model(self, enable_gradient_checkpointing):
        if self.use_vllm: return
        self.model = self.model.to(self.device)
        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.model).to(self.model.device)

    def _initialize_tokenizers(self):
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, padding_side="left"
            )

        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id

        if self.model and self.model.config:
            self.model.config.pad_token_id = self.processing_class.pad_token_id

    def _initialize_metrics(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        if not self.use_vllm:
            self._total_train_tokens = 0

    def _initialize_generation_config(self):
        if self.use_vllm: return
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        model = model.to(input_ids.device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]
        loss_mask = (
            attention_mask[:, -logits_to_keep:].to(dtype=logits.dtype).contiguous()
        )
        labels = input_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:].contiguous()
        logits = logits / self.args.temperature

        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction="none",
        ).view(logits.shape[0], logits.shape[1])

        token_log_probs = (
            token_log_probs * loss_mask
            + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        )
        return token_log_probs

    def compute_loss(
        self, model, inputs, num_items_in_batch=1, mode="train", return_metrics=False
    ):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.model.device
        )
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        if self.beta != 0.0:
            ref_per_token_logps = self._get_per_token_logps(
                self.ref_model, input_ids, attention_mask, logits_to_keep
            )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.args.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.epsilon,
            1 + self.epsilon_high if self.epsilon_high is not None else self.epsilon,
        )
        advantages = advantages.unsqueeze(dim=-1)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.item())
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())
        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if self.beta != 0.0 else None,
            "clip_ratio": clip_ratio.item(),
        }
        if return_metrics:
            return loss, metrics
        else:
            return loss

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        global_step += 1
        stage_inputs = state.get_stage_state(stage)
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
        assert stage_inputs is not None, f"No inputs found for stage {stage}"

        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [
            stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][
                index_mapping[idx][2]
            ] for idx, _ in enumerate(index_mapping)
        ]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"
        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"], model_inputs["prompt_mask"] = (
            processed_inputs.input_ids.to(self.model.device),
            processed_inputs.attention_mask.to(self.model.device),
        )
        processed_outputs = self._process_inputs(
            stage_outputs, with_template=False, for_training=True
        )
        model_inputs["completion_ids"], model_inputs["completion_mask"] = (
            processed_outputs.input_ids.to(self.model.device),
            processed_outputs.attention_mask.to(self.model.device),
        )
        rewards = reward_manager[stage]
        rewards = [
            rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards)
        with torch.no_grad():
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            if rewards.shape[1] > 1:
                advantages /= rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = torch.flatten(advantages).to(self.model.device)
        model_inputs["advantages"] = advantages.squeeze(dim=-1)
        model_inputs["old_per_token_logps"] = None
        with self.autocast:
            loss = self.compute_loss(self.model, model_inputs)
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        metrics = {"train/loss": loss.cpu().mean().item()}
        metrics.update({"train/rewards": rewards.cpu().mean().item()})
        self.log(metrics, global_step)
        self.cleanup_step()
        return global_step

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        pass

    def save(self, save_dir: str) -> None:
        if self.use_vllm:
            print("Save is not supported in vLLM mode.")
            return
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processing_class.save_pretrained(save_dir)
        torch.save(
            {
                "metrics": self._metrics,
                "total_train_tokens": self._total_train_tokens,
                "generation_config": self.generation_config,
            },
            os.path.join(save_dir, "trainer_state.pt"),
        )

    @classmethod
    def load(cls, load_dir: str) -> "GRPOLanguageTrainerModule":
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        tokenizer = AutoTokenizer.from_pretrained(load_dir)
        trainer = cls([model.config._name_or_path], processing_class=tokenizer, use_vllm=False)
        trainer_state = torch.load(os.path.join(load_dir, "trainer_state.pt"))
        trainer._metrics = trainer_state["metrics"]
        trainer._total_train_tokens = trainer_state["total_train_tokens"]
        trainer.generation_config = trainer_state["generation_config"]
        return trainer

    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()
