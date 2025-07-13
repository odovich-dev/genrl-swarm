import contextlib
import gc
import os
from collections import defaultdict
from datetime import datetime
import warnings
from typing import Any, List

import torch
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
# REMOVED from trl.data_utils import apply_chat_template # We will use tokenizer's method directly
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
        self.save_dir = kwargs.get("log_dir", f"./outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.callbacks = kwargs.get("callbacks", [])
        self.global_step = 0
        self.num_generations = kwargs.get("num_generations", 2)
        
        # GRPO fundamentally needs multiple generations for relative comparison
        if self.num_generations <= 1:
            warnings.warn(f"num_generations should be > 1 for GRPO training. Resetting to default value 2. Current value: {self.num_generations}")
            self.num_generations = 2

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
                raise ImportError("vLLm not installed. Please run `pip install vllm`")
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
            self.model = None # Model not loaded into HuggingFace format if using vLLM
            self.ref_model = None
            self.optimizer = None
            self.device = torch.device("cuda")
            print("✅ vLLM is initialized and active.")
        else:
            # --- STANDARD GRPO TRAINING PATH ---
            print("✅ Initializing with Hugging Face model for GRPO training.")

            quant_config = None
            try:
                # Check if BitsAndBytesConfig is available (implies bitsandbytes installed)
                from transformers import BitsAndBytesConfig as _BitsAndBytesConfig # Rename to avoid conflict if already imported
                quant_config = _BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print("✅ bitsandbytes found — enabling 4-bit quantization")
            except ImportError:
                print("⚠️ bitsandbytes not installed — skipping quantization for training.")
            except Exception as e:
                print(f"⚠️ Error initializing BitsAndBytesConfig: {e} — skipping quantization.")
                quant_config = None # Ensure it's None if an error occurred

            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16, # Use bfloat16 for better precision if available
            }
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
            
            # Initialize tokenizer first to ensure its vocabulary size is known
            self.processing_class = kwargs.get("processing_class", None)
            if self.processing_class is None:
                # If model_name is a string, use it. If it's a model object, get name from config
                tokenizer_path_or_name = model_name if isinstance(model_name, str) else (model_name.config._name_or_path if hasattr(model_name, 'config') else 'EleutherAI/gpt-neo-125M') # Fallback to a default if cannot infer
                self.processing_class = AutoTokenizer.from_pretrained(
                    tokenizer_path_or_name, padding_side="left"
                )
            # Ensure pad_token_id is set for generation and padding
            if self.processing_class.pad_token_id is None:
                self.processing_class.pad_token_id = self.processing_class.eos_token_id


            # The 'models' list is expected to contain the model object itself if not using vLLM
            if isinstance(model_name, str): # If model_name is a string, load it from pretrained
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            else: # Assume model_name is already a model object (e.g., passed directly)
                self.model = model_name
                # If a model object is passed, ensure it's on the correct device and dtype
                if quant_config is None: # If not quantized, move to device and set dtype
                    self.model = self.model.to(load_kwargs.get("device_map", "cpu")).to(dtype=load_kwargs.get("torch_dtype", torch.float32))
                # For quantized models, device_map handles it, and dtype is fixed by bnb config

            # --- FIX: Resize token embeddings to match tokenizer vocabulary ---
            # This is the crucial part for the size mismatch error
            if self.model.get_input_embeddings().weight.shape[0] != len(self.processing_class):
                old_vocab_size = self.model.get_input_embeddings().weight.shape[0]
                new_vocab_size = len(self.processing_class)
                print(f"🔄 Resizing model token embeddings from {old_vocab_size} to {new_vocab_size} to match tokenizer vocabulary.")
                self.model.resize_token_embeddings(new_vocab_size)
                # Also ensure the model's config pad_token_id is updated if tokenizer has one
                if self.processing_class.pad_token_id is not None and hasattr(self.model.config, 'pad_token_id'):
                    self.model.config.pad_token_id = self.processing_class.pad_token_id
            # --- END FIX ---


            config = kwargs.get("config", None)
            self.args = (
                config
                if isinstance(config, GRPOConfig)
                else GRPOConfig(config) if config else GRPOConfig()
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
            # self.processing_class is now initialized above

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
            elif torch.backends.mps.is_available(): # Added for Apple Silicon (MPS)
                self.device = torch.device("mps")
                self.autocast = contextlib.nullcontext() 
            else:
                self.device = torch.device("cpu")
                self.autocast = contextlib.nullcontext()

            self._initialize_model(self.enable_gradient_checkpointing) # This handles ref_model setup etc.
            # self._initialize_tokenizers() # No longer needed here, done above
            self._initialize_generation_config() # Moved here as it needs tokenizer & args

        # --- Common Initializations for both modes ---
        self._initialize_metrics()
        self.init_tracker(self.save_dir, log_with=self.log_with)

    def _initialize_model(self, enable_gradient_checkpointing):
        """Initialize the model and reference model."""
        if self.use_vllm: return # No HF model setup if using vLLM
        
        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Reference model setup
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.model)
            # Ensure ref_model is on the same device as model
            if self.model.device.type == 'cuda' or (hasattr(self.model, 'hf_device_map') and self.model.hf_device_map):
                self.ref_model = self.ref_model.to(self.model.device)
            else: # For CPU/MPS, explicitly move
                self.ref_model = self.ref_model.to(self.device)

    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model and reward models."""
        # This function is now mainly for ensuring pad_token_id consistency if tokenizer was passed via kwargs
        # The primary tokenizer loading is now in __init__ for the resizing logic.
        if self.processing_class is None: # Fallback if for some reason not set in __init__
            warnings.warn("Tokenizer not initialized in __init__. Attempting to load from model name.")
            model_path_or_name = self.model_name if isinstance(self.model_name, str) else self.model.config._name_or_path
            self.processing_class = AutoTokenizer.from_pretrained(
                model_path_or_name, padding_side="left"
            )
            
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id
            if self.model and hasattr(self.model.config, 'pad_token_id'):
                self.model.config.pad_token_id = self.processing_class.pad_token_id


    def _initialize_metrics(self):
        """Initialize metrics tracking for training and evaluation."""
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        if not self.use_vllm: # Total train tokens only relevant for HF training
            self._total_train_tokens = 0

    def _initialize_generation_config(self):
        """Initialize the generation configuration."""
        if self.use_vllm: return # vLLM uses its own SamplingParams

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

    def _process_inputs(self, inputs: Any, with_template: bool = True, for_training: bool = False):
        """
        Processes raw inputs into tokenized tensors or templated strings.

        Args:
            inputs: Raw input data (list of dicts, dict, or string).
            with_template: Whether to apply chat template.
            for_training: Special handling for training data (e.g., replicating prompts for num_generations).

        Returns:
            Tokenized inputs (transformers.tokenization_utils_base.BatchEncoding)
            or list of templated strings if use_vllm and not for_training.
        """
        # Ensure inputs is a list of dictionaries/strings for consistent processing
        if hasattr(inputs, "to_dict"): # Assuming it's a batch object
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]
        elif isinstance(inputs, str): # Handle single string input
            inputs = [inputs]
        # If inputs is already a list (e.g., list of strings/dicts), proceed

        templated_items = []

        for item in inputs:
            text_content = ""
            if isinstance(item, str):
                text_content = item
            elif isinstance(item, dict):
                # Prioritize 'prompt' then 'content'
                text_content = item.get("prompt") or item.get("content")
                if text_content is None:
                    raise ValueError(f"Input dictionary must have a 'prompt' or 'content' key. Found: {item.keys()}")
            else:
                raise TypeError(f"Unsupported input type for chat template processing: {type(item)}")
            
            # If text_content itself is a list (e.g., from a multi-turn dialogue within a single item)
            if isinstance(text_content, list):
                # Assuming simple concatenation for now, more complex chat history needs dedicated handling
                # if the apply_chat_template expects specific roles/formats.
                # Here we assume the single 'user' role is sufficient for templating based on text_content.
                text_content = "\n".join(map(str, text_content))
            
            # Apply chat template if required (for new prompts)
            if with_template:
                chat_history = [{"role": "user", "content": text_content}] # This is a list of dictionaries
                
                # --- FIX: Directly use self.processing_class.apply_chat_template ---
                # This bypasses the problematic trl.data_utils.apply_chat_template version
                # The tokenizer's method typically expects a list of dicts for chat_history
                # and supports add_generation_prompt and tokenize args.
                try:
                    formatted_prompt = self.processing_class.apply_chat_template(
                        chat_history, 
                        tokenize=False, # We usually want string output from templating
                        add_generation_prompt=True # Re-add if tokenizer supports it
                    )
                except TypeError as e:
                    if "'tokenize'" in str(e) or "'add_generation_prompt'" in str(e):
                        warnings.warn(f"Tokenizer's apply_chat_template does not support 'tokenize' or 'add_generation_prompt'. Trying without. Error: {e}")
                        try:
                            # Fallback if specific args are not supported by the tokenizer's method
                            formatted_prompt = self.processing_class.apply_chat_template(chat_history)
                        except Exception as inner_e:
                            raise TypeError(f"Tokenizer's apply_chat_template failed even with minimal args. Check your transformers version or chat_history format. Error: {inner_e}")
                    else:
                        raise # Re-raise if it's a different TypeError

                # The tokenizer's apply_chat_template returns a string, not a dict with a 'prompt' key
                # So no ["prompt"] needed if using tokenizer.apply_chat_template directly
                templated_items.append(formatted_prompt)
            else:
                # If no templating, assume text_content is the direct input to be tokenized/used
                templated_items.append(text_content)
        
        # For vLLM, we typically pass a list of strings directly
        if self.use_vllm and not for_training:
            return templated_items
        
        # For Hugging Face models, tokenize and return tensors
        # Note: 'for_training' specific replication logic is now primarily handled in the 'step' method
        # for more fine-grained control over prompt/completion separation.
        input_tokens = self.processing_class(
            text=templated_items, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens


    def generate(
        self, inputs: Any, return_completion_ids: bool = False, stage=0
    ) -> Any:
        """
        Generate outputs from the model for the given inputs.

        Args:
            inputs: Input data for generation (raw prompts).
            return_completion_ids: Whether to return completion IDs along with text.
            stage: Current stage (not directly used in generation, but part of interface).

        Returns:
            Generated outputs (list of lists of strings) or (list of lists of strings, list of lists of token IDs).
        """
        # _process_inputs here prepares prompts for generation (not for training loss)
        input_tokens_or_prompts = self._process_inputs(inputs, with_template=True, for_training=False)

        if self.use_vllm:
            if not hasattr(self, "vllm_engine"):
                raise RuntimeError("Trainer is not initialized for vLLM inference.")

            # vLLM expects a list of strings
            vllm_outputs = self.vllm_engine.generate(input_tokens_or_prompts, self.vllm_sampling_params)

            rollout = [[completion.text for completion in output.outputs] for output in vllm_outputs]

            if return_completion_ids:
                rollout_ids = [[completion.token_ids for completion in output.outputs] for output in vllm_outputs]
                return rollout, rollout_ids
            return rollout
        else:
            # Hugging Face generation
            input_tokens = input_tokens_or_prompts # This is already a BatchEncoding

            rollout = [[] for _ in range(input_tokens.input_ids.size(0))] # Initialize with lists for each input
            rollout_ids = [[] for _ in range(input_tokens.input_ids.size(0))]

            for _ in range(self.num_generations):
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_tokens.input_ids.to(self.model.device),
                        attention_mask=input_tokens.attention_mask.to(self.model.device),
                        generation_config=self.generation_config,
                    )
                
                prompt_length = input_tokens.input_ids.size(1)
                completion_ids_batch = outputs[:, prompt_length:] # Extract completion IDs
                completions_batch = self.processing_class.batch_decode(
                    completion_ids_batch, skip_special_tokens=True
                )

                for idx in range(len(completions_batch)):
                    rollout[idx].append(completions_batch[idx])
                    if return_completion_ids:
                        rollout_ids[idx].append(completion_ids_batch[idx]) # Store tensor for IDs

            if return_completion_ids:
                return rollout, rollout_ids
            else:
                return rollout

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep=None):
        """Get the per-token log probabilities for the input tokens.

        Args:
            model: The model to compute log probabilities for.
            input_ids: The input token IDs (full sequence of prompt + completion).
            attention_mask: The attention mask (full sequence).
            logits_to_keep: The number of logits to keep, corresponding to the completion length.
                            If None, calculates over the entire sequence starting from the second token.

        Returns:
            The per-token log probabilities (shape: batch_size, logits_to_keep).
        """
        # Ensure model is on the correct device for this call
        model = model.to(input_ids.device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits # (Batch_size, Sequence_length, Vocab_size)
        
        # Shift logits to align with labels for next token prediction
        # Logits for token_i predict token_{i+1}
        logits = logits[:, :-1, :] # (Batch_size, Sequence_length - 1, Vocab_size)
        
        # Labels are input_ids shifted by one, meaning token_{i+1} is the label for token_i's prediction
        labels = input_ids[:, 1:] # (Batch_size, Sequence_length - 1)
        loss_mask = attention_mask[:, 1:] # (Batch_size, Sequence_length - 1)

        # Apply logits_to_keep for completion part only
        if logits_to_keep is not None:
            if logits_to_keep == 0: # Handle case where completion is empty
                # If no completion tokens, return an empty tensor of correct batch size
                return torch.empty(input_ids.shape[0], 0, device=input_ids.device, dtype=torch.float32)

            logits = logits[:, -logits_to_to_keep:] # (Batch_size, logits_to_keep, Vocab_size)
            labels = labels[:, -logits_to_keep:] # (Batch_size, logits_to_keep)
            loss_mask = loss_mask[:, -logits_to_keep:] # (Batch_size, logits_to_keep)


        # Divide logits by sampling temperature.
        logits = logits / self.args.temperature
        
        # Calculate cross-entropy and reshape to per-token log probabilities
        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), # Flatten logits (all tokens, all vocab)
            labels.reshape(-1), # Flatten labels (all tokens)
            reduction="none",
        ).reshape(logits.shape[0], logits.shape[1]) # Reshape back to (Batch_size, Sliced_sequence_length)

        # Apply loss mask
        # Set log-probs of padded tokens to a very small number (effectively -infinity)
        token_log_probs = (
            token_log_probs * loss_mask
            + (1.0 - loss_mask) * torch.finfo(token_log_probs.dtype).min
        )
        return token_log_probs

    def compute_loss(
        self, model, inputs, num_items_in_batch=1, mode="train", return_metrics=False
    ):
        """Compute the GRPO loss.

        Args:
            model: The model to compute the loss for.
            inputs: The inputs containing prompt_ids, prompt_mask, completion_ids, completion_mask,
                    old_per_token_logps, ref_per_token_logps, and advantages.

        Returns:
            The loss value and metrics.
        """

        # Extract inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        # Concatenate prompt and completion
        # The first dimension (batch size) MUST match for concatenation along dim=1
        if prompt_ids.shape[0] != completion_ids.shape[0]:
            raise RuntimeError(f"Batch size mismatch for prompt_ids ({prompt_ids.shape[0]}) and completion_ids ({completion_ids.shape[0]}) before concatenation in compute_loss.")
        if prompt_mask.shape[0] != completion_mask.shape[0]:
            raise RuntimeError(f"Batch size mismatch for prompt_mask ({prompt_mask.shape[0]}) and completion_mask ({completion_mask.shape[0]}) before concatenation in compute_loss.")


        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.model.device
        )
        
        # We only need to compute the logits for the completion tokens to calculate loss
        logits_to_keep = completion_ids.size(1) 

        # Compute per-token log probabilities
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute KL divergence between model and reference model if beta > 0
        mean_kl = None
        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
            else:
                # Fallback if ref_model is None but beta is set. Should ideally not happen.
                warnings.warn("beta is non-zero but reference model is not initialized. Using policy log-probs as reference.")
                ref_per_token_logps = per_token_logps.clone().detach()

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            # Calculate mean_kl, handling division by zero for empty completions
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum() if completion_mask.sum() > 0 else torch.tensor(0.0, device=per_token_kl.device, dtype=per_token_kl.dtype)
            self._metrics[mode]["kl"].append(mean_kl.item())


        # Compute the loss
        advantages = inputs["advantages"] # This should be 1D (flattened across batch and generations)
        
        # old_per_token_logps should have the same shape as per_token_logps and align with advantages
        # It represents log-probs of actions under the *old* policy for comparison.
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if (self.args.num_iterations > 1 and "old_per_token_logps" in inputs and inputs["old_per_token_logps"] is not None) 
            else per_token_logps.detach().clone() # Detach and clone to ensure no gradient flows here
        )
        
        # Advantages needs to be applied per completion, so expand it
        # (batch_size * num_generations, 1) to broadcast correctly.
        advantages_expanded = advantages.unsqueeze(dim=-1) # (B*N, 1)

        # Calculate ratios and loss terms
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.epsilon,
            1 + self.epsilon_high if self.epsilon_high is not None else (1 + self.epsilon), # Corrected for proper upper bound symmetry
        )
        
        per_token_loss1 = coef_1 * advantages_expanded
        per_token_loss2 = coef_2 * advantages_expanded
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Final loss calculation, handling division by zero for empty masks
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum() if completion_mask.sum() > 0 else torch.tensor(0.0, device=per_token_loss.device, dtype=per_token_loss.dtype)


        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum() if completion_mask.sum() > 0 else torch.tensor(0.0, device=is_clipped.device, dtype=is_clipped.dtype)
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        # return for tensorboard
        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if mean_kl is not None else 0.0,
            "clip_ratio": clip_ratio.item(),
        }

        if return_metrics:
            return loss, metrics
        else:
            return loss

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """
        Train the model using the given game state and reward manager.

        Args:
            state: The current game state.
            data_manager: Manages data for the game state.
            reward_manager: The reward manager to use for computing rewards.
        """
        if self.use_vllm:
            print("Trainer is in vLLM inference mode. Skipping training step.")
            return # Training is not supported in vLLM mode

        self.model.train()
        global_step = self.global_step
        # Loop through stages to ensure all data is processed.
        # The 'state.stage' will typically be 1 for a single-stage GRPO training.
        for stage in range(state.stage):
            global_step = self.step(
                stage, state, data_manager, reward_manager, global_step
            )
        self.global_step = global_step
        self.model.eval()

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        global_step += 1

        # Prepare stage's inputs
        # stage_inputs_raw_batch will be a list of original prompts (e.g., [{'prompt': 'Write a story...'}])
        stage_inputs_raw_batch = state.get_stage_state(stage) 
        # index_mapping maps the flattened data back to original GameState structure
        stage_inputs_raw_batch, index_mapping = data_manager.prepare_input(stage_inputs_raw_batch, stage)
        assert stage_inputs_raw_batch is not None, f"No inputs found for stage {stage}"
        
        # stage_actions_raw will be a nested list like [[[completion1, completion2], [comp3, comp4]], ...]
        stage_actions_raw_batch = state.get_stage_actions(stage)

        # Extract outputs based on index_mapping
        # This will contain (num_original_prompts * num_generations) individual completion strings/tensors
        stage_outputs_flat = [
            stage_actions_raw_batch[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        assert stage_outputs_flat is not None, f"No outputs found for stage {stage}"
        # Ensure that len(stage_outputs_flat) == len(index_mapping) and == len(stage_inputs_raw_batch) * self.num_generations

        # Combine prompts and completions for consistent tokenization and padding
        combined_texts = []
        original_prompt_actual_lengths = [] 

        # We need to loop through the original inputs to get their formatted prompts
        # and then pair them with *all* their corresponding generations.
        # This loop runs `len(stage_inputs_raw_batch)` times.
        for i, prompt_item in enumerate(stage_inputs_raw_batch):
            # Get the formatted prompt string using chat template
            # --- FIX: Removed 'add_generation_prompt=True' and `tokenize=False` from `_process_inputs` call
            # Now, using tokenizer's apply_chat_template directly
            formatted_prompt_str = self.processing_class.apply_chat_template(
                [{"role": "user", "content": prompt_item.get("prompt") or prompt_item.get("content") or prompt_item}], # Ensure content is from item or default to empty
                tokenize=False, # We want the string output
                add_generation_prompt=True # Re-add if tokenizer supports it
            )
            
            # Tokenize this single prompt temporarily to get its actual length before padding
            temp_prompt_tokens = self.processing_class(
                text=[formatted_prompt_str], return_tensors="pt", padding=False, truncation=True
            )
            current_prompt_len = temp_prompt_tokens.input_ids.size(1)
            original_prompt_actual_lengths.append(current_prompt_len)

            # Get the slice of completions for this specific prompt from the flattened list
            # This is `self.num_generations` completions for the current prompt.
            start_idx = i * self.num_generations
            end_idx = start_idx + self.num_generations
            completions_for_this_prompt_slice = stage_outputs_flat[start_idx:end_idx]

            # Replicate and combine for each generation
            # This loop runs `self.num_generations` times.
            for j in range(self.num_generations):
                completion_str = ""
                if j < len(completions_for_this_prompt_slice):
                    current_completion = completions_for_this_prompt_slice[j]
                    if isinstance(current_completion, list): # Handle cases like vLLM returning list of token_ids
                        warnings.warn("Completion is a list. Taking the first element as string. Ensure this is intended for reward processing.")
                        completion_str = str(current_completion[0]) 
                    else:
                        completion_str = str(current_completion) 
                else:
                    warnings.warn(f"Not enough completions for prompt {i} (gen {j+1}/{self.num_generations}). Using empty string for extra generations.")
                    completion_str = "" 
                
                # Append the combined string (formatted prompt + completion)
                combined_texts.append(formatted_prompt_str + completion_str)
        
        # --- ASSERTION FOR BATCH SIZE CONSISTENCY ---
        expected_combined_texts_len = len(stage_inputs_raw_batch) * self.num_generations
        if len(combined_texts) != expected_combined_texts_len:
            raise RuntimeError(f"Mismatch in combined_texts length. Expected {expected_combined_texts_len}, got {len(combined_texts)}. This indicates an indexing issue upstream in data_manager or state. Number of original inputs: {len(stage_inputs_raw_batch)}, num_generations: {self.num_generations}")
        # --- END ASSERTION ---


        # Tokenize all combined prompt+completion sequences at once with batch padding
        tokenized_combined = self.processing_class(
            text=combined_texts, return_tensors="pt", padding=True, truncation=True
        )

        input_ids_full = tokenized_combined.input_ids.to(self.model.device)
        attention_mask_full = tokenized_combined.attention_mask.to(self.model.device)

        model_inputs = {}
        batch_size_flat_from_combined = input_ids_full.shape[0] # This should be len(stage_inputs_raw_batch) * self.num_generations

        prompt_ids_list = []
        prompt_mask_list = []
        completion_ids_list = []
        completion_mask_list = []
        
        # Loop over the flattened combined batch to split into prompts and completions
        for i in range(batch_size_flat_from_combined):
            # This maps the flat index `i` back to the original prompt's index
            original_prompt_batch_idx_for_current_flat_item = i // self.num_generations
            original_prompt_len_for_this_seq = original_prompt_actual_lengths[original_prompt_batch_idx_for_current_flat_item]

            # Slice out the prompt and completion parts
            prompt_part_ids = input_ids_full[i, :original_prompt_len_for_this_seq]
            prompt_part_mask = attention_mask_full[i, :original_prompt_len_for_this_seq]
            
            completion_part_ids = input_ids_full[i, original_prompt_len_for_this_seq:]
            completion_part_mask = attention_mask_full[i, original_prompt_len_for_this_seq:]
            
            # --- FIX: Ensure completion_ids are never completely empty (0-length) after slicing if there was content ---
            # This is the most likely culprit if `torch.stack` has issues with 0-length tensors in list
            # If the completion_part_ids is empty AFTER slicing (e.g., prompt fills max_seq_len),
            # or if it results in a [0] element tensor, ensure we handle it gracefully for stacking.
            if completion_part_ids.numel() == 0 and original_prompt_len_for_this_seq < input_ids_full.size(1):
                # Only add a placeholder if the entire sequence was not just the prompt
                # and the completion part is indeed empty.
                if self.processing_class.pad_token_id is not None:
                    completion_part_ids = torch.tensor([self.processing_class.pad_token_id], device=self.model.device, dtype=input_ids_full.dtype)
                    completion_part_mask = torch.tensor([0], device=self.model.device, dtype=attention_mask_full.dtype)
                else: # Fallback if no pad_token_id, try to use 0 as a placeholder
                    completion_part_ids = torch.tensor([0], device=self.model.device, dtype=input_ids_full.dtype)
                    completion_part_mask = torch.tensor([0], device=self.model.device, dtype=attention_mask_full.dtype)
            # --- END FIX ---


            prompt_ids_list.append(prompt_part_ids)
            prompt_mask_list.append(prompt_part_mask)
            completion_ids_list.append(completion_part_ids)
            completion_mask_list.append(completion_part_mask)

        # Pad individual prompt and completion tensors to their max lengths within the current batch
        max_prompt_len_in_batch = max([t.size(0) for t in prompt_ids_list]) if prompt_ids_list else 0
        max_completion_len_in_batch = max([t.size(0) for t in completion_ids_list]) if completion_ids_list else 0

        pad_id = self.processing_class.pad_token_id
        if pad_id is None:
             warnings.warn("pad_token_id is None. Using 0 for padding. Ensure tokenizer is properly configured.")
             pad_id = 0

        # Stack and pad prompt tensors to a uniform shape
        model_inputs["prompt_ids"] = torch.stack([
            torch.cat([p_ids, torch.full((max_prompt_len_in_batch - p_ids.size(0),), pad_id, dtype=p_ids.dtype).to(self.model.device)])
            for p_ids in prompt_ids_list
        ]).to(self.model.device)
        model_inputs["prompt_mask"] = torch.stack([
            torch.cat([p_mask, torch.zeros((max_prompt_len_in_batch - p_mask.size(0),), dtype=p_mask.dtype).to(self.model.device)])
            for p_mask in prompt_mask_list
        ]).to(self.model.device)

        # Stack and pad completion tensors to a uniform shape
        model_inputs["completion_ids"] = torch.stack([
            torch.cat([c_ids, torch.full((max_completion_len_in_batch - c_ids.size(0),), pad_id, dtype=c_ids.dtype).to(self.model.device)])
            for c_ids in completion_ids_list
        ]).to(self.model.device)
        model_inputs["completion_mask"] = torch.stack([
            torch.cat([c_mask, torch.zeros((max_completion_len_in_batch - c_mask.size(0),), dtype=c_mask.dtype).to(self.model.device)])
            for c_mask in completion_mask_list
        ]).to(self.model.device)

        # --- FINAL ASSERTION OF BATCH SIZE CONSISTENCY BEFORE COMPUTING LOSS ---
        if model_inputs["prompt_ids"].shape[0] != model_inputs["completion_ids"].shape[0]:
            raise RuntimeError(f"FATAL: Final batch size mismatch between prompts ({model_inputs['prompt_ids'].shape[0]}) and completions ({model_inputs['completion_ids'].shape[0]}) after tokenization and padding in step method. This is a critical internal error.")
        # --- END FINAL ASSERTION ---


        # Process rewards with shape fix and participation bonus
        rewards_from_manager = reward_manager[stage]
        
        rewards_list_for_tensor = []
        final_rewards_structured = [[] for _ in range(len(stage_inputs_raw_batch))]
        
        for flat_idx, (original_input_idx, sub_item_idx, generation_idx) in enumerate(index_mapping):
            reward_val = rewards_from_manager[stage][original_input_idx][sub_item_idx][generation_idx]

            if isinstance(reward_val, list):
                if len(reward_val) > 0:
                    reward_val = reward_val[0]
                else:
                    reward_val = 0.0
                    warnings.warn(f"Empty list as reward for prompt {original_input_idx}, sub_item {sub_item_idx}, gen {generation_idx}. Using 0.0.")
            elif not isinstance(reward_val, (int, float)):
                warnings.warn(f"Unexpected reward type ({type(reward_val)}) for prompt {original_input_idx}, sub_item {sub_item_idx}, gen {generation_idx}. Using 0.0.")
                reward_val = 0.0

            base_reward = 1.0
            perf_reward = max(0, float(reward_val))
            
            final_rewards_structured[original_input_idx].append(base_reward + perf_reward)

        for i in range(len(final_rewards_structured)):
            while len(final_rewards_structured[i]) < self.num_generations:
                final_rewards_structured[i].append(1.0)

        if not final_rewards_structured:
            warnings.warn("No rewards processed. Skipping loss calculation for this step.")
            return global_step 

        rewards = torch.tensor(final_rewards_structured, dtype=torch.float32).to(self.model.device) 

        assert rewards is not None, f"Rewards tensor is None after processing for stage {stage}"
        assert rewards.dim() == 2 and rewards.size(1) == self.num_generations, \
            f"Rewards tensor shape {rewards.shape} does not match expected [num_original_prompts, {self.num_generations}]. Found {rewards.shape}."

        with torch.no_grad():
            mean_rewards_per_prompt = rewards.mean(dim=1, keepdim=True)
            std_rewards_per_prompt = rewards.std(dim=1, keepdim=True)

            advantages_unflattened = rewards - mean_rewards_per_prompt

            zero_std_mask = std_rewards_per_prompt < 1e-8
            advantages_unflattened[~zero_std_mask] /= (std_rewards_per_prompt[~zero_std_mask] + 1e-8)

        advantages = torch.flatten(advantages_unflattened).to(self.model.device)
        model_inputs["advantages"] = advantages
        
        model_inputs["old_per_token_logps"] = None 

        with self.autocast:
            loss, metrics = self.compute_loss(self.model, model_inputs, return_metrics=True)

        if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() != 0.0:
            loss.backward()
            
            max_grad_norm = getattr(self.args, "max_grad_norm", 1.0) 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

            self.optimizer.step()
            self.model.zero_grad()
        else:
            warnings.warn(f"Skipping backward pass due to invalid loss (NaN/Inf/0.0): {loss.item()}")
            metrics["loss"] = 0.0 
            metrics["kl"] = metrics.get("kl", 0.0) 
            metrics["clip_ratio"] = metrics.get("clip_ratio", 0.0) 

        self.log({"train/loss": metrics["loss"], "train/rewards": rewards.cpu().mean().item()}, global_step)
        if metrics["kl"] is not None: 
            self.log({"train/kl": metrics["kl"]}, global_step)
        self.log({"train/clip_ratio": metrics["clip_ratio"]}, global_step)


        self.cleanup_step()
        return global_step

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        """
        Evaluate the model's performance on the given game state.
        This method is currently a placeholder.
        """
        print("Evaluation method is not yet implemented for GRPOLanguageTrainerModule.")
        pass

    def save(self, save_dir: str) -> None:
        """
        Save the model and trainer state to the given directory.

        Args:
            save_dir: The directory to save to.
        """
        if self.use_vllm:
            print("Save is not fully supported for the HuggingFace model when using vLLM for inference. Only tokenizer and trainer state will be saved.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        if self.model is not None:
            try:
                self.model.save_pretrained(save_dir)
            except Exception as e:
                warnings.warn(f"Failed to save model weights: {e}. This might happen if using certain quantization configurations or if the model is not a standard HF model.")

        if self.processing_class is not None:
            self.processing_class.save_pretrained(save_dir) 

        trainer_state = {
            "metrics": self._metrics,
            "total_train_tokens": self._total_train_tokens,
            "global_step": self.global_step,
            "num_generations": self.num_generations,
            "epsilon": self.epsilon,
            "epsilon_high": self.epsilon_high,
            "beta": self.beta,
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
            "model_name": self.model_name 
        }
        if hasattr(self, 'generation_config') and self.generation_config is not None:
             trainer_state["generation_config"] = self.generation_config.to_dict()
        
        if self.optimizer is not None:
            trainer_state["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(trainer_state, os.path.join(save_dir, "trainer_state.pt"))
        print(f"Trainer state saved to {save_dir}")


    @classmethod
    def load(cls, load_dir: str) -> "GRPOLanguageTrainerModule":
        """
        Load a trainer module from the given directory.

        Args:
            load_dir: The directory to load from.

        Returns:
            The loaded trainer module.
        """
        trainer_state_path = os.path.join(load_dir, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            raise FileNotFoundError(f"Trainer state file not found at {trainer_state_path}. Cannot load trainer.")
        
        trainer_state = torch.load(trainer_state_path, map_location="cpu") 
        
        model_name_for_load = trainer_state.get("model_name")
        if model_name_for_load is None:
            warnings.warn("Model name not found in trainer_state. Attempting to load model from directory directly.")
            model_name_for_load = load_dir 

        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(load_dir, padding_side="left")
            if loaded_tokenizer.pad_token_id is None:
                loaded_tokenizer.pad_token_id = loaded_tokenizer.eos_token_id
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {load_dir}: {e}")

        loaded_model = None
        try:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) 
            loaded_model = AutoModelForCausalLM.from_pretrained(load_dir, quantization_config=quant_config, device_map="auto")
            print("Attempting to load model with 4-bit quantization during load.")
        except ImportError:
            print("bitsandbytes not installed. Loading model without 4-bit quantization.")
            loaded_model = AutoModelForCausalLM.from_pretrained(load_dir)
        except Exception as e:
            warnings.warn(f"Failed to load quantized model: {e}. Falling back to non-quantized load.")
            loaded_model = AutoModelForCausalLM.from_pretrained(load_dir)
        
        # --- FIX: Resize token embeddings to match tokenizer vocabulary during LOAD ---
        if loaded_model.get_input_embeddings().weight.shape[0] != len(loaded_tokenizer):
            old_vocab_size = loaded_model.get_input_embeddings().weight.shape[0]
            new_vocab_size = len(loaded_tokenizer)
            print(f"🔄 Resizing loaded model token embeddings from {old_vocab_size} to {new_vocab_size} to match loaded tokenizer vocabulary.")
            loaded_model.resize_token_embeddings(new_vocab_size)
            if loaded_tokenizer.pad_token_id is not None and hasattr(loaded_model.config, 'pad_token_id'):
                loaded_model.config.pad_token_id = loaded_tokenizer.pad_token_id
        # --- END FIX ---


        trainer = cls(
            models=[loaded_model], 
            processing_class=loaded_tokenizer,
            num_generations=trainer_state.get("num_generations", 2),
            epsilon=trainer_state.get("epsilon", 0.2),
            epsilon_high=trainer_state.get("epsilon_high", 0.28),
            beta=trainer_state.get("beta", 0.0),
            enable_gradient_checkpointing=trainer_state.get("enable_gradient_checkpointing", True),
            log_dir=load_dir, 
            log_with=trainer_state.get("log_with"), 
            config=GRPOConfig(trainer_state.get("args", {})) if "args" in trainer_state else None 
        )

        trainer._metrics = trainer_state.get("metrics", defaultdict(list))
        trainer._total_train_tokens = trainer_state.get("total_train_tokens", 0)
        trainer.global_step = trainer_state.get("global_step", 0)

        gen_config_dict = trainer_state.get("generation_config")
        if gen_config_dict:
            trainer.generation_config = GenerationConfig.from_dict(gen_config_dict)
        else:
            trainer._initialize_generation_config()

        optimizer_state_dict = trainer_state.get("optimizer_state_dict")
        if optimizer_state_dict:
            if trainer.optimizer is None:
                trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=trainer.args.learning_rate)
            trainer.optimizer.load_state_dict(optimizer_state_dict)
        
        if trainer.device.type == 'cuda' and not (hasattr(trainer.model, 'hf_device_map') and trainer.model.hf_device_map):
            trainer.model.to(trainer.device)
        elif trainer.device.type == 'mps':
            trainer.model.to(trainer.device)

        if trainer.beta != 0.0 and trainer.ref_model is None: 
            trainer.ref_model = create_reference_model(trainer.model)
            if trainer.model.device.type == 'cuda' or (hasattr(trainer.model, 'hf_device_map') and trainer.model.hf_device_map):
                trainer.ref_model = trainer.ref_model.to(trainer.model.device)
            else:
                trainer.ref_model = trainer.ref_model.to(trainer.device)

        print(f"Trainer loaded from {load_dir}. Global step: {trainer.global_step}")
        return trainer

    def cleanup_step(self):
        """Cleans up GPU memory and Python garbage collector after each step."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def cleanup(self):
        """Performs final cleanup, including shutting down loggers."""
        self.cleanup_trackers() 
        print("Trainer cleanup complete.")
