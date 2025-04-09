# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


import json
import math
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pathlib
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
import datasets
import torch
import warnings
import transformers
import huggingface_hub.utils as hf_hub_utils
import numpy as np
from accelerate import skip_first_batches
from packaging import version
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    BitsAndBytesConfig,
    Trainer,
    deepspeed, Qwen2Model, is_datasets_available, is_torch_xla_available, Qwen2Config, is_apex_available, TrainerState,
    PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.debug_utils import DebugOption
from transformers.integrations import deepspeed_init, deepspeed_load_checkpoint, hp_params
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.trainer_callback import ExportableState
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, denumpify_detensorize, has_length, speed_metrics, \
    TrainOutput, HPSearchBackend, find_executable_batch_size, enable_full_determinism, get_last_checkpoint, set_seed
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import logging, is_sagemaker_mp_enabled, XLA_FSDPV2_MIN_VERSION, is_accelerate_available, \
    is_peft_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME

logger = logging.get_logger(__name__)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model, TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother, EvalLoopContainer, find_batch_size, IterableDatasetShard, \
    nested_detach, get_model_param_count

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
import torch.distributed as dist
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    label_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return




def preprocess(
        messages,
        users,
        items,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    user_li = []
    item_li = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    for i, user in enumerate(users):
        user_li.append(int(user))
    for i, item in enumerate(items):
        item_li.append(int(item))
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask, users=user_li, items=item_li
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = preprocess([self.raw_data[i]["messages"]], [self.raw_data[i]["source"]], [self.raw_data[i]["source_item"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["target_ids"][0],
            attention_mask=ret["attention_mask"][0],
            user_id=ret["users"],
            item_id=ret["items"]
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_data = []
    with open(data_args.data_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))

    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_data = []
        with open(data_args.eval_data_path, "r") as f:
            for line in f:
                eval_data.append(json.loads(line))
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

if is_apex_available():
    from apex import amp
import safetensors.torch
class Comtrainer(Trainer):

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            count = 0
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            )
                        ).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
     
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
                hasattr(self, "_eval_dataloaders")
                and dataloader_key in self._eval_dataloaders
                and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator
        self.user_list = []
        for i in eval_dataset:
            user = i['user_id'][0]
            self.user_list.append(int(user))
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")
        print(self._signature_columns)
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        # for i in eval_dataloader:
        #     print(i)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        # if has_labels or loss_without_labels:
        #     labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        #     if len(labels) == 1:
        #         labels = labels[0]
        # else:
        #     labels = None
        labels = inputs['labels']
        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # for i in eval_dataloader:
        #     print(i)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)
        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    if args.include_inputs_for_metrics:
                        metrics = self.compute_metrics(
                            EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs),
                            compute_result=is_last_step,
                        )
                    else:
                        metrics = self.compute_metrics(
                            EvalPrediction(predictions=logits, label_ids=labels),
                            compute_result=is_last_step,
                        )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
                self.compute_metrics is not None
                and all_preds is not None
                and all_labels is not None
                and not self.args.batch_eval_metrics
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels),
                                               self.user_list)
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def uAUC_me(user, predict, label):
    if not isinstance(predict, np.ndarray):
        predict = np.array(predict)
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)  # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]

        k += 1
    print("only one interaction users:", only_one_interaction)
    auc = []
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i, pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")

    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time() - start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user


def compute_metrics(eval_preds, user_list):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    uauc, _, _ = uAUC_me(user_list, list(pre[0]), list(pre[1]))
    return {'auc': auc, 'uauc': uauc}


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """

    labels_index = torch.argwhere(torch.bitwise_or(labels == 9454, labels == 2753))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 2753, 0, 1)
    labels_index[:, 1] = labels_index[:, 1] - 1
    logits = logits.softmax(dim=-1)
    logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:, [2753, 9454]], dim=-1)
    return logits[:, 1][2::3], gold[2::3]



class ComQwenCausalLLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = ComQwen(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            user_id: torch.LongTensor = None,
            item_id: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        # >>> from transformers import AutoTokenizer, Qwen2ForCausalLM
        #
        # >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        # >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        #
        # >>> prompt = "Hey, are you conscious? Can you talk to me?"
        # >>> inputs = tokenizer(prompt, return_tensors="pt")
        #
        # >>> # Generate
        # >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        # >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            user_id=user_id,
            item_id=item_id
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LinearProjection(nn.Module):
    def __init__(self, small_emb_dim, big_model_dim, small_model=None):
        super().__init__()
        self.linear_small2big_1 = nn.Linear(small_emb_dim, 1024)
        self.linear_small2big_2 = nn.Linear(1024, big_model_dim)
        if small_model!=None:
            self.linear_small2big_1.load_state_dict(small_model.trans_1.state_dict())
            self.linear_small2big_2.load_state_dict(small_model.trans_2.state_dict())
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.small2big1(x)
        x = self.gelu(x)
        x = self.small2big2(x)
        return x

import torch.nn.functional as F
class Big_LinearProjection(nn.Module):
    def __init__(self, big_model_dim, big_linear=None):
        super().__init__()
        self.linear_big2big = nn.Linear(big_model_dim, big_model_dim)
        if big_linear !=None:
            self.linear_big2big.load_state_dict(big_linear.linear_big2big.state_dict())
    def forward(self, x):
        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = self.linear_big2big(x)
        return x
    
class ComQwen(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.projection_model = None
        self.user_token_id = None
        self.item_token_id = None
        self.warm_token_id = None
        self.small_model = None
        self.big_linear_model = None
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.gelu = nn.GELU()
    def prepare_soft_prompt(self, soft_prompt_token_id):
        soft_prompt_num = len(soft_prompt_token_id)
        emb_dim = self.get_input_embeddings().embedding_dim
        self.soft_prompt_emb = nn.Parameter(torch.randn(soft_prompt_num, emb_dim))
        self.soft_prompt_token_id = soft_prompt_token_id

    def prepare_collm_prompt(self, user_token_id, item_token_id, warm_token_id, small_emb_dim, projection_model_path=None, big_linear_path=None,  pretrained_with_small=False):
        emb_dim = self.get_input_embeddings().embedding_dim
        self.user_token_id = user_token_id
        self.item_token_id = item_token_id
        self.warm_token_id = warm_token_id
        if projection_model_path == None:
            if pretrained_with_small == False:
                self.projection_model = LinearProjection(small_emb_dim, emb_dim)
            else:
                self.projection_model = LinearProjection(small_emb_dim, emb_dim, self.small_model)

        else:
            projection_model = LinearProjection(small_emb_dim, emb_dim)  # 创建新的模型实例
            projection_model.load_state_dict(torch.load(projection_model_path))
            self.projection_model = projection_model
            
        if big_linear_path == None:
            self.big_linear_model = Big_LinearProjection(emb_dim)
        else:
            big_linear_model = Big_LinearProjection(emb_dim) # 创建新的模型实例
            big_linear_model.load_state_dict(torch.load(big_linear_path))
            self.big_linear_model = big_linear_model

        self.register_module("projection_model", self.projection_model)
        self.register_module("big_linear_model", self.big_linear_model)
    
    def get_small_models(self, model_type, data_type, model_path, user_num, item_num, emb_size):
        mf_config = {
            "user_num": user_num,
            "item_num": item_num,
            "embedding_size": emb_size,
            }
        import omegaconf

        mf_config = omegaconf.OmegaConf.create(mf_config)
        from baseline_train_mf_book_step1_cl import MatrixFactorization
        model = MatrixFactorization(mf_config).cuda()
        model.load_state_dict(torch.load(model_path))
        self.small_model = model
        self.register_module("small_model", self.small_model)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            user_id: torch.LongTensor = None,
            item_id: torch.LongTensor = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_attentions = False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # sep_positions = [(input_ids == token_id).nonzero(as_tuple=True)[1] for token_id in self.soft_prompt_token_id]
        sep_positions_item = (input_ids == self.item_token_id).nonzero(as_tuple=True)[1]
        sep_positions_warm = (input_ids == self.warm_token_id).nonzero(as_tuple=True)[1]
        sep_positions_user = (input_ids == self.user_token_id).nonzero(as_tuple=True)[1]
        # 获取输入的嵌入
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)
        user_embeddings = self.projection_model.linear_small2big_2(self.gelu(self.projection_model.linear_small2big_1(self.small_model.user_encoder(user_id))))
        item_embeddings = self.projection_model.linear_small2big_2(self.gelu(self.projection_model.linear_small2big_1(self.small_model.item_encoder(item_id))))
        warm_embeddings = self.big_linear_model(self.small_model.item_embedding_llm(item_id))
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        for i in range(input_ids.shape[0]):  # 遍历批次中的每个样本
            # user_pos = sep_positions_user[i]
            # user_emb = torch.zeros_like(item_embeddings[i])
            # history = history_list[i][0]
            # count=0
            # for j in history:
            #     if j != 0 and j!= -1:
            #         count+=1
            #         user_emb += self.small_model.item_embedding_llm(j)
            # user_emb /= count
            # user_emb = self.big_linear_model.linear_big2big(user_emb)
            warm_pos = sep_positions_warm[i]
            inputs_embeds[i, warm_pos] = warm_embeddings[i]

            user_pos = sep_positions_user[i]
            inputs_embeds[i, user_pos] = user_embeddings[i]
            
            item_pos = sep_positions_item[i]
            inputs_embeds[i, item_pos] = item_embeddings[i]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
#         from matplotlib.lines import Line2D
#         # 创建保存目录
#         # os.makedirs(save_attention_dir, exist_ok=True)
#         save_attention_layer = -1
#         # # 确定要保存的层
#         layer_to_save = save_attention_layer if save_attention_layer >= 0 else len(all_self_attns) - 1
#         layer_to_save = min(layer_to_save, len(all_self_attns) - 1)
#         # 定义要保存的层
# # 定义要保存的层
# # 定义要保存的层
#         layers_to_save = [2, 27]  # 只要第2层和第27层

#         # 创建一个公共的颜色标尺范围
#         all_attention_values = []

#         # 首先收集所有层的注意力值以确定共同的颜色范围
#         for layer_idx in layers_to_save:
#             layer_idx = min(layer_idx, len(all_self_attns) - 1)  # 确保索引有效
#             for batch_idx in range(27, 32):  # 限制为最多5个样本
#                 if batch_idx < sep_positions_user.shape[0]:
#                     layer_attn = all_self_attns[layer_idx][batch_idx].detach().cpu()
#                     avg_attention = layer_attn.mean(dim=0).float().detach().cpu().numpy()
                    
#                     user_pos = sep_positions_user[batch_idx].item()
#                     item_pos = sep_positions_item[batch_idx].item()
#                     warm_pos = sep_positions_warm[batch_idx].item()
                    
#                     positions = [user_pos, item_pos, warm_pos]
#                     min_pos = max(0, min(positions)-20)
#                     max_pos = min(avg_attention.shape[0], max(positions) + 20)
                    
#                     cropped_attention = avg_attention[min_pos:max_pos, min_pos:max_pos]
#                     all_attention_values.append(cropped_attention)

# # 计算所有注意力值的共同标准化范围
#         all_attention_array = np.concatenate([att.flatten() for att in all_attention_values])
#         vmin = 0
#         vmax = np.percentile(all_attention_array, 95)  # 使用95百分位数作为最大值

#         # 为每个样本创建子图展示
#         for batch_idx in range(1,32):  # 限制为最多5个样本
#             current_user_id = user_id[batch_idx].item() if user_id is not None else 0
#             current_item_id = item_id[batch_idx].item() if item_id is not None else 0
            
#         # 创建带有两个子图的画布，为颜色条预留空间
#             fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
#             if batch_idx < sep_positions_user.shape[0]:
#                 user_pos = sep_positions_user[batch_idx].item()
#                 item_pos = sep_positions_item[batch_idx].item()
#                 warm_pos = sep_positions_warm[batch_idx].item()
                
#                 # 计算感兴趣区域的范围，并添加一些填充
#                 positions = [user_pos, item_pos, warm_pos]
#                 min_pos = max(0, min(positions)-20)
#                 max_pos = min(all_self_attns[0][batch_idx].shape[-1], max(positions) + 20)
                
#                 # 设置共同的颜色标尺
#                 norm = plt.Normalize(vmin=vmin, vmax=vmax)
                
#                 # 使用更高对比度的霓虹色
#                 user_color = '#8A2BE2'  # 紫色
#                 item_color = '#00008B'  # 深蓝色
#                 warm_color = '#FF0000'  # 红色
                
#                 line_width = 2.5  # 更粗的线
#                 box_width = 1.0  # 方框宽度
                
#                 # 处理每个层
#                 for ax_idx, layer_idx in enumerate(layers_to_save):
#                     layer_idx = min(layer_idx, len(all_self_attns) - 1)  # 确保索引有效
#                     layer_attn = all_self_attns[layer_idx][batch_idx].detach().cpu()
#                     avg_attention = layer_attn.mean(dim=0).float().detach().cpu().numpy()
                    
#                     # 裁剪注意力矩阵以仅显示感兴趣的区域
#                     cropped_attention = avg_attention[min_pos:max_pos, min_pos:max_pos]
                    
#                     # 绘制裁剪后的热力图
#                     im = axes[ax_idx].imshow(cropped_attention, cmap='viridis', norm=norm)
                    
#                     # 调整标记线的位置以匹配裁剪后的坐标
#                     user_pos_adj = user_pos - min_pos
#                     item_pos_adj = item_pos - min_pos
#                     warm_pos_adj = warm_pos - min_pos
                    
#                     # 计算矩阵尺寸
#                     matrix_size = cropped_attention.shape[0]
                    
#                     # 为每个位置创建列标记
#                     for x_pos, color in [(user_pos_adj, user_color), 
#                                     (item_pos_adj, item_color), 
#                                     (warm_pos_adj, warm_color)]:
#                         # 提前一格的箭头位置
#                         arrow_pos = max(0, x_pos - 1)
                        
#                         # 方框左右位置
#                         x_left = x_pos - box_width/2
#                         x_right = x_pos + box_width/2
                        
#                         # 提前一格的方框顶部位置
#                         y_top = max(0, x_pos)
                        
#                         # 绘制方框 - 填充为无，只有边框
#                         rect = plt.Rectangle((x_left, y_top), box_width, matrix_size-y_top-1, 
#                                         fill=False, edgecolor=color, linewidth=line_width, alpha=1.0)
#                         axes[ax_idx].add_patch(rect)
                        
#                         # 添加白色边缘发光效果增强可见性
#                         glow = plt.Rectangle((x_left, y_top), box_width, matrix_size-y_top-1, 
#                                         fill=False, edgecolor='white', linewidth=line_width+2, alpha=0.3)
#                         axes[ax_idx].add_patch(glow)
                        
#                         # 添加箭头指示器在顶部增强可见性 - 提前一格
#                         axes[ax_idx].scatter([x_pos], [arrow_pos], color=color, s=100, marker='v', 
#                                 edgecolor='white', linewidth=1.5, zorder=10)
                    
#                     # 更新坐标轴标签以反映实际位置
#                     axes[ax_idx].set_xticks(np.arange(0, max_pos-min_pos, 5))
#                     axes[ax_idx].set_xticklabels(np.arange(min_pos, max_pos, 5))
#                     axes[ax_idx].set_yticks(np.arange(0, max_pos-min_pos, 5))
#                     axes[ax_idx].set_yticklabels(np.arange(min_pos, max_pos, 5))
                    
#                     axes[ax_idx].set_title(f"Layer {layer_idx} Attention", fontsize=14)
#                     axes[ax_idx].set_xlabel("Attended Token Position", fontsize=12)
#                     axes[ax_idx].set_ylabel("Attending Token Position", fontsize=12)
                    
#                     # 添加与原始代码类似的图例
#                     from matplotlib.patches import Rectangle
#                     legend_width = 5  # 更宽的矩形
#                     legend_height = 1
                    
#                     # 创建自定义图例句柄
#                     legend_handles = [
#                         Rectangle((0, 0), legend_width, legend_height, fill=False, edgecolor=user_color, linewidth=line_width),
#                         Rectangle((0, 0), legend_width, legend_height, fill=False, edgecolor=item_color, linewidth=line_width),
#                         Rectangle((0, 0), legend_width, legend_height, fill=False, edgecolor=warm_color, linewidth=line_width)
#                         ]

#                     # 为每个子图创建图例，与原代码一致
#                     legend = axes[ax_idx].legend(handles=legend_handles, labels=['User', 'Item', 'Warm'], 
#                                     loc='upper right', framealpha=0.7, edgecolor='white', fontsize=13)
#                     for text, color in zip(legend.get_texts(), [user_color, item_color, warm_color]):
#                         text.set_color(color)
#                         text.set_fontweight('bold')
                
#                 # 在右侧添加共同的颜色条
#                 cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#                 cbar = fig.colorbar(im, cax=cbar_ax)
#                 cbar.set_label('Attention Strength', fontsize=12)
                
#                 # 调整子图之间的间距并添加总标题
#                 plt.suptitle(f"Attention Maps for User {current_user_id}, Item {current_item_id} (Sample {batch_idx})", fontsize=16)
#                 plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 为颜色条留出空间
                
#                 # 保存图像
#                 vis_path = os.path.join(
#                     "", 
#                     f"attention_comparison_user{current_user_id}_item{current_item_id}_sample{batch_idx}.pdf"
#                 )
#                 plt.savefig(vis_path, dpi=200, bbox_inches='tight')
#                 plt.close()
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def safe_save_model_for_hf_trainer(
        trainer: Comtrainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # 1. 获取模型的state_dict
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        
        # 这行代码从DeepSpeed包装的模型中获取16位精度的合并状态字典
        # print(state_dict)  # 打印查看state_dict的内容
        
        if state_dict!=None:
        # 2. 准备两个空字典来分别存储两个子模型的参数
            small_model_params = {}        # 存储small_model的参数
            projection_model_params = {}   # 存储projection_model的参数
            big_model_params = {}
            # 3. 定义两个前缀，用于识别参数属于哪个模型
            small_prefix = 'base_model.model.model.small_model.'
            projection_prefix = 'base_model.model.model.projection_model.'
            big_prefix = 'base_model.model.model.big_linear_model.'
            for key, value in state_dict.items():
                if key.startswith(small_prefix):
                    # 去掉前缀，只保留实际的参数名
                    new_key = key.replace(small_prefix, '')
                    small_model_params[new_key] = value

                elif key.startswith(projection_prefix):
                    new_key = key.replace(projection_prefix, '')
                    projection_model_params[new_key] = value

                elif key.startswith(big_prefix):
                    new_key = key.replace(big_prefix, '')
                    big_model_params[new_key] = value
                    
            torch.save(projection_model_params, '***path_to_projection_model***')
            torch.save(small_model_params, '***path_to_small_model***')
            torch.save(big_model_params, '***path_to_big_model***')
        # projection_params = {}
        # small_model_params = {}
        # for key, value in state_dict.items():
        #     if 'base_model.model.model.projection_model' in key:
        #         projection_params[key] = value
        #
        # for key, value in state_dict.items():
        #     if 'base_model.model.model.small_model' in key:
        #         small_model_params[key] = value
        #
        # # 保存提取的参数


    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if (
            getattr(training_args, "deepspeed", None)
            and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    label_names: List[str] = ['labels', 'input_ids', 'attention_mask', 'user_id','item_id']
    setattr(training_args, 'label_names', label_names)

    local_rank = training_args.local_rank
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config = transformers.Qwen2Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    model = ComQwenCausalLLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        #attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing,
            )

        # model = get_peft_model(model, lora_config)
        sep_tokens = ["<User_ID>", "<Item_ID>", "<Warm_ID>"]
        lora_model_path = "***path_to_lora_checkpoints***"
        special_tokens_dict = {'additional_special_tokens': sep_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        sep_token_id_li = []
        for sep in sep_tokens:
            token_id = tokenizer.convert_tokens_to_ids(sep)                
            sep_token_id_li.append(token_id)                                
        model.model.get_small_models(model_type='MF', data_type='movie',  
                                     model_path="***path_to_small_model***",
                                     user_num=22967, item_num=34154, emb_size=256)
        
        model.model.prepare_collm_prompt(user_token_id=sep_token_id_li[0], item_token_id=sep_token_id_li[1], warm_token_id=sep_token_id_li[2],
                                         small_emb_dim=256, 
                                         projection_model_path='***path_to_projection_model***', 
                                         big_linear_path='***path_to_big_model***',
                                         pretrained_with_small=False
                                         )
        
        peft_model = PeftModel.from_pretrained(model, lora_model_path)
        peft_model.eval()
        
        for name, param in peft_model.named_parameters():
            if 'small_model' in name:
                if 'embedding' in name:
                    # if 'llm' not in name:
                    param.requires_grad = True
            if 'projection_model' in name:
                param.requires_grad = True

            if 'big_linear_model' in name:
                param.requires_grad = True
        peft_model.print_trainable_parameters()
        if training_args.gradient_checkpointing:
            peft_model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    
    # Start trainer
    trainer = Comtrainer(
        model=peft_model, tokenizer=tokenizer, args=training_args, **data_module,
        compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # trainer.train()
    
    # safe_save_model_for_hf_trainer(
    #     trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    # )
    eval_results = trainer.evaluate()
    print(eval_results)



if __name__ == "__main__":
    train()
# -*- coding: utf-8 -*-
