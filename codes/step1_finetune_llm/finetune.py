# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


import json
import os
import pathlib
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# 导入深度学习相关的库：datasets（数据集处理），torch（PyTorch），transformers（用于NLP模型），
# deepspeed（用于大规模训练优化），peft（LoRA配置），以及Scikit-learn的roc_auc_score用于评估。
import datasets
import torch
import transformers
import numpy as np
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader


# 导入transformers库中的模型、分词器、训练工具等，主要是Qwen2系列模型和配置。
from transformers import (
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    BitsAndBytesConfig,
    Trainer,
    deepspeed,Qwen2Model,is_datasets_available, is_torch_xla_available,
)
# from transformers.integrations import deepspeed
# 导入transformers库中的集成工具、缓存管理、评估相关工具，并设置日志记录器。
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations import deepspeed_init
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, PREFIX_CHECKPOINT_DIR, denumpify_detensorize, has_length
from transformers.utils import logging
logger = logging.get_logger(__name__)

# 导入一些与模型输出、模型结构、训练、数据集相关的工具函数和类。
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from transformers.trainer_pt_utils import LabelSmoother, EvalLoopContainer, find_batch_size, IterableDatasetShard


# 如果可用，则导入torch_xla用于TPU支持。
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


# 定义忽略标签ID，用于标签平滑（Label Smoothing）
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# 定义一个模板字符串，生成消息格式化的字符串。模板用于将消息列表（messages）转换为某种自定义格式。
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

# 初始化local_rank变量，用于分布式训练中确定当前进程的排名。
local_rank = None

# 定义一个打印函数，在分布式训练中只有rank0（即主进程）会打印消息。
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ModelArguments 类用于存储与模型相关的参数，@dataclass 装饰器简化了初始化和表示
@dataclass
class ModelArguments:
    # model_name_or_path: 模型的名称或路径，默认为 "Qwen/Qwen2-7B"。此路径可以是预训练模型的名称，或者是本地模型的路径。
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")


# DataArguments 类用于存储与数据相关的参数。
@dataclass
class DataArguments:
    # data_path: 训练数据的路径，必须提供，用于指定训练数据文件的位置。
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

    # eval_data_path: 评估数据的路径，必须提供，用于指定评估数据文件的位置。
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )

    # lazy_preprocess: 是否使用延迟预处理，默认为False。延迟预处理会在数据加载时按需进行，而不是预先处理所有数据。
    lazy_preprocess: bool = False

# 继承自transformers.TrainingArguments, TrainingArguments 类存储与训练过程相关的参数，允许用户定制训练过程。
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # cache_dir: 用于缓存数据的目录路径，默认为 None。可以为模型缓存、数据缓存指定一个目录。
    cache_dir: Optional[str] = field(default=None)

    # optim: 选择优化器，默认为 "adamw_torch"。指定用于训练的优化器类型，如AdamW优化器。
    optim: str = field(default="adamw_torch")

    # model_max_length: 模型的最大序列长度，默认为 8192。超过此长度的序列会被截断，较短的序列会被填充。
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # use_lora: 是否启用 LoRA（低秩适配）技术，默认为 False。如果启用 LoRA，它可以帮助减少训练时的内存使用。
    use_lora: bool = False


# LoraArguments 类用于存储与LoRA（Low-Rank Adaptation）相关的参数
@dataclass
class LoraArguments:
    # lora_r: LoRA的秩（rank），表示在低秩适配中，秩的大小。默认为64。
    lora_r: int = 64  

    # lora_alpha: LoRA的缩放系数，用于控制低秩适配的学习能力。默认为16。
    lora_alpha: int = 16

    # lora_dropout: LoRA适配层的dropout比例，用于正则化。默认为0.05。
    lora_dropout: float = 0.05

    # lora_target_modules: LoRA技术应用的目标模块。
    # 默认为 ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]。
    # 这些通常是模型中需要插入低秩适配矩阵的层。
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

    # lora_weight_path: 指定LoRA权重文件的路径。默认为空字符串，表示不加载任何LoRA权重。
    lora_weight_path: str = ""

    # lora_bias: LoRA偏置类型，通常用于选择如何应用LoRA的偏置。默认为 "none"。
    lora_bias: str = "none"

     # q_lora: 是否启用量化LoRA（QLoRA）。默认为 False。如果启用QLoRA，则会对LoRA的权重进行量化，从而减少内存使用。
    q_lora: bool = False

# maybe_zero_3函数用于处理模型参数的Zero3分区。
# Zero3是DeepSpeed中的一个内存优化技术，主要用于大模型的分布式训练
def maybe_zero_3(param):
    # 如果参数有 "ds_id" 属性，说明它是一个分布式训练的参数。
    if hasattr(param, "ds_id"):
        ds_status = getattr(param, "ds_status", None)
        if ds_status == ZeroParamStatus.AVAILABLE:
            param = param.data.detach().cpu().clone()
        else:
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
    else:
        # 如果参数没有 "ds_id" 属性，说明它不是分布式训练中的参数。
        # 将参数从GPU转移到CPU，并克隆一份新的Tensor。
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
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def save_minimal_pretrained(trainer: transformers.Trainer, output_dir: str, state_dict: Dict[str, torch.Tensor]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = trainer.model
    try:
        model_to_save = unwrap_model(trainer.model)
    except Exception:
        model_to_save = trainer.model
    model_to_save.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=bool(getattr(trainer.args, "save_safetensors", True)),
    )
    torch.save(trainer.args, os.path.join(output_dir, "training_args.bin"))


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    if trainer.args.use_lora:
        state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), bias)
    elif deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.is_world_process_zero():
        save_minimal_pretrained(trainer, output_dir, state_dict=state_dict)


def preprocess(
    messages,
    users,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    user_li = []
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
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask, users=user_li
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
        ret = preprocess([self.raw_data[i]["messages"]],[self.raw_data[i]["source"]],self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["target_ids"][0],
            attention_mask=ret["attention_mask"][0],
            user_id=ret["users"]
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


# Comtrainer 类继承自Trainer，重写了get_eval_dataloader方法，用于获取评估数据加载器（DataLoader）。
class Comtrainer(Trainer):
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """

        # 如果 eval_dataset 和 self.eval_dataset 都为 None，抛出错误，因为没有评估数据集
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        # 如果已经有持久化的 workers（即数据加载器），并且 eval_dataset 需要重用，则直接返回已准备好的数据加载器
        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"


        if (
                hasattr(self, "_eval_dataloaders")
                and dataloader_key in self._eval_dataloaders
                and self.args.dataloader_persistent_workers
        ):
            # 使用 accelerator.prepare 准备数据加载器
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])
        
        # 如果传入的 eval_dataset 是字符串，使用该键从 self.eval_dataset 中提取对应的评估数据集
        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        # 初始化数据整理器（collator）
        data_collator = self.data_collator
        self.user_list = []

        # 获取每个数据样本中的 'user_id'，并将其存储到 user_list 中
        for i in eval_dataset:
            user = i['user_id'][0]
            self.user_list.append(int(user))

        # 如果数据集是HuggingFace的 datasets.Dataset 类型，则清理不必要的列
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        else:
            # 否则，清理数据集中的无用列
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")
        
        # 设置 DataLoader 参数
        dataloader_params = {
            "batch_size": self.args.eval_batch_size, # 设置评估批次大小
            "collate_fn": data_collator,# 设置数据整理器
            "num_workers": self.args.dataloader_num_workers,# 设置数据加载的工作线程数
            "pin_memory": self.args.dataloader_pin_memory,# 是否将数据加载到固定内存
            "persistent_workers": self.args.dataloader_persistent_workers,# 是否使用持久化工作线程
        }

        # 如果 eval_dataset 不是 IterableDataset，设置采样器、丢弃最后一批数据的策略以及预取因子
        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # 创建 DataLoader 对象，用于评估数据集
        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

        # 如果启用了持久化的 workers，将数据加载器保存在 _eval_dataloaders 字典中
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}
        
        # 使用 accelerator.prepare 准备数据加载器，支持分布式训练的优化
        return self.accelerator.prepare(eval_dataloader)
    
    
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
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), self.user_list)
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

    def _save_checkpoint(self, model, trial, metrics=None):
        run_dir = self._get_output_dir(trial=trial)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if self.args.should_save and self.is_world_process_zero():
            safe_save_model_for_hf_trainer(self, output_dir=output_dir, bias="none")
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            try:
                for name in os.listdir(output_dir):
                    if name.startswith("global_step"):
                        shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)
            except Exception:
                pass

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

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
        if len(np.unique(label_i)) == 1:  # 如果只有一个标签
            only_one_class += 1
            continue  # 跳过此用户的AUC计算
        try:
            ui_auc = roc_auc_score(label_i, pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            print("only one class:",only_one_class)

    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time() - start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user

def ranking_metrics(user, scores, labels):
    if not isinstance(user, np.ndarray):
        user = np.array(user)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    scores = scores.squeeze()
    labels = labels.squeeze().astype(np.int64)

    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)

    aps = []
    ndcgs = []
    errs = []

    total_num = 0
    skipped_users = 0
    for i, u_i in enumerate(u):
        start_id, end_id = total_num, total_num + counts[i]
        total_num = end_id
        idx = index[start_id:end_id]
        if idx.size == 0:
            continue

        s = scores[idx]
        y = labels[idx]
        pos_total = int(y.sum())
        neg_total = int(y.shape[0] - pos_total)
        if pos_total <= 0 or neg_total <= 0:
            skipped_users += 1
            continue

        order = np.argsort(-s, kind="mergesort")
        y_sorted = y[order]
        n = int(y_sorted.shape[0])

        hit = 0
        ap_sum = 0.0
        for rank, rel in enumerate(y_sorted, start=1):
            if rel:
                hit += 1
                ap_sum += hit / rank
        aps.append(float(ap_sum / pos_total) if pos_total > 0 else 0.0)

        discounts = 1.0 / np.log2(np.arange(2, n + 2))
        dcg = float(np.sum(y_sorted * discounts))
        idcg = float(np.sum(discounts[:pos_total]))
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)

        max_rel = int(np.max(y_sorted))
        if max_rel <= 0:
            errs.append(0.0)
        else:
            Rs = (np.power(2.0, y_sorted) - 1.0) / np.power(2.0, max_rel)
            p_continue = 1.0
            err = 0.0
            for rank, R in enumerate(Rs, start=1):
                err += p_continue * float(R) / rank
                p_continue *= (1.0 - float(R))
            errs.append(float(err))

    print(f"Evaluated users: {len(aps)}, Skipped (invalid data): {skipped_users}")
    if len(aps) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(aps)), float(np.mean(ndcgs)), float(np.mean(errs))

def compute_metrics(eval_preds, user_list):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    uauc,_,_ = uAUC_me(user_list, list(pre[0]), list(pre[1]))
    map_v, ndcg_v, err_v = ranking_metrics(user_list, pre[0], pre[1])
    return {'auc': auc, 'uauc': uauc, 'map': map_v, 'ndcg': ndcg_v, 'err': err_v}

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
class ComQwen(Qwen2Model):
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
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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
def train():
    # 用于设置local_rank，表示在分布式训练中的进程编号。
    global local_rank

    # 使用transformers库的HfArgumentParser解析命令行参数，将它们映射到对应的类中
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    # 将传入的命令行参数解析到对应的数据类中
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # 单GPU训练的QLoRA支持
    # This serves for single-gpu qlora.

    # 如果使用DeepSpeed并且是单GPU训练，则设置分布式类型为DEEPSPEED
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    
    # 设置local_rank
    local_rank = training_args.local_rank

    # 初始化device_map为None，用于分布式训练时设置模型放置的设备
    device_map = None

    # 从环境中获取world_size,不存在时默认为1
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("worldsize:",world_size)
    ddp = world_size != 1

    # 如果启用了QLoRA，则设置device_map并检查与FSDP或ZeRO3的兼容性
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")
    
    # 设置模型加载的额外配置
    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }


    print(3333333333)

    # 设置计算数据类型，根据训练配置选择FP16, BF16或者FP32
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # 加载模型和分词器
    # Load model and tokenizer
    config = transformers.Qwen2Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    # 禁用缓存以节省内存
    config.use_cache = False

    print("44444444444")

    # 加载模型
    model = ComQwenCausalLLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, # 使用4位量化
            bnb_4bit_use_double_quant=True, # 双重量化
            bnb_4bit_quant_type="nf4", # 量化类型
            bnb_4bit_compute_dtype=compute_dtype, # 计算数据类型
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )

    print("5555555555555555555555555")

    # 加载分词器
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    print("66666666666666666666666666666")
    # 如果使用LoRA进行训练
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM", # 任务类型为因果语言模型
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing,
            )
        
        # 获取LoRA模型
        model = get_peft_model(model, lora_config)
        # #### 测试模型所用代码
        # lora_model_path = "/workspace/wzh/self_reasearch/Chat_Qwen/examples/sft/data/data_llm/movie-stage1/movie-twoepoch/checkpoint-114/"
        
        # 加载LoRA配置
        # peft_config = PeftConfig.from_pretrained(lora_model_path)
        
        # 将LoRA权重加载到量化模型中
        # model = PeftModel.from_pretrained(model, lora_model_path)

        # 设置模型为评估模式
        model.eval()

        # 如果启用了LoRA，并且模型中的参数包含LoRA相关的权重，确保这些参数可以训练
        #### 测试模型所用代码
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        # 打印可训练的参数数量
        model.print_trainable_parameters()

        # 如果启用了梯度检查点，确保输入参数要求梯度
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    
    print("777777777777777777777777777777777777777777")
    # 加载数据
    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    
    # 创建训练器
    # Start trainer
    trainer = Comtrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module,
        compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    #     # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    # if (
    #     list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    #     and not training_args.use_lora
    # ):
    #     trainer.train(resume_from_checkpoint=True)
    # else: 
    print("8888888888888888888888888888888888888888888")
    
    trainer.train()

    if trainer.is_world_process_zero():
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)
        try:
            tokenizer.save_pretrained(training_args.output_dir)
        except Exception:
            pass
    
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(
    #     trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    # )
    # eval_results = trainer.evaluate()
    # print(eval_results)


if __name__ == "__main__":
    train()
