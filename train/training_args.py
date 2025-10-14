from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments as TA

fsdp_config = {
    "Qwen/Qwen3-4B-Base": {
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
        "fsdp_sync_module_states": "false",
    },
    "Qwen/Qwen3-8B-Base": {
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
        "fsdp_sync_module_states": "false",
    },
    "EleutherAI/pythia-2.8b": {
        "fsdp_transformer_layer_cls_to_wrap": ["GPTNeoXLayer"],
        "fsdp_sync_module_states": "false",
    },
    "EleutherAI/pythia-6.9b": {
        "fsdp_transformer_layer_cls_to_wrap": ["GPTNeoXLayer"],
        "fsdp_sync_module_states": "false",
    },
    "meta-llama/Llama-3.2-3B": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_sync_module_states": "false",
    },
}


@dataclass
class TrainingArguments(TA):
    save_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "The base directory where the model checkpoints will be saved."
        },
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per device accelerator core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device accelerator core/CPU for evaluation."},
    )
    num_train_epochs: float = field(
        default=1.0, metadata={"help": "Total number of training epochs to perform."}
    )
    gradient_accumulation_steps: int = field(
        default=256,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(
        default=10,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_strategy: str = field(
        default="no", metadata={"help": "When to save checkpoints"}
    )
    save_steps: Optional[int] = field(
        default=10000, metadata={"help": "How often to save a model"}
    )
    save_total_limit: int = field(
        default=1,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    logging_dir: Optional[str] = field(
        default=None, metadata={"help": "Tensorboard log dir."}
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(
        default=False, metadata={"help": "Log the first global_step"}
    )
    logging_steps: float = field(
        default=5,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    report_to: Union[None, str, list[str]] = field(
        default="none",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    max_steps: int = field(
        default=500,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    learning_rate: float = field(
        default=3e-4, metadata={"help": "The initial learning rate for AdamW."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.95, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.01,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )

    ## Custom arguments that we add for Boomerang Distillation
    use_cross_entropy: bool = field(
        default=True,
        metadata={"help": "Whether to use cross-entropy loss."},
    )
    use_kl_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use KL divergence loss on the logits."},
    )
    kl_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the KL divergence loss."},
    )
    T: float = field(
        default=1.0,
        metadata={"help": "Temperature for the KL divergence loss."},
    )
    use_cosine_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use cosine embedding loss on the hidden states."},
    )
    cosine_loss_weight: float = field(
        default=2.0,
        metadata={
            "help": "Weight for the cosine embedding loss. This weight is divided by the number of aligned hidden states during training."
        },
    )

    def __post_init__(self):
        # note: set fsdp_config to a teacher model name to replicate the settings we use for that teacher model in the paper
        if isinstance(self.fsdp_config, str) and self.fsdp_config in fsdp_config:
            self.fsdp_config = fsdp_config[self.fsdp_config]
            self.fsdp = "full_shard auto_wrap"
        super().__post_init__()
