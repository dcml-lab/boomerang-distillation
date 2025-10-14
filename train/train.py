import argparse
import gc
import json
from typing import Dict

import torch

try:
    import wandb
except ImportError:
    pass
from datasets import load_dataset
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from patching.utils import cut_layers
from train.data_args import DataArguments
from train.model_args import ModelArguments
from train.training_args import TrainingArguments


def is_main_process():
    try:
        import torch.distributed as dist

        return (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )
    except Exception:
        return True


def get_layer_mappings(
    first_layers_to_keep: int = 1,
    last_layers_to_keep: int = 2,
    alternate_every_n_layers: int = 2,
    num_layers: int = 36,
) -> tuple[Dict[int, int], list[int]]:
    """
    Get the mapping between student layers and teacher blocks, as well as the layers to prune from the teacher.

    Args:
        first_layers_to_keep (int): Number of initial layers to keep from the teacher.
        last_layers_to_keep (int): Number of final layers to keep from the teacher.
        alternate_every_n_layers (int): Frequency of layers to keep in between the first and last layers.
        num_layers (int): Total number of layers in the teacher model.

    Returns:
        tuple: A dictionary mapping student activation indices to teacher activation indices, and a list of layers to cut.
    """
    # We start by mapping all of first_layers_to_keep between the student and teacher.
    # Note that the activations of the embedding layer are at index 0, so the output activations of the transformer blocks start at index 1.
    aligned_layers = {i: i for i in range(first_layers_to_keep)}
    teacher_idx = first_layers_to_keep + alternate_every_n_layers - 1

    # Add mapping between every alternate_every_n_layers layers from first_layers_to_keep to last_layers_to_keep.
    align_idxs = []
    while teacher_idx <= num_layers - last_layers_to_keep:
        align_idxs.append(teacher_idx)
        teacher_idx += alternate_every_n_layers
    largest_aligned = 0
    for i, idx in enumerate(align_idxs):
        aligned_layers[i + first_layers_to_keep] = idx
        largest_aligned = i + first_layers_to_keep

    # Add mapping between the last last_layers_to_keep student and teacher layers.
    for i in range(num_layers - last_layers_to_keep + 1, num_layers):
        largest_aligned += 1
        aligned_layers[largest_aligned] = i
    aligned_layers[-1] = -1

    # Create list of teacher layers to prune in order to initialize the student.
    layers_to_cut = []
    for idx in align_idxs:
        for i in range(alternate_every_n_layers - 1, 0, -1):
            layers_to_cut.append(idx - i)
    return aligned_layers, layers_to_cut


def prepare_student(
    teacher_model: AutoModelForCausalLM,
    layers_to_cut: list[int],
    model_type: str,
    random_initialization: bool = False,
) -> AutoModelForCausalLM:
    """
    Prepare the student model by cutting layers from the teacher or doing random initialization with the same
    architecture.

    Args:
        teacher_model: The teacher model from which layers will be cut.
        layers_to_cut (list[int]): List of layer indices to be removed from the teacher model.
        model_type (str): The type of the model, e.g., 'pythia', 'qwen', etc.
        random_initialization (bool): If True, initializes a model with random weights with the same architecture as the student.
    """
    if not random_initialization:
        student_model = cut_layers(teacher_model, layers_to_cut, model_type=model_type)
        student_model.requires_grad_(True)
    else:
        student_model_config = teacher_model.config
        student_model_config.torch_dtype = torch.float32
        student_model_config.use_cache = False
        student_model = AutoModelForCausalLM.from_config(student_model_config)
        student_model = cut_layers(student_model, layers_to_cut, model_type=model_type)
        student_model.requires_grad_(True)
    return student_model


class BoomerangDistillationTrainer(Trainer):
    """
    Custom trainer with additional optional loss terms for aligned distillation to the teacher model, including KL
    divergence loss on the logits and cosine embedding loss on the output hidden states of each student layer and
    its corresponding teacher block.

    Args:
        *args: Arguments passed to the Trainer class.
        teacher_model: The teacher model used for distillation.
        aligned_layers (dict): A mapping between the output activations of student layers and teacher blocks to be aligned.
        pad_token (int): The padding token ID used in the tokenizer.
        **kwargs: Additional keyword arguments passed to the Trainer class.
    """

    def __init__(
        self,
        args,
        teacher_model: AutoModelForCausalLM = None,
        aligned_layers: dict = None,
        pad_token: int = None,
        **kwargs,
    ):
        super().__init__(args=args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_loss_weight = args.kl_loss_weight
        self.cosine_loss_weight = args.cosine_loss_weight

        if aligned_layers is None:
            self.aligned_layers = {-1: -1}
        else:
            self.aligned_layers = aligned_layers

        self.teacher_model = self.teacher_model.to(self.args.device)

        self.use_cross_entropy = args.use_cross_entropy
        self.CELoss = None
        if self.use_cross_entropy:
            self.CELoss = nn.CrossEntropyLoss()

        self.use_kl_loss = args.use_kl_loss
        self.KDLoss = None
        if self.use_kl_loss:
            self.KDLoss = torch.nn.KLDivLoss(reduction="batchmean")
            assert (
                self.teacher_model is not None
            ), "Teacher model must be provided if using KL loss"

        self.use_cosine_loss = args.use_cosine_loss
        self.rep_loss = None
        if self.use_cosine_loss:
            self.rep_loss = torch.nn.CosineEmbeddingLoss()
            assert (
                self.teacher_model is not None
            ), "Teacher model must be provided if using cosine loss"

        assert (
            self.use_cross_entropy or self.use_kl_loss or self.use_cosine_loss
        ), "At least one loss term must be used"

        self.pad_token = pad_token
        self.T = args.T if hasattr(args, "T") else 1.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            mask = (
                (inputs.get("attention_mask") > 0)
                .unsqueeze(-1)
                .expand_as(outputs.hidden_states[-1])
            )
            hid_dim = outputs.hidden_states[-1].size(-1)
            hidden_states = {
                l: torch.masked_select(outputs.hidden_states[l], mask)
                for l in self.aligned_layers
            }

            # Compute cross entropy loss
            if self.use_cross_entropy or not model.training:
                labels = inputs.get("labels")
                loss_fct = torch.nn.CrossEntropyLoss()
                pred_logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                vocab_size = (
                    model.module.config.vocab_size
                    if hasattr(model, "module")
                    else model.config.vocab_size
                )
                ce_loss = loss_fct(pred_logits.view(-1, vocab_size), labels.view(-1))
            else:
                ce_loss = torch.tensor(0.0, device=model.device, dtype=torch.bfloat16)

            loss = [ce_loss]

            if model.training and (self.use_kl_loss or self.use_cosine_loss):
                teacher_model_outputs = None
                with torch.no_grad():
                    teacher_model_outputs = self.teacher_model(
                        **inputs, output_hidden_states=True
                    )
                    teacher_hidden_states = teacher_model_outputs.hidden_states
                    teacher_hidden_states = {
                        self.aligned_layers[l]: torch.masked_select(
                            teacher_hidden_states[self.aligned_layers[l]], mask
                        )
                        for l in self.aligned_layers
                    }

                # compute KL loss
                distill_loss = torch.tensor(
                    0.0, device=model.device, dtype=torch.bfloat16
                )
                if self.use_kl_loss:
                    teacher_model_outputs.logits.detach()
                    soft_targets = nn.functional.softmax(
                        teacher_model_outputs.logits.view(-1, logits.size(-1)) / self.T,
                        dim=-1,
                    )
                    soft_prob = nn.functional.log_softmax(
                        logits.view(-1, logits.size(-1)) / self.T, dim=-1
                    )

                    distill_loss = self.KDLoss(soft_prob, soft_targets) * (self.T**2)
                    loss.append(self.kl_loss_weight * distill_loss)

                # compute cosine alignment loss
                if self.use_cosine_loss:
                    for l in self.aligned_layers:
                        teacher_layer = self.aligned_layers[l]
                        with torch.no_grad():
                            curr_teacher_hidden = teacher_hidden_states[
                                teacher_layer
                            ].view(-1, hid_dim)
                            del teacher_hidden_states[teacher_layer]

                        cosine_loss = torch.tensor(
                            0.0, device=model.device, dtype=torch.bfloat16
                        )
                        if self.teacher_model is not None:
                            curr_hidden = hidden_states[l].view(-1, hid_dim)
                            target = curr_hidden.new(curr_hidden.size(0)).fill_(1)

                            cosine_loss = self.rep_loss(
                                curr_teacher_hidden.float(),
                                curr_hidden.float(),
                                target.float(),
                            )

                            loss.append(
                                (self.cosine_loss_weight * cosine_loss)
                                / len(self.aligned_layers)
                            )

        total_loss = sum(loss)

        return (total_loss, outputs) if return_outputs else total_loss


def create_model_out_str(args: argparse.Namespace) -> str:
    # Create a string representation of the model output directory
    model_str = args.teacher_model_name_or_path.split("/")[-1]
    model_out_str = f"{model_str}"
    model_out_str += f"_initialization_every_{args.alternate_every_n_layers}"
    model_out_str += (
        f"_first_{args.first_layers_to_keep}_last_{args.last_layers_to_keep}"
    )
    model_out_str += f"_random_{args.random_initialization}"
    model_out_str += f"_cross_entropy_{args.use_cross_entropy}"
    model_out_str += f"_distill_loss_{args.use_kl_loss}_{args.kl_loss_weight}_{args.T}"
    model_out_str += f"_cosine_loss_{args.use_cosine_loss}_{args.cosine_loss_weight}"
    model_out_str += f"_{args.seed}_{args.max_steps}"
    return model_out_str


def main():
    hfparser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = hfparser.parse_args_into_dataclasses()
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    if args.seed is not None:
        set_seed(args.seed)

    if args.report_to == "wandb":
        wandb.init(
            project="boomerang-distillation",
            config=args.to_sanitized_dict(),
            dir=f"{args.save_directory}/wandb",
            settings=wandb.Settings(init_timeout=120),
        )

    model_out_str = create_model_out_str(args)
    args.output_dir = f"{args.save_directory}/models/{model_out_str}"

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name_or_path, torch_dtype=torch.float32
    )
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    last_layer = teacher_model.config.num_hidden_layers

    # get mapping between student layers and teacher blocks
    aligned_layers = None
    layers_to_cut = []
    aligned_layers, layers_to_cut = get_layer_mappings(
        args.first_layers_to_keep,
        args.last_layers_to_keep,
        args.alternate_every_n_layers,
        last_layer,
    )

    # initialize student model by dropping layers_to_cut from teacher
    model_str = args.teacher_model_name_or_path.split("/")[-1]
    student_model = prepare_student(
        teacher_model, layers_to_cut, model_str, args.random_initialization
    )
    teacher_model.to(torch.bfloat16)

    gc.collect()
    torch.cuda.empty_cache()

    student_model.train()

    train_dataset = load_dataset(
        args.dataset,
        split="train",
        streaming=True,
    )

    test = train_dataset.take(512)
    # Note: skipping the first 50k examples for reproducibility. This does not affect functionality.
    train_dataset = train_dataset.skip(50000)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )

    tokenized_train = train_dataset.shuffle(buffer_size=100_000, seed=args.seed).map(
        tokenize_function, batched=True
    )
    eval_dataset = test.map(tokenize_function, batched=True)

    trainer = BoomerangDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        teacher_model=teacher_model,
        aligned_layers=aligned_layers,
        pad_token=tokenizer.pad_token_id,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    if is_main_process():
        student_config_dct = {
            "pruned_teacher_layers": layers_to_cut,
            "copied_teacher_layers": [
                l
                for l in range(teacher_model.config.num_hidden_layers)
                if l not in layers_to_cut
            ],
        }

        with open(f"{args.output_dir}/student_config.json", "w") as f:
            json.dump(student_config_dct, f)


if __name__ == "__main__":
    main()
