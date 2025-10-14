from __future__ import annotations

import argparse
import gc
import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from patching.patch import build_intermediate_model

DEFAULT_CLASSIFICATION_TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "race",
    "mmlu",
    "rte",
]
DEFAULT_PERPLEXITY_TASKS = ["wikitext"]
DEFAULT_GENERATIVE_TASKS = [
    "gsm8k_cot",
    "ifeval",
    "hendrycks_math",
]


class MyCustomWrapper:
    # this is a bit hacky, but we need to do this for lm_eval compatibility with our method
    def __init__(self, actual_model):
        self._model = actual_model

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate all attribute access to the base model
        return getattr(self._model, name)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)


def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, np.dtype):
        return str(obj)
    return obj


def lm_eval_evaluate(model, tokenizer, tasks: str = None, batch_size: int = 4):
    """
    Use lm-evaluation-harness to evaluate the model.

    Args:
        model: Model to be evaluated
        tokenizer: Tokenizer for the model
        tasks (str): Comma-separated string of tasks to evaluate on. If None, use DEFAULT_TASKS.
        batch_size (int): Batch size for evaluation

    Returns:
        A dictionary with evaluation results.
    """
    # setting these to override model defaults if they exist and use standard lm-eval-harness configuration
    model.generation_config.max_new_tokens = None
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    if tasks is None:
        tasks = DEFAULT_CLASSIFICATION_TASKS
    else:
        tasks = tasks.split(",")
    print(f"Evaluating on tasks: {tasks}")

    all_res = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
    )

    # groups is not always present
    if "groups" not in all_res:
        all_res["groups"] = {}

    return {"results": all_res["results"], "groups": all_res["groups"]}


def get_n_params(model):
    """
    Get the number of training-time and inference-time parameters of the model.

    Args:
        model: The model to be evaluated.

    Returns:
        A dictionary with 'training_time' and 'inference_time' parameter counts.
    """
    # TODO: check this works
    lm_head_params = 0
    if (
        (hasattr(model.config, "tie_weights") and model.config.tie_weights is True)
        or (
            hasattr(model.config, "tie_word_embeddings")
            and model.config.tie_word_embeddings is True
        )
    ) and hasattr(model, "lm_head"):
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "training_time": total_params - lm_head_params,
        "inference_time": total_params,
    }


def main(
    teacher_name_or_path: str,
    student_name_or_path: str,
    save_directory: str,
    num_layers_to_patch: int = 1,
    patch_first_k_layers: bool = False,
    batch_size: int = 4,
    tasks: Optional[Union[str, List[str]]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    model_results = {}

    intermediate, tokenizer = build_intermediate_model(
        teacher_name_or_path=teacher_name_or_path,
        student_name_or_path=student_name_or_path,
        num_layers_to_patch=num_layers_to_patch,
        patch_first_k_layers=patch_first_k_layers,
        dtype=dtype,
    )
    intermediate.to(device)
    intermediate.eval()

    print(f"Final number of layers: {intermediate.config.num_hidden_layers}")
    model_results["parameters"] = get_n_params(intermediate)

    # this for lm-eval compatibility so that the weights are not tied. Without this, lm-eval automatically performs weight tying.
    intermediate = MyCustomWrapper(intermediate)

    # include
    tasks_str = (
        tasks if isinstance(tasks, str) else "_".join(tasks) if tasks else "all_default"
    )
    curr_out_file = os.path.join(
        f"{save_directory}",
        f"lm_eval_patch_first_{patch_first_k_layers}_n_layers_{intermediate.config.num_hidden_layers}_{tasks_str}.json",
    )

    # create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    model_results["lm_eval_results"] = lm_eval_evaluate(
        intermediate, tokenizer, batch_size=batch_size, tasks=tasks
    )

    with open(curr_out_file, "w+") as f:
        json.dump(model_results, f, default=to_serializable, indent=2)

    del intermediate
    gc.collect()
    torch.cuda.empty_cache()

    return curr_out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_model_name_or_path",
        required=True,
        help="huggingface base model from which layers were cut",
    )
    parser.add_argument(
        "--student_model_name_or_path", required=True, help="student name or path"
    )
    parser.add_argument(
        "--save_directory", required=True, help="directory to save the results in"
    )
    parser.add_argument(
        "--num_layers_to_patch",
        type=int,
        default=1,
        help="number of student layers to patch",
    )
    parser.add_argument(
        "--patch_first_k_layers",
        action="store_true",
        help="patch student layers starting from first layers of model if true, last layers otherwise",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        required=False,
        help="which tasks to evaluate in lm-eval-harness (input as comma-separated string)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        required=False,
        help="batch size for lm-eval-harness evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="data type for model weights",
    )
    parser.add_argument(
        "--override_llama_patching",
        action="store_true",
        help="If true, overrides the default patching order behavior for llama",
    )
    args = parser.parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if (
        not args.override_llama_patching
        and "llama" in args.teacher_model_name_or_path.lower()
    ):
        print("Using default patching order for llama models (first k layers)")
        args.patch_first_k_layers = True

    out_file = main(
        teacher_name_or_path=args.teacher_model_name_or_path,
        student_name_or_path=args.student_model_name_or_path,
        save_directory=args.save_directory,
        num_layers_to_patch=args.num_layers_to_patch,
        patch_first_k_layers=args.patch_first_k_layers,
        batch_size=args.eval_batch_size,
        tasks=args.tasks,
        dtype=dtype_map[args.dtype],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("Results saved to", out_file)
