import gc
import json
import os
from typing import Callable, Dict

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import patching.default as patch_default
import patching.pythia as patch_pythia

PatchFn = Callable[[torch.nn.Module, torch.nn.Module, int, dict, bool], torch.nn.Module]


def _patch_default(student, teacher, k, student_config, first):
    fn = (
        patch_default.patch_first_k_layers
        if first
        else patch_default.patch_last_k_layers
    )
    return fn(student, teacher, k=k, student_config=student_config)


def _patch_pythia(student, teacher, k, student_config, first):
    fn = (
        patch_pythia.patch_first_k_layers if first else patch_pythia.patch_last_k_layers
    )
    return fn(student, teacher, k=k, student_config=student_config)


PATCHERS: Dict[str, PatchFn] = {
    "pythia": _patch_pythia,
    "default": _patch_default,
}


def build_intermediate_model(
    teacher_name_or_path: str,
    student_name_or_path: str,
    num_layers_to_patch: int,
    patch_first_k_layers: bool,
    dtype: torch.dtype,
):
    """
    Build the intermediate model by patching the student model with layers from the teacher model.

    Args:
        teacher_name_or_path (str): Huggingface model name or local path for the teacher model
        student_name_or_path (str): Huggingface model name or local path for the student model
        num_layers_to_patch (int): Number of student layers to patch
        patch_first_k_layers (bool): If True, patch starting from the first layers, else from the last layers
        dtype (torch.dtype): Data type for model weights

    Returns:
        The patched intermediate model and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(teacher_name_or_path)
    if getattr(tokenizer, "eos_token", None) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # left padding is typical for causal LM eval
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name_or_path, torch_dtype=dtype
    )
    if num_layers_to_patch == teacher_model.config.num_hidden_layers:
        return teacher_model, tokenizer
        
    student = AutoModelForCausalLM.from_pretrained(
        student_name_or_path, torch_dtype=dtype
    )

    if os.path.exists(os.path.join(student_name_or_path, "student_config.json")):
        with open(os.path.join(student_name_or_path, "student_config.json")) as f:
            student_config = json.load(f)
    else:
        # assume this is a huggingface model and load the file from the hub
        student_config_path = hf_hub_download(
            repo_id=student_name_or_path, filename="student_config.json"
        )
        with open(student_config_path) as f:
            student_config = json.load(f)

    key = "pythia" if "pythia" in teacher_name_or_path.lower() else "default"
    intermediate = PATCHERS[key](
        student, teacher, num_layers_to_patch, student_config, patch_first_k_layers
    )

    del student, teacher
    gc.collect()
    return intermediate, tokenizer
