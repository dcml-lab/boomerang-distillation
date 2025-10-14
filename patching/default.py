from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM

from patching.utils import (
    get_layer_cutoffs,
    postprocess_intermediate_model,
    reload_student_model,
)


def patch_first_k_layers(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    k: int,
    student_config: dict = None,
) -> AutoModelForCausalLM:
    """
    Patch the first k layers of the student model with the corresponding layers from the teacher model.

    Args:
        student_model: The student model to be patched.
        teacher_model: The teacher model from which layers will be taken.
        k (int): The number of student layers to patch.
        student_config (dict): Configuration dictionary containing 'copied_teacher_layers' and 'pruned_teacher_layers'.

    Returns:
        The patched intermediate model.
    """
    assert student_config is not None, "student_config must be provided to patch layers"

    # TODO: this code ignores contiguous layers -- should we keep this?
    student_idx, teacher_idx = get_layer_cutoffs(
        student_config["copied_teacher_layers"], k, patch_from_last=False
    )
    intermediate_model = reload_student_model(
        student_model, teacher_model, student_config["pruned_teacher_layers"]
    )

    # core patching code
    teacher_layers = torch.nn.ModuleList(
        [deepcopy(teacher_model.model.layers[i]) for i in range(teacher_idx)]
    )
    student_layers = intermediate_model.model.layers[student_idx:]

    intermediate_model.model.layers = teacher_layers + student_layers
    intermediate_model.model.embed_tokens = deepcopy(teacher_model.model.embed_tokens)

    # postprocessing to set layer numbers correctly
    intermediate_model = postprocess_intermediate_model(intermediate_model)

    return intermediate_model


def patch_last_k_layers(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    k: int,
    student_config: dict = None,
) -> AutoModelForCausalLM:
    """
    Patch the last k layers of the student model with the corresponding layers from the teacher model.

    Args:
        student_model: The student model to be patched.
        teacher_model: The teacher model from which layers will be taken.
        k (int): The number of student layers to patch.
        student_config (dict): Configuration dictionary containing 'copied_teacher_layers' and 'pruned_teacher_layers'.

    Returns:
        The patched intermediate model.
    """
    assert student_config is not None, "student_config must be provided to patch layers"

    # TODO: this code ignores contiguous layers -- should we keep this?
    student_idx, teacher_idx = get_layer_cutoffs(
        student_config["copied_teacher_layers"], k, patch_from_last=True
    )
    intermediate_model = reload_student_model(
        student_model, teacher_model, student_config["pruned_teacher_layers"]
    )

    # core patching code
    student_layers = intermediate_model.model.layers[:student_idx]
    teacher_layers = torch.nn.ModuleList(
        [
            deepcopy(teacher_model.model.layers[i])
            for i in range(teacher_idx, teacher_model.config.num_hidden_layers)
        ]
    )

    intermediate_model.model.layers = student_layers + teacher_layers
    intermediate_model.model.norm = deepcopy(teacher_model.model.norm)
    intermediate_model.model.rotary_emb = deepcopy(teacher_model.model.rotary_emb)
    intermediate_model.lm_head = deepcopy(teacher_model.lm_head)

    # postprocessing to set layer numbers correctly
    intermediate_model = postprocess_intermediate_model(intermediate_model)
    return intermediate_model
