from copy import deepcopy

from transformers import AutoModelForCausalLM


def cut_layers(
    teacher_model: AutoModelForCausalLM,
    layers_to_cut: list[int],
    model_type: str = "qwen",
):
    """
    Removes layers_to_cut from the teacher model to initialize the student.

    Args:
        teacher_model (AutoModelForCausalLM): The teacher model from which layers will be cut.
        layers_to_cut (list[int]): List of layer indices to be removed from the teacher model.
        model_type (str): The type of the model, e.g., 'pythia', 'qwen', etc.

    Returns:
        AutoModelForCausalLM: The modified model with specified layers removed.
    """
    model_copy = deepcopy(teacher_model)
    base_state_dict = model_copy.state_dict()

    model_copy.load_state_dict(base_state_dict, strict=False)

    for diff_lay in layers_to_cut[::-1]:
        if "pythia" in model_type:
            del model_copy.gpt_neox.layers[diff_lay]
        else:
            del model_copy.model.layers[diff_lay]

    if "pythia" in model_type:
        model_copy = postprocess_intermediate_pythia_model(model_copy)
    else:
        model_copy = postprocess_intermediate_model(model_copy)

    return model_copy


def reload_student_model(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    layers_to_cut: list[int],
    model_type: str = "qwen",
) -> AutoModelForCausalLM:
    """
    Reload the student model because of limited compatibility with huggingface autoclasses

    Args:
        student_model: The student model to be reloaded.
        teacher_model: The teacher model from which layers will be taken.
        layers_to_cut (list[int]): List of layer indices to cut from the teacher model.
        model_type (str): The type of model architecture (eg "qwen").

    Returns:
        The reloaded student model.
    """
    model = cut_layers(teacher_model, layers_to_cut, model_type=model_type)
    model.load_state_dict(student_model.state_dict(), strict=True)
    return model


def get_layer_cutoffs(all_layers_to_keep, k: int, patch_from_last: bool = True):
    """
    Get the layer cutoffs for patching the first k layers.

    Args:
        all_layers_to_keep (list[int]): List of layer indices that are kept when initializing the student model.
        k (int): The number of student layers to patch.
        patch_from_last (bool): If True, patch starting from the last layers, else starting from the first layers.

    Returns:
        A tuple of (student_layer_index, teacher_layer_index) indicating the cutoffs for patching.
    """
    layers_to_keep = []
    small_model_layers = []
    if not patch_from_last:
        for i, l in enumerate(all_layers_to_keep[:-1]):
            if all_layers_to_keep[i + 1] - l > 1:
                layers_to_keep.append(l)
                small_model_layers.append(i)
        if k >= len(small_model_layers):
            # assumes that last layer is kept in teacher
            return len(all_layers_to_keep), all_layers_to_keep[-1] + 1
        return small_model_layers[k], layers_to_keep[k]
    else:
        for i, l in enumerate(all_layers_to_keep[1:]):
            if l - all_layers_to_keep[i] > 1:
                layers_to_keep.append(l)
                small_model_layers.append(i + 1)
        if (k+1) > len(small_model_layers):
            return 0, 0
        return small_model_layers[-1 * (k + 1)], layers_to_keep[-1 * (k + 1)]


def set_layer_idx_recursive(block, idx, attr="layer_idx"):
    """Set `attr` on `block` and all its submodules."""
    setattr(block, attr, idx)

    def _set(m):
        setattr(m, attr, idx)

    block.apply(_set)


def postprocess_intermediate_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Postprocess the intermediate model to set the layer indices correctly.
    """
    model.config.num_hidden_layers = len(model.model.layers)
    for i, block in enumerate(model.model.layers):
        set_layer_idx_recursive(block, i)

    return model


def postprocess_intermediate_pythia_model(
    model: AutoModelForCausalLM,
) -> AutoModelForCausalLM:
    """
    Postprocess the intermediate model to set the layer indices correctly.
    """
    model.config.num_hidden_layers = len(model.gpt_neox.layers)
    for i, block in enumerate(model.gpt_neox.layers):
        set_layer_idx_recursive(block, i)
    return model
