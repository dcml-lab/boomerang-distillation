from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    teacher_model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen3-4B-Base",
        metadata={"help": "Path or huggingface identifier of teacher model"},
    )
    alternate_every_n_layers: int = field(
        default=2,
        metadata={
            "help": "Keep every n-th layer from the teacher model when initializing the student. For example, if set to 2, every second layer will be kept."
        },
    )
    first_layers_to_keep: int = field(
        default=1,
        metadata={
            "help": "Number of initial layers to keep from the teacher model when initializing the student. This is in addition to the layers kept by alternate_every_n_layers."
        },
    )
    last_layers_to_keep: int = field(
        default=2,
        metadata={
            "help": "Number of final layers to keep from the teacher model when initializing the student. This is in addition to the layers kept by alternate_every_n_layers."
        },
    )
    random_initialization: bool = field(
        default=False,
        metadata={
            "help": "If True, initialize the student model with random weights instead of copying from the teacher."
        },
    )
