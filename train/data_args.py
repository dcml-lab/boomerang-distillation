from dataclasses import dataclass, field


@dataclass
class DataArguments:
    dataset: str = field(
        default="EleutherAI/the_pile_deduplicated",
        metadata={"help": "Which dataset to train on."},
    )
    max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
