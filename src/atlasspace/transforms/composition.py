from __future__ import annotations

from atlasspace.runtime.transforms import TransformSequence


def invert_transform_sequence(
    transform_sequence: TransformSequence,
) -> TransformSequence:
    return transform_sequence.inverted()


def concatenate_transform_sequences(
    first: TransformSequence,
    second: TransformSequence,
) -> TransformSequence:
    if first.target_space != second.source_space:
        raise ValueError(
            "Transform sequences are not composable: "
            "first.target_space must equal second.source_space."
        )

    return TransformSequence(
        source_space=first.source_space,
        target_space=second.target_space,
        forward_paths=list(second.forward_paths) + list(first.forward_paths),
        inverse_paths=list(first.inverse_paths) + list(second.inverse_paths),
    )
