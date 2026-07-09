from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atlasspace.config.image_models import ImageConfig


@dataclass
class TemplateAccumulationResult:
    reference_config: ImageConfig
    weighted_sum: np.ndarray
    weight_sum: np.ndarray
    plain_sum: np.ndarray
    valid_support_count: np.ndarray
    support_count: np.ndarray
    confidence_sum: np.ndarray | None = None
    subject_count: int = 0
