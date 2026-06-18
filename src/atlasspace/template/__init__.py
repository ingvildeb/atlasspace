from atlasspace.template.averaging import (
    accumulate_template_inputs,
    blend_template_with_new_average,
    build_mean_confidence_image,
    build_mean_weight_image,
    build_support_count_image,
    build_weight_sum_image,
    finalize_plain_average_template,
    finalize_weighted_average_template,
)
from atlasspace.template.confidence import (
    build_confidence_map,
    confidence_to_weight_map,
)
from atlasspace.runtime.template import TemplateAccumulationResult

__all__ = [
    "TemplateAccumulationResult",
    "accumulate_template_inputs",
    "finalize_weighted_average_template",
    "finalize_plain_average_template",
    "build_weight_sum_image",
    "build_support_count_image",
    "build_mean_weight_image",
    "build_mean_confidence_image",
    "blend_template_with_new_average",
    "build_confidence_map",
    "confidence_to_weight_map",
]
