from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from atlasspace.config.config_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.runtime.transforms import TransformSequence
from atlasspace.transforms import transform_image, transform_segmentation
from atlasspace.transforms.antspy_transformation import ants, apply_transform_paths_to_image


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

REGISTRATION_DIR = Path(
    r"Z:\path\to\registration_output_folder"
)
SOURCE_IMAGE_PATH = Path(
    r"Z:\path\to\source_template_image.nii.gz"
)
SOURCE_MASK_PATH = Path(
    r"Z:\path\to\source_template_mask.nii.gz"
)
REFERENCE_IMAGE_PATH = Path(
    r"Z:\path\to\reference_subject_image.nii.gz"
)

SOURCE_SPACE = SpaceDefinition(
    space_name="template_space",
    orientation="lsp",
    resolution_um=(20.0, 20.0, 20.0),
)
TARGET_SPACE = SpaceDefinition(
    space_name="subject_space",
    orientation="las",
    resolution_um=(20.0, 20.0, 20.0),
)
TARGET_TRANSFORM_SPACE = SpaceDefinition(
    space_name="subject_space",
    orientation="lsp",
    resolution_um=(20.0, 20.0, 20.0),
)

OUTPUT_FOLDER_NAME = "debug_transform_application5"


def print_nifti_stats(label: str, path: Path) -> None:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    print(f"{label}:")
    print(f"  path={path}")
    print(f"  shape={image.shape}")
    print(f"  min={float(np.min(data))}")
    print(f"  max={float(np.max(data))}")
    print(f"  mean={float(np.mean(data))}")
    print(f"  nonzero_fraction={float(np.count_nonzero(data) / data.size)}")
    print(f"  affine=\n{image.affine}")


registration_dir = REGISTRATION_DIR
output_dir = registration_dir / OUTPUT_FOLDER_NAME
output_dir.mkdir(parents=True, exist_ok=True)

source_config = ImageConfig(
    image_id="updated_template",
    image=SOURCE_IMAGE_PATH,
    space=SOURCE_SPACE,
)
source_mask_config = ImageConfig(
    image_id="template_mask",
    image=SOURCE_MASK_PATH,
    space=SOURCE_SPACE,
)
reference_config = ImageConfig(
    image_id="subject_native",
    image=REFERENCE_IMAGE_PATH,
    space=TARGET_SPACE,
)
subject_config = ImageConfig(
    image_id="subject_native",
    image=REFERENCE_IMAGE_PATH,
    space=TARGET_SPACE,
)
template_reference_config = ImageConfig(
    image_id="template_reference",
    image=SOURCE_IMAGE_PATH,
    space=SOURCE_SPACE,
)

transform_sequence = TransformSequence.from_antspy_output(
    registration_dir,
    source_space=SOURCE_SPACE,
    target_space=TARGET_SPACE,
    source_transform_space=SOURCE_SPACE,
    target_transform_space=TARGET_TRANSFORM_SPACE,
)

transform_layer_result = transform_image(
    source_config,
    transform_sequence,
    reference_config,
    direction="forward",
    interpolation="linear",
    output_dir=output_dir,
    write_intermediates=True,
)

transform_layer_output = transform_layer_result.image
prepared_reference_output = transform_layer_output.with_name(
    transform_layer_output.name.replace(".nii.gz", "_prepared_reference.nii.gz")
)
prepared_input_output = transform_layer_output.with_name(
    transform_layer_output.name.replace(".nii.gz", "_prepared_input.nii.gz")
)

segmentation_transform_result = transform_segmentation(
    source_mask_config,
    transform_sequence,
    reference_config,
    direction="forward",
    interpolation="nearestNeighbor",
    output_dir=output_dir,
    write_intermediates=False,
)
segmentation_transform_output = segmentation_transform_result.image

inverse_segmentation_result = transform_segmentation(
    segmentation_transform_result,
    transform_sequence,
    template_reference_config,
    direction="inverse",
    interpolation="nearestNeighbor",
    output_dir=output_dir,
    write_intermediates=False,
)
inverse_segmentation_output = inverse_segmentation_result.image

inverse_transform_result = transform_image(
    subject_config,
    transform_sequence,
    template_reference_config,
    direction="inverse",
    interpolation="linear",
    output_dir=output_dir,
    write_intermediates=False,
)
inverse_transform_output = inverse_transform_result.image

source_ants = ants.image_read(str(prepared_input_output))
reference_ants = ants.image_read(str(prepared_reference_output))
direct_ants = apply_transform_paths_to_image(
    source_ants,
    reference_ants,
    transform_sequence.paths_for_direction("forward"),
    interpolation="linear",
)
inverse_source_ants = ants.image_read(str(REFERENCE_IMAGE_PATH))
inverse_reference_ants = ants.image_read(str(SOURCE_IMAGE_PATH))
direct_inverse_ants = apply_transform_paths_to_image(
    inverse_source_ants,
    inverse_reference_ants,
    transform_sequence.paths_for_direction("inverse"),
    interpolation="linear",
)

saved_original_output = registration_dir / "ANTsPy_Warped.nii.gz"
saved_inverse_output = registration_dir / "ANTsPy_InverseWarped.nii.gz"

print("TransformSequence")
print(f"  forward_paths={transform_sequence.forward_paths}")
print(f"  inverse_paths={transform_sequence.inverse_paths}")
print()

print("Arrays in memory")
print(f"  source_ants_shape={source_ants.shape}")
print(f"  reference_ants_shape={reference_ants.shape}")
print(f"  direct_ants_shape={direct_ants.shape}")
print(f"  direct_ants_numpy_shape={direct_ants.numpy().shape}")
print(f"  transform_layer_output={transform_layer_result.image}")
print(f"  segmentation_transform_output={segmentation_transform_output}")
print(f"  inverse_segmentation_output={inverse_segmentation_output}")
print(f"  inverse_transform_output={inverse_transform_output}")
print(f"  source_transform_space={transform_sequence.source_transform_space}")
print(f"  target_transform_space={transform_sequence.target_transform_space}")
print()

print_nifti_stats("saved_original_output", saved_original_output)
print_nifti_stats("saved_inverse_output", saved_inverse_output)
print_nifti_stats("prepared_input_output", prepared_input_output)
print_nifti_stats("prepared_reference_output", prepared_reference_output)
print_nifti_stats("transform_layer_output", transform_layer_output)
print_nifti_stats("segmentation_transform_output", segmentation_transform_output)
print_nifti_stats("inverse_segmentation_output", inverse_segmentation_output)
print_nifti_stats("inverse_transform_output", inverse_transform_output)

transform_layer_data = np.asarray(nib.load(str(transform_layer_output)).dataobj, dtype=np.float32)
direct_ants_data = direct_ants.numpy().astype(np.float32, copy=False)
segmentation_data = np.asarray(nib.load(str(segmentation_transform_output)).dataobj)
inverse_transform_data = np.asarray(nib.load(str(inverse_transform_output)).dataobj, dtype=np.float32)
direct_inverse_data = direct_inverse_ants.numpy().astype(np.float32, copy=False)
saved_inverse_data = np.asarray(nib.load(str(saved_inverse_output)).dataobj, dtype=np.float32)
inverse_segmentation_data = np.asarray(nib.load(str(inverse_segmentation_output)).dataobj)
original_segmentation_data = np.asarray(nib.load(str(SOURCE_MASK_PATH)).dataobj)

print()
print("Output comparisons")
print(
    "  transform_output_vs_direct_ants_max_abs_diff="
    f"{float(np.max(np.abs(transform_layer_data - direct_ants_data)))}"
)
print(
    "  transform_output_vs_direct_ants_mean_abs_diff="
    f"{float(np.mean(np.abs(transform_layer_data - direct_ants_data)))}"
)
print(
    "  inverse_transform_output_vs_direct_inverse_ants_max_abs_diff="
    f"{float(np.max(np.abs(inverse_transform_data - direct_inverse_data)))}"
)
print(
    "  inverse_transform_output_vs_direct_inverse_ants_mean_abs_diff="
    f"{float(np.mean(np.abs(inverse_transform_data - direct_inverse_data)))}"
)
print(
    "  inverse_transform_output_vs_saved_inverse_max_abs_diff="
    f"{float(np.max(np.abs(inverse_transform_data - saved_inverse_data)))}"
)
print(
    "  inverse_transform_output_vs_saved_inverse_mean_abs_diff="
    f"{float(np.mean(np.abs(inverse_transform_data - saved_inverse_data)))}"
)
print(f"  segmentation_unique_values={np.unique(segmentation_data)[:20]}")
print(f"  inverse_segmentation_unique_values={np.unique(inverse_segmentation_data)[:20]}")
print(
    "  inverse_segmentation_vs_original_max_abs_diff="
    f"{float(np.max(np.abs(inverse_segmentation_data.astype(np.float32) - original_segmentation_data.astype(np.float32))))}"
)
