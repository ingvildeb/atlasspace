from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(r"C:\Users\SmartBrain_32C_TR\Documents\GitHub\atlasbuilder")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from atlasbuilder.config.config_models import ImageConfig  # noqa: E402
from atlasbuilder.config.space_models import SpaceDefinition  # noqa: E402
from atlasbuilder.image.masking import apply_binary_mask  # noqa: E402
from atlasbuilder.image.pose_standardization import (  # noqa: E402
    TemplateLandmarks,
    standardize_image_pose,
)


LANDMARK_FILE = Path(
    r"Z:\Labmembers\Ingvild\Atlasing\LSFM_templates_working_area\orientation_testing\template_landmarks.xlsx"
)
MASK_FILE = Path(
    r"Z:\Labmembers\Ingvild\RM1\atlas_templates\P56\20um\template_updates\current_template\T_P56_iteration0_n5_NeuN_template0_20um_mask.nii.gz"
)
TEMPLATE_FILE = Path(
    r"Z:\Labmembers\Ingvild\RM1\atlas_templates\P56\20um\template_variants\T_P56_iteration0_n5_NeuN_template0.nii.gz"
)

TEMPLATE_NAME = "LSFM-NeuN-P56"
REFERENCE_TEMPLATE_NAME = "CCFv3"
COORDINATE_BASE = 1
INTERPOLATION = "linear"
FILL_VALUE = 0.0
TEMPLATE_ORIENTATION = "lsp"
TEMPLATE_RESOLUTION_UM = (20.0, 20.0, 20.0)


def _with_suffix(path: Path, suffix: str) -> Path:
    if path.name.endswith(".nii.gz"):
        base = path.name[:-7]
        return path.with_name(f"{base}{suffix}.nii.gz")
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _parse_coordinate_triplet(
    row: pd.Series,
    columns: tuple[str, str, str],
    label: str,
) -> np.ndarray:
    values: list[int] = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            raise ValueError(f"Encountered an empty value in column '{column}' for {label}.")
        try:
            values.append(int(value))
        except ValueError as exc:
            raise ValueError(
                f"Column '{column}' for {label} contains a non-integer value: {value}"
            ) from exc
    return np.asarray(values, dtype=np.float64)


def _load_landmarks(
    sheet_path: Path,
    template_name: str,
    reference_name: str,
) -> tuple[TemplateLandmarks, TemplateLandmarks]:
    df = pd.read_excel(sheet_path)
    required_columns = {"template", "Ws", "Wh", "Wc", "CCs", "CCh", "CCc"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Coordinate sheet is missing required columns: {sorted(missing_columns)}"
        )

    template_row = df.loc[df["template"] == template_name]
    if template_row.empty:
        raise ValueError(f"Template '{template_name}' was not found in {sheet_path}.")
    if len(template_row) > 1:
        raise ValueError(f"Template '{template_name}' appears multiple times in {sheet_path}.")

    reference_row = df.loc[df["template"] == reference_name]
    if reference_row.empty:
        raise ValueError(f"Reference template '{reference_name}' was not found in {sheet_path}.")
    if len(reference_row) > 1:
        raise ValueError(
            f"Reference template '{reference_name}' appears multiple times in {sheet_path}."
        )

    template_row = template_row.iloc[0]
    reference_row = reference_row.iloc[0]

    source_landmarks = TemplateLandmarks(
        whs=_parse_coordinate_triplet(template_row, ("Ws", "Wh", "Wc"), f"{template_name} WHS"),
        central_canal=_parse_coordinate_triplet(
            template_row,
            ("CCs", "CCh", "CCc"),
            f"{template_name} central canal",
        ),
    )
    reference_landmarks = TemplateLandmarks(
        whs=_parse_coordinate_triplet(
            reference_row,
            ("Ws", "Wh", "Wc"),
            f"{reference_name} WHS",
        ),
        central_canal=_parse_coordinate_triplet(
            reference_row,
            ("CCs", "CCh", "CCc"),
            f"{reference_name} central canal",
        ),
    )
    return source_landmarks, reference_landmarks


def main() -> None:
    masked_output = _with_suffix(TEMPLATE_FILE, "_masked")
    standardized_output = _with_suffix(masked_output, "_pose_standardized")
    template_config = ImageConfig(
        image_id=TEMPLATE_FILE.stem.split(".")[0],
        image=TEMPLATE_FILE,
        space=SpaceDefinition(
            space_name=TEMPLATE_NAME,
            orientation=TEMPLATE_ORIENTATION,
            resolution_um=TEMPLATE_RESOLUTION_UM,
        ),
    )
    mask_config = ImageConfig(
        image_id=MASK_FILE.stem.split(".")[0],
        image=MASK_FILE,
        space=SpaceDefinition(
            space_name=f"{TEMPLATE_NAME}_mask",
            orientation=TEMPLATE_ORIENTATION,
            resolution_um=TEMPLATE_RESOLUTION_UM,
        ),
    )

    print(f"Masking template:\n  image: {TEMPLATE_FILE}\n  mask:  {MASK_FILE}")
    masked_config = apply_binary_mask(
        image_config=template_config,
        mask_config=mask_config,
        output_path=masked_output,
        fill_value=FILL_VALUE,
    )
    print(f"Masked template saved to:\n  {masked_output}")

    source_landmarks, reference_landmarks = _load_landmarks(
        LANDMARK_FILE,
        TEMPLATE_NAME,
        REFERENCE_TEMPLATE_NAME,
    )

    print(
        "Standardizing pose using landmarks:\n"
        f"  source template:   {TEMPLATE_NAME}\n"
        f"  reference template:{REFERENCE_TEMPLATE_NAME}"
    )
    standardize_image_pose(
        image_config=masked_config,
        output_path=standardized_output,
        source_landmarks=source_landmarks,
        reference_landmarks=reference_landmarks,
        coordinate_base=COORDINATE_BASE,
        interpolation=INTERPOLATION,
        fill_value=FILL_VALUE,
    )
    print(f"Pose-standardized template saved to:\n  {standardized_output}")


if __name__ == "__main__":
    main()
