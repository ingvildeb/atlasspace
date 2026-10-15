from atlasspace import registration
from atlasspace import transforms


def test_registration_result_contract_is_public() -> None:
    assert registration.REGISTRATION_RESULT_FILENAME == "registration_result.json"
    assert registration.REGISTRATION_RESULT_SCHEMA_VERSION == 1
    assert callable(registration.load_registration_result_manifest)
    assert callable(registration.migrate_legacy_registration_output)
    assert callable(registration.transform_segmentation_from_registration)
    assert callable(transforms.transform_points_from_registration)
