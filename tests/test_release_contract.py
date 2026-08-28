from atlasspace import registration


def test_registration_result_contract_is_public() -> None:
    assert registration.REGISTRATION_RESULT_FILENAME == "registration_result.json"
    assert registration.REGISTRATION_RESULT_SCHEMA_VERSION == 1
    assert callable(registration.load_registration_result_manifest)
    assert callable(registration.migrate_legacy_registration_output)
