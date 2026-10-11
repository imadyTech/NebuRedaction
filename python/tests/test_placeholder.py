import neburedaction


def test_version_is_a_three_part_version_string():
    parts = neburedaction.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_no_model_is_embedded_before_one_is_trained():
    assert neburedaction.has_embedded_model() is False
