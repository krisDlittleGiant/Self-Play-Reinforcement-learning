"""A D5 worker may never silently take an unlisted or duplicate HPU."""

import pytest

from self_play_grpo.training.roles import role_layout_from_modules


def test_explicit_four_plus_four_layout():
    layout = role_layout_from_modules("2,3,5,6,0,1,4,7")
    assert layout.rollout_modules == (2, 3, 5, 6)
    assert layout.trainer_modules == (0, 1, 4, 7)


@pytest.mark.parametrize("value", ["", "0,1,2,3", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,-1", "0,1,2,3,4,5,6,x"])
def test_invalid_layout_rejected(value):
    with pytest.raises(ValueError):
        role_layout_from_modules(value)
