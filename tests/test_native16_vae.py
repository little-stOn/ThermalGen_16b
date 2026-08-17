from pathlib import Path

import numpy as np
import pytest

from native16_gligen.native16_vae import assert_clean_official_parent
from native16_gligen.thermal16 import RadiometricProfile, decode_native16, encode_native16


def test_native16_codec_is_roundtrip_and_absolute():
    profile = RadiometricProfile(storage_min=0.0, storage_max=65535.0)
    raw = np.array([[23, 43], [523, 543]], dtype=np.uint16)
    encoded = encode_native16(raw, profile)
    assert encoded.shape == (1, 2, 2)
    assert encoded[0, 1, 0] > encoded[0, 0, 0]
    np.testing.assert_array_equal(decode_native16(encoded, profile), raw)


def test_native16_codec_rejects_rgb():
    profile = RadiometricProfile()
    rgb = np.zeros((2, 2, 3), dtype=np.uint16)
    try:
        encode_native16(rgb, profile)
    except TypeError:
        pass
    else:
        raise AssertionError("RGB input must not enter the native16 branch")


def test_official_parent_requires_diffusers_vae_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="vae/config.json"):
        assert_clean_official_parent(tmp_path)

    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "config.json").write_text("{}", encoding="utf-8")
    assert assert_clean_official_parent(tmp_path) == tmp_path
