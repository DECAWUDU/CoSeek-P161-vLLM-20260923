import numpy as np

from videoseek.tools import frame_verify as verifier


def test_source_header_preserves_video_pixels_and_image_limit():
    frame = np.full((1024, 720, 3), [36, 90, 160], dtype=np.uint8)
    original = frame.copy()
    labeled = verifier._image_from_data_url(verifier._frame_data_url(frame, max_side=1024, label='F2505'))
    pixels = np.asarray(labeled)
    assert max(labeled.size) <= 1024
    assert np.array_equal(frame, original)
    assert (pixels[3:27, 8:100].max(axis=2) > 200).any()
    assert np.abs(pixels[50:-10, 10:-10].mean(axis=(0, 1)) - [36, 90, 160]).max() < 3


def test_source_id_changes_header_without_changing_video_region():
    frame = np.full((200, 300, 3), 80, dtype=np.uint8)
    first = np.asarray(verifier._image_from_data_url(verifier._frame_data_url(frame, label='F100')))
    second = np.asarray(verifier._image_from_data_url(verifier._frame_data_url(frame, label='F200')))
    plain = verifier._image_from_data_url(verifier._frame_data_url(frame))
    assert plain.size == (300, 200)
    assert not np.array_equal(first[:32], second[:32])
    assert np.array_equal(first[48:], second[48:])
