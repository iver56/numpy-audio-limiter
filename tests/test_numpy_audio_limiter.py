import numpy as np
import numpy_audio_limiter


def test_mono():
    x = np.random.randn(1800).astype(np.float32)
    y = numpy_audio_limiter.limit(
        signal=x.reshape((1, -1)),
        attack_coeff=0.99,
        release_coeff=0.99,
        delay=527,
        threshold=0.5,
    )
    assert y.shape == (1, 1800)


def test_stereo():
    x = np.random.randn(1800).astype(np.float32).reshape((2, 900))
    y = numpy_audio_limiter.limit(
        signal=x,
        attack_coeff=0.99,
        release_coeff=0.99,
        delay=527,
        threshold=0.5,
    )
    assert y.shape == (2, 900)
