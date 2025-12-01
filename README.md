# numpy-audio-limiter

A simple Python library for dynamic range compression of audio. Made for integration with [audiomentations](https://github.com/iver56/audiomentations/). Based on cylimiter (C++) by pzelasko, but has a few extras for fast delay compensation. The main motivation for porting it to Rust was that cylimiter appeared to be unmaintained and did not install well on Python 3.12 anymore. As audiomentations maintainer, I aim to keep it easy to install for relevant Python versions and operating systems, without fiddling with compilers and special environment variables. An extra bonus with this port is that it is **~30% faster** than cylimiter (in the context of audiomentations).

# Installation

[![PyPI version](https://img.shields.io/pypi/v/numpy-audio-limiter.svg?style=flat)](https://pypi.org/project/numpy-audio-limiter/)
![python 3.10, 3.11, 3.12, 3.13, 3.14](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)
![os: Linux, macOS, Windows](https://img.shields.io/badge/OS-Linux%20%28arm%20%26%20x86--64%29%20|%20macOS%20%28arm%29%20|%20Windows%20%28x86--64%29-blue)

```
$ pip install numpy-audio-limiter
```

## Code example

```
import numpy as np
import numpy_audio_limiter

x = np.random.randn(1800).astype(np.float32)
y = numpy_audio_limiter.limit(
    signal=x.reshape((1, -1)),
    attack_coeff=0.99,
    release_coeff=0.99,
    delay=527,
    threshold=0.5,
)
assert y.shape == (1, 1800)
```

## Features

* The output is aligned with the input.
* Supports mono and multichannel audio. Mono audio is represented as a 2D-array with 1 in the first dimension.
* Adjustable threshold, which determines the audio level above which the limiter kicks in.
* Adjustable attack time, which denotes how quickly the limiter kicks in once the audio signal starts exceeding the threshold.
* Adjustable release time, which determines how quickly the limiter stops working after the signal drops below the threshold.

## Changelog

## [0.1.0] - 2025-07-02

Initial release

For the complete changelog, go to [CHANGELOG.md](CHANGELOG.md)

## Development setup

* `conda create --name numpy-audio-limiter python=3.11`
* `conda activate numpy-audio-limiter`
* `pip install -r dev_requirements.txt`
* `maturin develop`
* `pytest`
