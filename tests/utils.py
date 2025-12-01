import numpy as np
from numpy.typing import NDArray


def fast_autocorr(original: NDArray, delayed: NDArray, t: int = 0):
    """Only every 4th sample is considered in order to improve execution time"""
    if t == 0:
        return np.corrcoef([original[::4], delayed[::4]])[1, 0]
    elif t < 0:
        return np.corrcoef([original[-t::4], delayed[:t:4]])[1, 0]
    else:
        return np.corrcoef([original[:-t:4], delayed[t::4]])[1, 0]


def find_best_alignment_offset_with_corr_coef(
    reference_signal: NDArray[np.float32],
    delayed_signal: NDArray[np.float32],
    min_offset_samples: int,
    max_offset_samples: int,
    lookahead_samples: int | None = None,
    consider_both_polarities: bool = True,
):
    """
    Returns the estimated delay (in samples) between the original and delayed signal,
    calculated using correlation coefficients. The delay is optimized to maximize the
    correlation between the signals.

    Args:
        reference_signal (NDArray[np.float32]): The original signal array.
        delayed_signal (NDArray[np.float32]): The delayed signal array.
        min_offset_samples (int): The minimum delay offset to consider, in samples.
                                  Can be negative.
        max_offset_samples (int): The maximum delay offset to consider, in samples.
        lookahead_samples (Optional[int]): The number of samples to look at
                                           while estimating the delay. If None, the
                                           whole delayed signal is considered.
        consider_both_polarities (bool): If True, the function will consider both positive
                                         and negative correlations, which corresponds to
                                         the same or opposite polarities in signals,
                                         respectively. Defaults to True.

    Returns:
        tuple: Estimated delay (int) and correlation coefficient (float).
    """
    if lookahead_samples is not None and len(reference_signal) > lookahead_samples:
        middle_of_signal_index = int(np.floor(len(reference_signal) / 2))
        original_signal_slice = reference_signal[
            middle_of_signal_index : middle_of_signal_index + lookahead_samples
        ]
        delayed_signal_slice = delayed_signal[
            middle_of_signal_index : middle_of_signal_index + lookahead_samples
        ]
    else:
        original_signal_slice = reference_signal
        delayed_signal_slice = delayed_signal

    coefs = []
    for lag in range(min_offset_samples, max_offset_samples):
        correlation_coef = fast_autocorr(
            original_signal_slice, delayed_signal_slice, t=lag
        )
        coefs.append(correlation_coef)

    if consider_both_polarities:
        # In this mode we aim to find the correlation coefficient of highest magnitude.
        # We do this to consider the possibility that the delayed signal has opposite
        # polarity compared to the original signal, in which case the correlation
        # coefficient would be negative.
        most_extreme_coef_index = int(np.argmax(np.abs(coefs)))
    else:
        most_extreme_coef_index = int(np.argmax(coefs))
    most_extreme_coef = coefs[most_extreme_coef_index]
    offset = most_extreme_coef_index + min_offset_samples
    return offset, most_extreme_coef
