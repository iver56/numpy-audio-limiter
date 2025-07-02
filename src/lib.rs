use numpy::{
    ndarray::{Axis, Ix2},
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArrayDyn,
};
use pyo3::{exceptions::PyValueError, prelude::*, Bound};

struct Limiter {
    attack: f32,
    release: f32,
    threshold: f32,
    delay: usize,
    delay_idx: usize,
    envelope: f32,
    gain: f32,
    delay_line: Vec<f32>,
}

impl Limiter {
    fn new(attack: f32, release: f32, delay: usize, threshold: f32) -> Self {
        Self {
            attack,
            release,
            threshold,
            delay,
            delay_idx: 0,
            envelope: 0.0,
            gain: 1.0,
            delay_line: vec![0.0; delay],
        }
    }

    fn process(&mut self, data: &mut [f32]) {
        let len = data.len();

        for i in 0..(len + self.delay) {
            let current = if i < len { data[i] } else { 0.0 };
            let delayed = self.delay_line[self.delay_idx];

            self.envelope = (self.envelope * self.release).max(current.abs());

            let target_gain = if self.envelope > self.threshold {
                self.threshold / self.envelope
            } else {
                1.0
            };
            self.gain = self.gain * self.attack + target_gain * (1.0 - self.attack);

            if i >= self.delay {
                data[i - self.delay] = delayed * self.gain;
            }

            self.delay_line[self.delay_idx] = current;
            self.delay_idx += 1;
            if self.delay_idx == self.delay {
                self.delay_idx = 0;
            }
        }
    }
}

#[pyfunction]
fn limit<'py>(
    py: Python<'py>,
    signal: PyReadonlyArrayDyn<'py, f32>,
    attack_coeff: f32,
    release_coeff: f32,
    delay: usize,
    threshold: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut out = signal
        .to_owned_array()
        .into_dimensionality::<Ix2>()
        .map_err(|_| PyValueError::new_err("signal must have shape (channels, samples)"))?;

    for mut chan in out.axis_iter_mut(Axis(0)) {
        let slice = chan.as_slice_mut().expect("channel not contiguous");
        let mut lim = Limiter::new(attack_coeff, release_coeff, delay, threshold);
        lim.process(slice);
    }

    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn numpy_audio_limiter(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(limit, m)?)?;
    Ok(())
}
