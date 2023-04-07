use num_traits::{Num, NumCast};

pub struct RunningAverage<T, const N: usize> {
    values: [T; N],
    index: usize,
    sum: T,
}

impl<T, const N: usize> RunningAverage<T, N>
where
    T: Copy + Num + NumCast,
{
    pub fn new() -> Self {
        Self {
            values: [T::zero(); N],
            index: 0,
            sum: T::zero(),
        }
    }

    pub fn add(&mut self, value: T) {
        self.sum = self.sum - self.values[self.index] + value;
        self.values[self.index] = value;
        self.index = (self.index + 1) % N;
    }

    pub fn average(&self) -> T {
        self.sum / T::from(N).unwrap()
    }
}
