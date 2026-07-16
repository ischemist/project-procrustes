//! Deterministic sampling compatible with `random.Random` in CPython.
//!
//! Curation outputs are durable datasets, so their seeded selection order is
//! part of RetroCast's behavior. This module reproduces CPython's integer seed,
//! MT19937, `_randbelow`, and `sample` path instead of merely choosing the same
//! family of random-number generator.

use std::collections::HashSet;

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const STATE_LEN: usize = 624;
const STATE_MIDDLE: usize = 397;

#[derive(Debug, Error)]
pub enum SamplingError {
    #[error("invalid integer seed {0:?}")]
    InvalidSeed(String),
    #[error("cannot sample {sample_size} from {population_size} items")]
    Oversampling {
        sample_size: usize,
        population_size: usize,
    },
    #[error("group target count does not match the grouped pool dimensions")]
    GroupCountMismatch,
}

/// The location of a selected item in priority-ordered, pre-grouped pools.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SampleCoordinate {
    pub pool: usize,
    pub index: usize,
}

/// Select population indexes in exactly the order produced by CPython 3.11.
pub fn sample_indices(
    population_size: usize,
    sample_size: usize,
    seed: &str,
) -> Result<Vec<usize>, SamplingError> {
    let mut random = PythonRandom::from_integer(seed)?;
    random.sample_indices(population_size, sample_size)
}

/// Fill each group from priority-ordered pools while sharing one RNG stream.
///
/// Python callers may supply an arbitrary grouping callback. They evaluate that
/// callback once, then pass only the resulting group sizes here; all selection
/// policy and random state remain owned by the core.
pub fn sample_stratified_priority_indices(
    grouped_pool_sizes: &[Vec<usize>],
    target_counts: &[usize],
    seed: &str,
) -> Result<Vec<Vec<SampleCoordinate>>, SamplingError> {
    if grouped_pool_sizes.len() != target_counts.len() {
        return Err(SamplingError::GroupCountMismatch);
    }

    let mut random = PythonRandom::from_integer(seed)?;
    let mut selected_groups = Vec::with_capacity(target_counts.len());
    for (pool_sizes, &target_count) in grouped_pool_sizes.iter().zip(target_counts) {
        let mut selected = Vec::with_capacity(target_count);
        for (pool, &available) in pool_sizes.iter().enumerate() {
            let needed = target_count.saturating_sub(selected.len());
            if needed == 0 {
                break;
            }
            if available > needed {
                selected.extend(
                    random
                        .sample_indices(available, needed)?
                        .into_iter()
                        .map(|index| SampleCoordinate { pool, index }),
                );
            } else {
                selected.extend((0..available).map(|index| SampleCoordinate { pool, index }));
            }
        }
        selected_groups.push(selected);
    }
    Ok(selected_groups)
}

/// Sample indexes from several groups using one uninterrupted RNG stream.
pub fn sample_index_groups(
    groups: &[Vec<usize>],
    sample_sizes: &[usize],
    seed: &str,
) -> Result<Vec<Vec<usize>>, SamplingError> {
    if groups.len() != sample_sizes.len() {
        return Err(SamplingError::GroupCountMismatch);
    }
    let mut random = PythonRandom::from_integer(seed)?;
    groups
        .iter()
        .zip(sample_sizes)
        .map(|(group, &sample_size)| {
            random
                .sample_indices(group.len(), sample_size)
                .map(|selected| selected.into_iter().map(|index| group[index]).collect())
        })
        .collect()
}

struct PythonRandom {
    state: [u32; STATE_LEN],
    index: usize,
}

impl PythonRandom {
    fn from_integer(seed: &str) -> Result<Self, SamplingError> {
        let magnitude = seed.strip_prefix(['-', '+']).unwrap_or(seed).as_bytes();
        let value = BigUint::parse_bytes(magnitude, 10)
            .ok_or_else(|| SamplingError::InvalidSeed(seed.to_owned()))?;
        let mut words = value.to_u32_digits();
        if words.is_empty() {
            words.push(0);
        }
        Ok(Self::from_seed_words(&words))
    }

    fn from_seed_words(words: &[u32]) -> Self {
        let mut random = Self {
            state: [0; STATE_LEN],
            index: STATE_LEN,
        };
        random.init_genrand(19_650_218);

        let mut state_index = 1;
        let mut word_index = 0;
        for _ in 0..STATE_LEN.max(words.len()) {
            let previous = random.state[state_index - 1];
            random.state[state_index] = (random.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(words[word_index])
            .wrapping_add(word_index as u32);
            state_index += 1;
            word_index += 1;
            if state_index >= STATE_LEN {
                random.state[0] = random.state[STATE_LEN - 1];
                state_index = 1;
            }
            if word_index >= words.len() {
                word_index = 0;
            }
        }
        for _ in 0..STATE_LEN - 1 {
            let previous = random.state[state_index - 1];
            random.state[state_index] = (random.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(state_index as u32);
            state_index += 1;
            if state_index >= STATE_LEN {
                random.state[0] = random.state[STATE_LEN - 1];
                state_index = 1;
            }
        }
        random.state[0] = 0x8000_0000;
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..STATE_LEN {
            let previous = self.state[index - 1];
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= STATE_LEN {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn twist(&mut self) {
        for index in 0..STATE_LEN {
            let value = (self.state[index] & 0x8000_0000)
                | (self.state[(index + 1) % STATE_LEN] & 0x7fff_ffff);
            self.state[index] = self.state[(index + STATE_MIDDLE) % STATE_LEN]
                ^ (value >> 1)
                ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
        }
        self.index = 0;
    }

    fn getrandbits(&mut self, bits: u32) -> u64 {
        if bits <= 32 {
            return u64::from(self.next_u32() >> (32 - bits));
        }
        let mut value = 0_u64;
        let words = bits.div_ceil(32);
        for word_index in 0..words {
            let remaining = bits - word_index * 32;
            let word = if remaining < 32 {
                self.next_u32() >> (32 - remaining)
            } else {
                self.next_u32()
            };
            value |= u64::from(word) << (word_index * 32);
        }
        value
    }

    fn randbelow(&mut self, upper_bound: usize) -> usize {
        debug_assert!(upper_bound > 0);
        let bits = usize::BITS - upper_bound.leading_zeros();
        loop {
            let value = self.getrandbits(bits) as usize;
            if value < upper_bound {
                return value;
            }
        }
    }

    fn sample_indices(
        &mut self,
        population_size: usize,
        sample_size: usize,
    ) -> Result<Vec<usize>, SamplingError> {
        if sample_size > population_size {
            return Err(SamplingError::Oversampling {
                sample_size,
                population_size,
            });
        }

        let mut set_size = 21;
        if sample_size > 5 {
            let mut table_size = 4;
            while table_size < sample_size * 3 {
                table_size *= 4;
            }
            set_size += table_size;
        }

        if population_size <= set_size {
            let mut pool: Vec<_> = (0..population_size).collect();
            let mut selected = Vec::with_capacity(sample_size);
            for index in 0..sample_size {
                let pool_index = self.randbelow(population_size - index);
                selected.push(pool[pool_index]);
                pool[pool_index] = pool[population_size - index - 1];
            }
            Ok(selected)
        } else {
            let mut seen = HashSet::with_capacity(sample_size);
            let mut selected = Vec::with_capacity(sample_size);
            while selected.len() < sample_size {
                let index = self.randbelow(population_size);
                if seen.insert(index) {
                    selected.push(index);
                }
            }
            Ok(selected)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython_getrandbits() {
        let mut random = PythonRandom::from_integer("1").unwrap();
        assert_eq!(random.getrandbits(1), 0);
        assert_eq!(random.getrandbits(2), 2);
        assert_eq!(random.getrandbits(5), 27);
        assert_eq!(random.getrandbits(32), 3_445_702_192);
        assert_eq!(random.getrandbits(33), 3_280_387_012);
        assert_eq!(random.getrandbits(64), 2_175_216_119_781_798_972);
    }

    #[test]
    fn matches_cpython_sample_paths() {
        assert_eq!(sample_indices(4, 2, "1").unwrap(), [1, 2]);
        assert_eq!(sample_indices(100, 6, "7").unwrap(), [41, 19, 50, 83, 6, 9]);
    }

    #[test]
    fn negative_integer_seeds_use_their_magnitude() {
        assert_eq!(
            sample_indices(40, 8, "-7").unwrap(),
            sample_indices(40, 8, "7").unwrap()
        );
    }
}
