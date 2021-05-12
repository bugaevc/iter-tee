//! Make several clones of an iterator.
//!
//! Each handle to the iterator is represented with an instance of [`Tee`]. A
//! `Tee` is itself an iterator which will yield the same sequence of items as
//! the original iterator. A `Tee` can be freely cloned at any point to create
//! more handles to the same underlying iterator. Once cloned, the two `Tee`s
//! are identical, but separate: they will yield the same items.
//!
//! The implementation uses a single ring buffer for storing items already
//! pulled from the underlying iterator, but not yet consumed by all the `Tee`s.
//! The buffer is protected with a [`RwLock`], and [atomics](std::sync::atomic)
//! are used to keep item reference counts.
//!
//! While the implementation tries to be efficient, it will not be as efficient
//! as natively cloning the underlying iterator if it implements [`Clone`].
//!
//! # Examples
//!
//! ```
//! use iter_tee::Tee;
//!
//! // Wrap an iterator in a Tee:
//! let mut tee1 = Tee::new(0..10);
//!
//! // It yields the same items:
//! assert_eq!(tee1.next(), Some(0));
//! assert_eq!(tee1.next(), Some(1));
//!
//! // Create a second Tee:
//! let mut tee2 = tee1.clone();
//!
//! // Both yield the same items:
//! assert_eq!(tee1.next(), Some(2));
//! assert_eq!(tee2.next(), Some(2));
//!
//! // Create a third Tee:
//! let mut tee3 = tee2.clone();
//!
//! // All three yield the same items:
//! assert_eq!(tee1.next(), Some(3));
//! assert_eq!(tee2.next(), Some(3));
//! assert_eq!(tee3.next(), Some(3));
//!
//! // The Tees can be advanced independently:
//! assert_eq!(tee1.next(), Some(4));
//! assert_eq!(tee1.next(), Some(5));
//! assert_eq!(tee2.next(), Some(4));
//! ```

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

struct BufferItem<T> {
    value: T,
    ref_count: AtomicUsize,
}

struct Shared<I: Iterator> {
    iter: Option<I>,
    buffer: VecDeque<BufferItem<I::Item>>,
    next_item_ref_count: AtomicUsize,
    num_items_dropped: usize,
}

#[derive(Debug)]
enum Outcome<T> {
    /// The value is ready, and the ref count has already been advanced.
    Ready(Option<T>),
    /// Will need to re-lock for writing and pull the next item from the
    /// iterator. The ref count has not been advanced.
    PastTheBuffer,
    /// Will need to re-lock for writing and take the last item (without cloning
    /// its value). The ref count has not been advanced.
    TakeTail,
    /// The value is ready, but we will need to re-lock for writing and clean up
    /// the tail. The ref count has already been advanced.
    DropTail(T),
}

impl<I> Shared<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn offset(&self, pos: usize) -> usize {
        debug_assert!(pos >= self.num_items_dropped);
        let offset = pos - self.num_items_dropped;
        debug_assert!(offset <= self.buffer.len());
        offset
    }

    fn inc_ref_count(&self, offset: usize) {
        let count = if offset == self.buffer.len() {
            &self.next_item_ref_count
        } else {
            &self.buffer[offset].ref_count
        };
        count.fetch_add(1, Ordering::Relaxed);
    }

    fn dec_ref_count(&self, offset: usize) -> bool {
        let count = if offset == self.buffer.len() {
            &self.next_item_ref_count
        } else {
            &self.buffer[offset].ref_count
        };
        count.fetch_sub(1, Ordering::Relaxed) == 1
    }

    fn advance_ref_count(&self, offset: usize) -> bool {
        self.inc_ref_count(offset + 1);
        self.dec_ref_count(offset)
    }

    fn try_take(&self, offset: usize) -> Outcome<I::Item> {
        if offset == self.buffer.len() {
            // We're past the buffer; need to pull the next
            // item from the iterator. If there is still an
            // iterator in the first place.
            if self.iter.is_some() {
                Outcome::PastTheBuffer
            } else {
                Outcome::Ready(None)
            }
        } else if offset > 0 {
            // Fast path: we're in the middle of the buffer.
            let value = self.buffer[offset].value.clone();
            self.advance_ref_count(offset);
            Outcome::Ready(Some(value))
        } else if self.buffer[0].ref_count.load(Ordering::Relaxed) == 1 {
            // We're the only one still interested in that item;
            // take it without cloning.
            Outcome::TakeTail
        } else {
            let value = self.buffer[0].value.clone();
            let was_last = self.advance_ref_count(0);
            if was_last {
                Outcome::DropTail(value)
            } else {
                Outcome::Ready(Some(value))
            }
        }
    }

    /// Attempts to pull the next item from the iterator.
    ///
    /// Advances the ref count.
    fn pull_next_item(&mut self) -> Option<I::Item> {
        let iter = self.iter.as_mut().expect("iter should not be none here");
        let value = match iter.next() {
            Some(value) => value,
            None => {
                // We have exhausted the underlying iterator; drop it.
                self.iter = None;
                return None;
            }
        };
        if self.buffer.is_empty() && *self.next_item_ref_count.get_mut() == 1 {
            // We're the only consumer out there!
            // Skip the buffering altogether.
            self.num_items_dropped += 1;
            return Some(value);
        }
        // So far, we're the only one interested in the *next* next item.
        let new_item_ref_count = std::mem::replace(self.next_item_ref_count.get_mut(), 1) - 1;
        let new_item = BufferItem {
            value: value.clone(),
            ref_count: AtomicUsize::new(new_item_ref_count),
        };
        self.buffer.push_back(new_item);
        Some(value)
    }

    /// Drops any unused tail of the buffer.
    fn drop_tail(&mut self) {
        while let Some(buffer_item) = self.buffer.front_mut() {
            if *buffer_item.ref_count.get_mut() > 0 {
                break;
            }
            self.buffer.pop_front();
            self.num_items_dropped += 1;
        }
    }

    fn take(this: &RwLock<Self>, pos: usize) -> Option<I::Item> {
        let mut outcome;
        let mut offset;
        // First, lock for reading and see if that's enough.
        {
            let shared = this.read().unwrap();
            offset = shared.offset(pos);
            outcome = shared.try_take(offset);
        };
        if let Outcome::Ready(item) = outcome {
            return item;
        }

        // Now, lock for writing.
        let mut shared = this.write().unwrap();
        // If we were past the buffer, we might be in any situation now.
        // Re-evaluate.
        if let Outcome::PastTheBuffer = outcome {
            offset = shared.offset(pos);
            outcome = shared.try_take(offset);
        }

        match outcome {
            Outcome::Ready(item) => item,
            Outcome::PastTheBuffer => shared.pull_next_item(),
            Outcome::TakeTail => {
                debug_assert_eq!(offset, 0);
                shared.advance_ref_count(0);
                let mut buffer_item = shared
                    .buffer
                    .pop_front()
                    .expect("the buffer should not be empty here");
                debug_assert_eq!(*buffer_item.ref_count.get_mut(), 0);
                shared.num_items_dropped += 1;
                Some(buffer_item.value)
            }
            Outcome::DropTail(item) => {
                debug_assert_eq!(offset, 0);
                shared.drop_tail();
                Some(item)
            }
        }
    }
}

/// Shared iterator handle.
///
/// `Tee`s can be freely cloned at any point to get several independent handles
/// to the same underlying iterator.
pub struct Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    shared: Arc<RwLock<Shared<I>>>,
    pos: usize,
}

impl<I> Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    /// Wraps an iterator into a new `Tee`.
    pub fn new(iter: I) -> Self {
        let shared = Shared {
            iter: Some(iter),
            buffer: VecDeque::new(),
            next_item_ref_count: AtomicUsize::new(1),
            num_items_dropped: 0,
        };
        Tee {
            shared: Arc::new(RwLock::new(shared)),
            pos: 0,
        }
    }
}

impl<I> Clone for Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn clone(&self) -> Self {
        {
            let shared = self.shared.read().unwrap();
            let offset = shared.offset(self.pos);
            shared.inc_ref_count(offset);
        }
        Tee {
            shared: self.shared.clone(),
            pos: self.pos,
        }
    }
}

impl<I> Drop for Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn drop(&mut self) {
        let need_to_drop;

        if let Ok(shared) = self.shared.read() {
            let offset = shared.offset(self.pos);
            let was_last = shared.dec_ref_count(offset);
            need_to_drop = offset == 0 && was_last;
        } else {
            // If the lock is poisoned, do not propagate the panic into this
            // thread. It's fine if we leave an extra ref count.
            return;
        }
        if !need_to_drop {
            return;
        }
        if let Ok(mut shared) = self.shared.write() {
            shared.drop_tail();
        }
    }
}

impl<I> Iterator for Tee<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = Shared::take(&self.shared, self.pos);
        if item.is_some() {
            self.pos += 1;
        }
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let shared = self.shared.read().unwrap();
        let total_buffered = shared.num_items_dropped + shared.buffer.len();
        let more_in_buffer = total_buffered - self.pos;
        let (iter_min, iter_max) = match &shared.iter {
            Some(iter) => iter.size_hint(),
            None => (0, Some(0)),
        };
        (
            more_in_buffer + iter_min,
            iter_max.map(|im| more_in_buffer + im),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Tee;
    use std::{fmt::Debug, thread};

    fn make_string_iter() -> impl Iterator<Item = String> {
        (0..1024).map(|i| i.to_string())
    }

    fn assert_iter_eq<I1, I2>(mut i1: I1, mut i2: I2)
    where
        I1: Iterator,
        I2: Iterator<Item = I1::Item>,
        I1::Item: PartialEq + Debug,
    {
        while let Some(item1) = i1.next() {
            assert_eq!(item1, i2.next().unwrap());
        }
        assert!(i2.next().is_none());
    }

    #[test]
    fn just_one_tee() {
        let tee = Tee::new(make_string_iter());
        assert_iter_eq(tee, make_string_iter());
    }

    #[test]
    fn two_tees() {
        let tee1 = Tee::new(make_string_iter());
        let tee2 = tee1.clone();
        assert_iter_eq(tee1, make_string_iter());
        assert_iter_eq(tee2, make_string_iter());
    }

    #[test]
    fn two_tees_parallel() {
        let tee1 = Tee::new(make_string_iter());
        let tee2 = tee1.clone();
        let t1 = thread::spawn(|| assert_iter_eq(tee1, make_string_iter()));
        let t2 = thread::spawn(|| assert_iter_eq(tee2, make_string_iter()));
        t1.join().unwrap();
        t2.join().unwrap();
    }

    #[test]
    fn ten_tees_parallel() {
        let tee = Tee::new(make_string_iter());
        let mut threads = vec![];
        for tee in vec![tee; 10] {
            let t = thread::spawn(|| assert_iter_eq(tee, make_string_iter()));
            threads.push(t);
        }
        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn drop_in_the_middle() {
        let tee = Tee::new(make_string_iter());
        let mut threads = vec![];
        for (i, tee) in vec![tee; 10].into_iter().enumerate() {
            let t = thread::spawn(move || assert_iter_eq(tee.take(i), make_string_iter().take(i)));
            threads.push(t);
        }
        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn clone_in_the_middle() {
        let mut tee1 = Tee::new(make_string_iter());
        assert_iter_eq(
            tee1.by_ref().take(10),
            make_string_iter().take(10)
        );
        let tee2 = tee1.clone();

        assert_iter_eq(
            tee1,
            make_string_iter().skip(10)
        );
        assert_iter_eq(
            tee2,
            make_string_iter().skip(10)
        );
    }
}
