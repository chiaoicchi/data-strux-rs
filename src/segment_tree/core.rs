use crate::segment_tree::monoid::Monoid;

/// A segment tree for efficient point updates and range queries.
///
/// Given a monoid `(S, op, id)`, this data structure supports:
/// - Point update: `set(i, x)` sets `a[i] = x`
/// - Point operation: `operate(i, x)` sets `a[i] = op(a[i], x)`
/// - Range query: `range_fold(l..r)` returns `op(a[l], op(a[l+1], ..., a[r-1]))`
///
/// Both operations run in O(log n) time.
pub struct SegmentTree<S: Monoid>(
    /// Binary heap-like array storing the tree nodes.
    /// Index 1 is the root, index `size + i` is the leaf for element `i`.
    Box<[S]>,
);

impl<S: Monoid> SegmentTree<S> {
    /// Creates a new segment tree with `n` elements, all initialized to `S::id()`.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn new(n: usize) -> Self {
        Self(vec![S::id(); n << 1].into_boxed_slice())
    }

    /// Creates a new segment tree from a vec.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn from_vec(mut v: Vec<S>) -> Self {
        let n = v.len();
        v.reserve(n);
        unsafe {
            let v = v.as_mut_ptr();
            v.copy_to(v.add(n), n);
            for i in (1..n).rev() {
                v.add(i)
                    .write(S::op(&*v.add(i << 1), &*v.add((i << 1) + 1)));
            }
            v.write(S::id());
        }
        unsafe {
            v.set_len(n << 1);
        }
        Self(v.into_boxed_slice())
    }

    /// Creates a new segment tree from a slice.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn from_slice(v: &[S]) -> Self {
        let n = v.len();
        let mut data = vec![S::id(); n << 1];
        unsafe {
            let d = data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(v.as_ptr(), d.add(n), n);
            for i in (1..n).rev() {
                *d.add(i) = S::op(&*d.add(i << 1), &*d.add((i << 1) + 1));
            }
        }

        Self(data.into_boxed_slice())
    }

    /// Sets the value at index `i` to `x`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()` in debug builds.
    #[inline]
    pub fn set(&mut self, mut i: usize, x: S) {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.len();
        unsafe {
            let d = self.0.as_mut_ptr();
            *d.add(i) = x;
            while i > 1 {
                i >>= 1;
                *d.add(i) = S::op(&*d.add(i << 1), &*d.add((i << 1) + 1));
            }
        }
    }

    /// Applies `op(a[i], x)` to the element at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()` in debug builds.
    #[inline]
    pub fn operate(&mut self, mut i: usize, x: S) {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.len();
        unsafe {
            let d = self.0.as_mut_ptr();
            *d.add(i) = S::op(&*d.add(i), &x);
            while i > 1 {
                i >>= 1;
                *d.add(i) = S::op(&*d.add(i << 1), &*d.add((i << 1) + 1));
            }
        }
    }

    /// Returns the value at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(1)
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()` in debug builds.
    #[inline]
    pub fn get(&self, i: usize) -> S {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        unsafe { self.0.get_unchecked(self.len() + i).clone() }
    }

    /// Returns `op(a[l], a[l+1], ..., a[r-1])` for the given range.
    ///
    /// Returns `S::id()` if the range is empty.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    ///
    /// # Panics
    ///
    /// Panics if the range is invalid or out of bounds in debug builds.
    #[inline]
    pub fn range_fold(&self, range: impl std::ops::RangeBounds<usize>) -> S {
        let mut l = match range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Included(&x) => x,
            std::ops::Bound::Excluded(&x) => x + 1,
        } + self.len();
        let mut r = match range.end_bound() {
            std::ops::Bound::Unbounded => self.len(),
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
        } + self.len();
        debug_assert!(
            l <= r,
            "left bound must be less than or equal to right bound: l={}, r={}",
            l - self.len(),
            r - self.len(),
        );
        debug_assert!(
            r <= self.len() << 1,
            "index out of bounds: r={}, len={}",
            r - self.len(),
            self.len(),
        );
        if l == r {
            return S::id();
        }
        l >>= l.trailing_zeros();
        r >>= r.trailing_zeros();

        let mut left = S::id();
        let mut right = S::id();

        unsafe {
            let d = self.0.as_ptr();
            loop {
                if l >= r {
                    left = S::op(&left, &*d.add(l));
                    l += 1;
                    l >>= l.trailing_zeros();
                } else {
                    r -= 1;
                    right = S::op(&*d.add(r), &right);
                    r >>= r.trailing_zeros();
                }
                if l == r {
                    break;
                }
            }
        }
        S::op(&left, &right)
    }

    /// Returns `op(a[0], a[1], ..., a[n-1])`.
    ///
    /// # Time complexity
    ///
    /// O(1)
    pub fn all_fold(&self) -> S {
        unsafe { self.0.get_unchecked(1).clone() }
    }

    #[inline]
    pub fn max_right<P>(&self, _l: usize, _p: P) -> usize
    where
        P: Fn(&S) -> bool,
    {
        todo!();
    }

    #[inline]
    pub fn min_left<P>(&self, _r: usize, _p: P) -> usize
    where
        P: Fn(&S) -> bool,
    {
        todo!();
    }

    /// Returns the number of elements.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len() >> 1
    }

    /// Returns `true` if the segment tree is empty.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
