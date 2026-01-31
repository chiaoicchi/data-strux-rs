use crate::segment_tree::{Action, Monoid};

/// A lazy segment tree for efficient range updates and range queries.
///
/// Given a monoid `(S, op, id)` and an action monoid `(F, compose, id)`,
/// this data structure supports:
/// - Point update: `set(i, x)` sets `a[i] = x`
/// - Point action: `apply(i, f)` sets `a[i] = f.act(a[i])`
/// - Range action: `range_apply(l..r, f)` applies `f` to all elements in range
/// - Range query: `range_fold(l..r)` returns `op(a[l], ..., a[r-1])`
///
/// All operations run in O(log n) time.
#[repr(C)]
pub struct LazySegmentTree<S: Monoid, F: Action<S>> {
    /// Binary heap-like array storing the tree nodes.
    /// Index 1 is the root, index `size + i` is the leaf for element `i`.
    data: Box<[S]>,
    /// Lazy heap-like array storing the tree nodes.
    lazy: Box<[F]>,
    /// Number of elements in the original array.
    n: usize,
    /// log2(size), used for iteration bounds.
    log: usize,
}

impl<S: Monoid, F: Action<S>> LazySegmentTree<S, F> {
    /// Creates a new lazy segment tree with `n` elements, all initialized to `S::id()`.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn new(n: usize) -> Self {
        let size = n.next_power_of_two();
        Self {
            data: vec![S::id(); size << 1].into_boxed_slice(),
            lazy: vec![F::id(); size].into_boxed_slice(),
            n,
            log: size.trailing_zeros() as usize,
        }
    }

    /// Creates a new lazy segment tree from a vec.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn from_vec(mut v: Vec<S>) -> Self {
        let n = v.len();
        let size = n.next_power_of_two();
        v.resize(size << 1, S::id());
        unsafe {
            let v = v.as_mut_ptr();
            std::ptr::copy(v, v.add(size), n);
            for i in (1..size).rev() {
                v.add(i)
                    .write(S::op(&*v.add(i << 1), &*v.add((i << 1) + 1)));
            }
            v.write(S::id());
        }
        Self {
            data: v.into_boxed_slice(),
            lazy: vec![F::id(); size].into_boxed_slice(),
            n,
            log: size.trailing_zeros() as usize,
        }
    }

    /// Creates a new lazy segment tree from a slice.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn from_slice(v: &[S]) -> Self {
        let n = v.len();
        let size = n.next_power_of_two();
        let mut data = vec![S::id(); size << 1];
        unsafe {
            let d = data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(v.as_ptr(), d.add(size), n);
            for i in (1..size).rev() {
                *d.add(i) = S::op(&*d.add(2 * i), &*d.add(2 * i + 1));
            }
        }
        Self {
            data: data.into_boxed_slice(),
            lazy: vec![F::id(); size].into_boxed_slice(),
            n,
            log: size.trailing_zeros() as usize,
        }
    }

    /// Sets the value at index `i` to `x`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[inline]
    pub fn set(&mut self, mut i: usize, x: S) {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.size();
        for t in (1..=self.log).rev() {
            self.push(i >> t);
        }
        unsafe {
            *self.data.get_unchecked_mut(i) = x;
        }
        while i > 1 {
            i >>= 1;
            self.update(i);
        }
    }

    /// Applies `S::op(a[i], x)` to the element at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[inline]
    pub fn operate(&mut self, mut i: usize, x: S) {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.size();
        for t in (1..=self.log).rev() {
            self.push(i >> t);
        }
        unsafe {
            *self.data.get_unchecked_mut(i) = S::op(&self.data.get_unchecked(i), &x);
        }
        while i > 1 {
            i >>= 1;
            self.update(i);
        }
    }

    /// Applies action `f` to the element at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[inline]
    pub fn apply(&mut self, mut i: usize, f: F) {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.size();
        for t in (1..=self.log).rev() {
            self.push(i >> t);
        }
        unsafe {
            *self.data.get_unchecked_mut(i) = f.act(self.data.get_unchecked(i));
        }
        while i > 1 {
            i >>= 1;
            self.update(i);
        }
    }

    /// Applies action `f` to all elements in the given range.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[inline]
    pub fn range_apply(&mut self, range: impl std::ops::RangeBounds<usize>, f: F) {
        let mut l = match range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Included(&x) => x,
            std::ops::Bound::Excluded(&x) => x + 1,
        } + self.size();
        let mut r = match range.end_bound() {
            std::ops::Bound::Unbounded => self.len(),
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
        } + self.size();
        debug_assert!(
            l <= r,
            "left bound must be less than or equal to right bound: l={}, r={}",
            l - self.size(),
            r - self.size(),
        );
        debug_assert!(
            r <= self.len() + self.size(),
            "index out of bounds: r={}, len={}",
            r - self.size(),
            self.len(),
        );
        if l == r {
            return;
        }
        l >>= l.trailing_zeros();
        r >>= r.trailing_zeros();

        for t in (1..usize::BITS - l.leading_zeros()).rev() {
            self.push(l >> t);
        }
        for t in (1..usize::BITS - r.leading_zeros()).rev() {
            self.push((r - 1) >> t);
        }

        {
            let (mut l, mut r) = (l, r);
            unsafe {
                let data = self.data.as_mut_ptr();
                let lazy = self.lazy.as_mut_ptr();
                loop {
                    if l >= r {
                        *data.add(l) = f.act(&*data.add(l));
                        if l < self.size() {
                            *lazy.add(l) = F::op(&f, &*lazy.add(l));
                        }
                        l += 1;
                        l >>= l.trailing_zeros();
                    } else {
                        r -= 1;
                        *data.add(r) = f.act(&*data.add(r));
                        if r < self.size() {
                            *lazy.add(r) = F::op(&f, &*lazy.add(r));
                        }
                        r >>= r.trailing_zeros();
                    }
                    if l == r {
                        break;
                    }
                }
            }
        }

        while l > 1 {
            l >>= 1;
            self.update(l);
        }
        r -= 1;
        while r > 1 {
            r >>= 1;
            self.update(r);
        }
    }

    /// Returns the value at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[inline]
    pub fn get(&mut self, mut i: usize) -> S {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.size();
        for t in (1..=self.log).rev() {
            if (i >> t) << t != i {
                self.push(i >> t);
            }
        }
        unsafe { self.data.get_unchecked(i).clone() }
    }

    /// Returns `op(a[l], a[l+1], ..., a[r-1])` for the given range.
    ///
    /// Returns `S::id()` if the range is empty.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[inline]
    pub fn range_fold(&self, range: impl std::ops::RangeBounds<usize>) -> S {
        let mut l = match range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Included(&x) => x,
            std::ops::Bound::Excluded(&x) => x + 1,
        } + self.size();
        let mut r = match range.end_bound() {
            std::ops::Bound::Unbounded => self.len(),
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
        } + self.size();
        debug_assert!(
            l <= r,
            "left bound must be less than or equal to right bound: l={}, r={}",
            l - self.size(),
            r - self.size(),
        );
        debug_assert!(
            r <= self.len() + self.size(),
            "index out of bounds: r={}, len={}",
            r - self.size(),
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
            let data = self.data.as_ptr();
            let lazy = self.lazy.as_ptr();
            loop {
                if l >= r {
                    let mut i = l >> 1;
                    left = S::op(&left, &*data.add(l));
                    l += 1;
                    l >>= l.trailing_zeros();
                    while i > l >> 1 {
                        left = (*lazy.add(i)).act(&left);
                        i >>= 1;
                    }
                } else {
                    let mut i = r >> 1;
                    r -= 1;
                    right = S::op(&*data.add(r), &right);
                    r >>= r.trailing_zeros();
                    while i > r >> 1 {
                        right = (*lazy.add(i)).act(&right);
                        i >>= 1;
                    }
                }
                if l == r {
                    break;
                }
            }
        }
        let mut res = S::op(&left, &right);
        let mut i = l >> 1;
        unsafe {
            let lazy = self.lazy.as_ptr();
            while i > 0 {
                res = (*lazy.add(i)).act(&res);
                i >>= 1;
            }
        }
        res
    }

    /// Returns `op(a[0], a[1], ..., a[n-1])`.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline]
    pub fn all_fold(&self) -> S {
        unsafe { self.data.get_unchecked(1).clone() }
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

    #[inline(always)]
    fn push(&mut self, i: usize) {
        let data = self.data.as_mut_ptr();
        let lazy = self.lazy.as_mut_ptr();
        unsafe {
            let f = std::ptr::replace(lazy.add(i), F::id());
            *data.add(i << 1) = f.act(&*data.add(i << 1));
            *data.add((i << 1) + 1) = f.act(&*data.add((i << 1) + 1));
            if i << 1 < self.size() {
                *lazy.add(i << 1) = F::op(&f, &*lazy.add(i << 1));
                *lazy.add((i << 1) + 1) = F::op(&f, &*lazy.add((i << 1) + 1));
            }
        }
    }

    #[inline(always)]
    fn update(&mut self, i: usize) {
        let data = self.data.as_mut_ptr();
        unsafe {
            *data.add(i) = S::op(&*data.add(i << 1), &*data.add((i << 1) + 1));
        }
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.lazy.len()
    }

    /// Returns the number of elements.
    ///
    /// # Time complexity
    ///
    /// O(1)    
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the lazy segment tree is empty.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
