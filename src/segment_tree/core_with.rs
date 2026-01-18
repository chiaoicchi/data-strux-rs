/// A segment tree for efficient point updates and range queries with operator.
///
/// Given a monoid `(S, op, id)`, this data structure supports:
/// - Point update: `set(i, x)` sets `a[i] = x`
/// - Point operation: `operate(i, x)` sets `a[i] = op(a[i], x)`
/// - Range query: `range_fold(l..r)` returns `op(a[l], op(a[l+1], ..., a[r-1]))`
///
/// Both operations run in O(log n) time.
#[repr(C)]
pub struct SegmentTreeWith<S, Op>
where
    S: Clone,
    Op: Fn(&S, &S) -> S,
{
    /// Binary heap-like array storing the tree nodes.
    /// Index 1 is the root, index `size + i` is the leaf for element `i`.
    data: Box<[S]>,
    /// Number of elements in the original array.
    n: usize,
    // Size of the tree (smallest power of 2 >= n).
    size: usize,
    /// log2(size), used for iteration bounds.
    log: usize,
    /// Identity element of the monoid.
    id: S,
    /// Binary operation of the monoid.
    op: Op,
}

impl<S, Op> SegmentTreeWith<S, Op>
where
    S: Clone,
    Op: Fn(&S, &S) -> S,
{
    /// Creates a new segment tree with `n` elements, all initialized to `id`.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn new(n: usize, id: S, op: Op) -> Self {
        let size = n.next_power_of_two();
        Self {
            data: vec![id.clone(); 2 * size].into_boxed_slice(),
            n,
            size,
            log: size.trailing_zeros() as usize,
            id,
            op,
        }
    }

    /// Creates a new segment tree from a slice.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn from_slice(v: &[S], id: S, op: Op) -> Self {
        let n = v.len();
        let size = n.next_power_of_two();
        let mut data = vec![id.clone(); 2 * size];
        data[size..size + n].clone_from_slice(v);
        unsafe {
            let d = data.as_mut_ptr();
            for i in (1..size).rev() {
                *d.add(i) = op(&*d.add(2 * i), &*d.add(2 * i + 1));
            }
        }
        Self {
            data: data.into_boxed_slice(),
            n,
            size,
            log: size.trailing_zeros() as usize,
            id,
            op,
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
            i < self.n,
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.size;
        unsafe {
            let d = self.data.as_mut_ptr();
            *d.add(i) = x;
            for t in 1..=self.log {
                let p = i >> t;
                *d.add(p) = (self.op)(&*d.add(2 * p), &*d.add(2 * p + 1));
            }
        }
    }

    /// Applies `op(a[i], x)` to the element at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[inline]
    pub fn operate(&mut self, mut i: usize, x: S) {
        debug_assert!(
            i < self.n,
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        i += self.size;
        unsafe {
            let d = self.data.as_mut_ptr();
            *d.add(i) = (self.op)(&*d.add(i), &x);
            for t in 1..=self.log {
                let p = i >> t;
                *d.add(p) = (self.op)(&*d.add(2 * p), &*d.add(2 * p + 1));
            }
        }
    }

    /// Returns the value at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline]
    pub fn get(&self, i: usize) -> S {
        debug_assert!(
            i < self.n,
            "index out of bounds: i={}, len={}",
            i,
            self.len(),
        );
        unsafe { self.data.get_unchecked(self.size + i).clone() }
    }

    /// Returns `op(a[l], a[l+1], ..., a[r-1])` for the given range.
    ///
    /// Returns `id` if the range is empty.
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
        } + self.size;
        let mut r = match range.end_bound() {
            std::ops::Bound::Unbounded => self.n,
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
        } + self.size;
        debug_assert!(
            l <= r,
            "left bound must be less than or equal to right bound: l={}, r={}",
            l - self.size,
            r - self.size,
        );
        debug_assert!(
            r <= self.size + self.n,
            "index out of bounds: r={}, len={}",
            r - self.size,
            self.n,
        );

        let mut left = self.id.clone();
        let mut right = self.id.clone();
        unsafe {
            let d = self.data.as_ptr();
            while l < r {
                if l & 1 == 1 {
                    left = (self.op)(&left, &*d.add(l));
                    l += 1;
                }
                if r & 1 == 1 {
                    r -= 1;
                    right = (self.op)(&*d.add(r), &right);
                }
                l >>= 1;
                r >>= 1;
            }
        }
        (self.op)(&left, &right)
    }

    /// Returns `op(a[0], a[1], ..., a[n-1])`.
    ///
    /// # Time complexity
    ///
    /// O(1)
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

    /// Returns the number of elements.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the segment tree is empty.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}
