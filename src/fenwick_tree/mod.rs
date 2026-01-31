/// A monoid is an algebraic structure consisting of a set equipped with
/// an associative binary operation and an identity element.
///
/// # Laws
///
/// Implementations must satisfy the following laws:
///
/// - **Identity**: `op(id(), x) == x` and `op(x, id()) == x`
/// - **Associativity**: `op(op(x, y), z) == op(x, op(y, z))`
pub trait Monoid: Clone {
    /// Returns the identity element of the monoid.
    fn id() -> Self;

    /// Performs the binary operation of the monoid.
    fn op(&self, other: &Self) -> Self;
}

/// A trait for monoids where every element has an inverse.
///
/// # Laws
///
/// Implementations must satisfy the following law:
///
/// - **Inverse**: `op(x, inv(x)) == id()` and `op(inv(x), x) == id()`
pub trait HasInverse: Monoid {
    /// Returns the inverse of the element.
    fn inv(&self) -> Self;
}

/// A group is a monoid where every element has an inverse.
///
/// This trait is automatically implemented for any type that implements
/// both [`Monoid`] and [`HasInverse`].
///
/// # Laws
///
/// Implementations must satisfy all laws of [`Monoid`] and [`HasInverse`].
pub trait Group: Monoid + HasInverse {}
impl<T: Monoid + HasInverse> Group for T {}

/// A fenwick tree for efficient point operates and range queries.
///
/// Given a monoid `(S, op, id)`, this data structure supports:
/// - Point operation: `operate(i, x)` sets `a[i] = op(a[i], x)`
/// - Prefix query: `prefix_fold(r)` returns `op(a[0], ..., a[r - 1])`
///
/// Both operations run in O(log n) time.
///
/// If monoid has inverse function, this data structure additionally supports:
/// - Range query: `range_fold(l..r)` returns `op(a[l], ..., a[r - 1])`
pub struct FenwickTree<S: Monoid>(Vec<S>);

impl<S: Monoid> FenwickTree<S> {
    /// Creates a new fenwick tree with `n` elements, all initialized to `S::id()`.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn new(n: usize) -> Self {
        Self(vec![S::id(); n + 1])
    }

    /// Creates a new fenwick tree from a vec.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn from_vec(mut v: Vec<S>) -> Self {
        let n = v.len();
        v.reserve(1);
        unsafe {
            let ptr = v.as_mut_ptr();
            std::ptr::copy(ptr, ptr.add(1), n);
            ptr.write(S::id());
            v.set_len(n + 1);
            for i in 1..=n {
                let lsb = i & i.wrapping_neg();
                if i + lsb <= n {
                    *ptr.add(i + lsb) = S::op(&*ptr.add(i + lsb), &*ptr.add(i));
                }
            }
        }
        Self(v)
    }

    /// Creates a new fenwick tree from a slice.
    ///
    /// # Time complexity
    ///
    /// O(n)
    pub fn from_slice(v: &[S]) -> Self {
        let n = v.len();
        let mut data = Vec::with_capacity(n + 1);
        data.push(S::id());
        data.extend_from_slice(v);
        unsafe {
            let d = data.as_mut_ptr();
            for i in 1..=n {
                let lsb = i & i.wrapping_neg();
                if i + lsb <= n {
                    *d.add(i + lsb) = S::op(&*d.add(i + lsb), &*d.add(i));
                }
            }
        }
        Self(data)
    }

    /// Appends an element to the end.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    pub fn push(&mut self, mut x: S) {
        let lsb = self.0.len() & self.0.len().wrapping_neg();
        let mut t = 1;
        unsafe {
            let d = self.0.as_mut_ptr();
            while t < lsb {
                x = S::op(&x, &*d.add(self.0.len() - t));
                t <<= 1;
            }
            self.0.push(x);
        }
    }

    /// Removes the last elements.
    ///
    /// # Time complexity
    ///
    /// O(1)
    pub fn pop(&mut self) -> Option<S> {
        if self.len() == 0 { None } else { self.0.pop() }
    }

    /// Creates a new empty fenwick tree with the specified capacity.
    ///
    /// # Time complexity
    ///
    /// O(1)
    pub fn with_capacity(capacity: usize) -> Self {
        let mut v = Vec::with_capacity(capacity + 1);
        v.push(S::id());
        Self(v)
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// # Time complexity
    ///
    /// O(n) worst case
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
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
            self.len()
        );
        i += 1;
        unsafe {
            let d = self.0.as_mut_ptr();
            while i < self.0.len() {
                *d.add(i) = S::op(&*d.add(i), &x);
                i += i & i.wrapping_neg();
            }
        }
    }

    /// Returns `op(a[0], ..., a[r - 1])` for the given range.
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
    pub fn prefix_fold(&self, mut r: usize) -> S {
        debug_assert!(
            r <= self.len(),
            "index out of bounds: r={}, len={}",
            r,
            self.len()
        );
        unsafe {
            let mut res = self.0.get_unchecked(r).clone();
            let d = self.0.as_ptr();
            while r > 0 {
                r &= r - 1;
                res = S::op(&*d.add(r), &res);
            }
            res
        }
    }

    /// Returns `op(a[0], a[1], ..., a[n-1])`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    pub fn all_fold(&self) -> S {
        self.prefix_fold(self.len())
    }

    /// Returns the smallest `r` such that `pred(prefix_fold(r))` is true.
    ///
    /// If no such `r` exists, returns `len()`.
    ///
    /// Assumes that `pred` is monotonic: if `pred(prefix_fold(r))` is true,
    /// then `pred(prefix_fold(r'))` is true for all `r' > r`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    pub fn lower_bound<P>(&self, _p: P) -> usize
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
        self.0.len() - 1
    }

    /// Returns `true` if the fenwick tree is empty.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S: Group> FenwickTree<S> {
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
    pub fn set(&mut self, i: usize, x: S) {
        debug_assert!(
            i < self.len(),
            "index out of bounds: i={}, len={}",
            i,
            self.len()
        );
        let diff = S::op(&self.get(i).inv(), &x);
        self.operate(i, diff);
    }

    /// Returns the value at index `i`.
    ///
    /// # Time complexity
    ///
    /// O(log n)
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
            self.len()
        );
        S::op(&self.prefix_fold(i).inv(), &self.prefix_fold(i + 1))
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
        let l = match range.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Included(&x) => x,
            std::ops::Bound::Excluded(&x) => x + 1,
        };
        let r = match range.end_bound() {
            std::ops::Bound::Unbounded => self.len(),
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
        };
        debug_assert!(
            l <= r,
            "left bound must be less than or equal to right bound: l={}, r={}",
            l,
            r,
        );
        debug_assert!(
            r <= self.len(),
            "index out of bounds: r={}, len={}",
            r,
            self.len(),
        );
        S::op(&self.prefix_fold(l).inv(), &self.prefix_fold(r))
    }
}
