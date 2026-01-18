/// A disjoint set union (DSU) data structure.
///
/// Uses path halving and union by size, achieving amortized O(α(n)) time per operation,
/// where α is the inverse Ackermann function.
#[derive(Clone, Debug)]
pub struct Dsu {
    /// If negative, this node is a root and the absolute value is the size of the set.
    /// If non-negative, this is the index of the parent node.
    parent: Box<[i32]>,
    num_components: usize,
}

impl Dsu {
    /// Creates a new DSU with `n` elements, where each element is initially in its own set.
    ///
    /// # Time complexity
    ///
    /// O(n)
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `n >= 2^31`
    pub fn new(n: usize) -> Self {
        debug_assert!(n < (1 << 31), "`n` must be less than 2^31");
        Self {
            parent: vec![-1; n].into_boxed_slice(),
            num_components: n,
        }
    }

    /// Returns the representative (root) of the set containing `x`.
    ///
    /// Applies path compression using path halving.
    ///
    /// # Time complexity
    ///
    /// Amortized O(α(n))
    #[inline(always)]
    pub fn root(&mut self, mut x: usize) -> usize {
        debug_assert!(
            x < self.len(),
            "index out of bounds: x={}, len={}",
            x,
            self.len()
        );
        unsafe {
            let p = self.parent.as_mut_ptr();
            while *p.add(x) >= 0 {
                let px = *p.add(x) as usize;
                if *p.add(px) >= 0 {
                    *p.add(x) = *p.add(px);
                }
                x = px;
            }
        }
        x
    }

    /// Returns `true` if `x` is the representative of its set.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline]
    pub fn is_root(&self, x: usize) -> bool {
        debug_assert!(
            x < self.len(),
            "index out of bounds: x={}, len={}",
            x,
            self.len()
        );
        unsafe { *self.parent.get_unchecked(x) < 0 }
    }

    /// Unites the sets containing `x` and `y`.
    ///
    /// Returns `true` if `x` and `y` were in different sets, `false` otherwise.
    /// Uses union by size: the smaller set is merged into the larger one.
    ///
    /// # Time complexity
    ///
    /// Amortized O(α(n))
    #[inline]
    pub fn unite(&mut self, x: usize, y: usize) -> bool {
        debug_assert!(
            x < self.len(),
            "index out of bounds: x={}, len={}",
            x,
            self.len()
        );
        debug_assert!(
            y < self.len(),
            "index out of bounds: y={}, len={}",
            y,
            self.len()
        );
        let (mut rx, mut ry) = (self.root(x), self.root(y));
        if rx == ry {
            return false;
        }
        unsafe {
            let p = self.parent.as_mut_ptr();
            if *p.add(rx) > *p.add(ry) {
                std::mem::swap(&mut rx, &mut ry);
            }
            *p.add(rx) += *p.add(ry);
            *p.add(ry) = rx as i32;
        }
        self.num_components -= 1;
        true
    }

    /// Returns `true` if `x` and `y` belong to the same set.
    ///
    /// # Time complexity
    ///
    /// Amortized O(α(n))
    #[inline]
    pub fn same(&mut self, x: usize, y: usize) -> bool {
        debug_assert!(
            x < self.len(),
            "index out of bounds: x={}, len={}",
            x,
            self.len()
        );
        debug_assert!(
            y < self.len(),
            "index out of bounds: y={}, len={}",
            y,
            self.len()
        );
        self.root(x) == self.root(y)
    }

    /// Returns the size of the set containing `x`.
    ///
    /// # Time complexity
    ///
    /// Amortized O(α(n))
    #[inline]
    pub fn size(&mut self, x: usize) -> usize {
        debug_assert!(
            x < self.len(),
            "index out of bounds: x={}, len={}",
            x,
            self.len()
        );
        let root = self.root(x);
        unsafe { (-self.parent.get_unchecked(root)) as usize }
    }

    /// Returns all sets as a vector of vectors.
    ///
    /// Each inner vector contains the elements of one set in ascending order.
    ///
    /// # Time complexity
    ///
    /// O(n α(n))
    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut groups = vec![vec![]; self.len()];
        for i in 0..self.len() {
            groups[self.root(i)].push(i);
        }
        groups.into_iter().filter(|g| !g.is_empty()).collect()
    }

    /// Returns the number of disjoint sets.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline]
    pub fn num_components(&self) -> usize {
        self.num_components
    }

    /// Returns the total number of elements.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Returns `true` if the DSU contains no elements.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}
