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

/// An action of a monoid `F` on a monoid `S`.
///
/// This represents a homomorphism from `F` to the endomorphism monoid of `S`.
///
/// # Laws
///
/// Implementations must satisfy the following laws:
///
/// - **Identity action**: `F::id().act(s) == s`
/// - **Compatibility**: `f.op(g).act(s) == f.act(g.act(s))`
pub trait Action<S: Monoid>: Monoid {
    /// Applies the action to an element of `S`.
    fn act(&self, s: &S) -> S;
}
