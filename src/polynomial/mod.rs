pub mod binomial;
pub mod derivative;
pub(crate) mod find_roots;
pub mod integral;
pub mod monomial;
pub mod poly_math;
#[allow(clippy::module_inception)]
pub mod polynomial;
pub mod sparsepolynomial;
pub mod traits;
pub mod trinomial;

#[cfg(feature = "array_polynomials")]
pub mod array_polynomial;
