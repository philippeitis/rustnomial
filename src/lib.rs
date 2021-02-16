#![no_std]
#![deny(clippy::missing_errors_doc)]
#![deny(clippy::doc_markdown)]
#![deny(clippy::redundant_clone)]
#![deny(clippy::needless_pass_by_value)]
#![cfg_attr(feature = "array_polynomials", feature(min_const_generics))]

#[macro_use]
extern crate alloc;

#[macro_use]
mod macros;
#[macro_use]
mod strings;

pub mod err;
mod numerics;
mod polynomial;
pub mod terms;

pub use crate::err::{PolynomialFromStringError, TryAddError};
pub use crate::polynomial::binomial::LinearBinomial;
pub use crate::polynomial::derivative::Derivable;
pub use crate::polynomial::find_roots::Roots;
pub use crate::polynomial::integral::{Integrable, Integral};
pub use crate::polynomial::monomial::Monomial;
pub use crate::polynomial::poly_math;
pub use crate::polynomial::polynomial::Polynomial;
pub use crate::polynomial::sparsepolynomial::SparsePolynomial;
pub use crate::polynomial::traits::{
    Evaluable, FreeSizePolynomial, MutablePolynomial, SizedPolynomial,
};
pub use crate::polynomial::trinomial::QuadraticTrinomial;
pub use crate::terms::{Degree, Term};
