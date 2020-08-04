extern crate core;
extern crate num;

mod rustnomial;
pub use rustnomial::terms::{Degree, Term};
pub use rustnomial::derivative::Derivable;
pub use rustnomial::err::TryAddError;
pub use rustnomial::integral::{Integrable, Integral};
pub use rustnomial::monomial::Monomial;
pub use rustnomial::binomial::LinearBinomial;
pub use rustnomial::polynomial::Polynomial;
pub use rustnomial::sparsepolynomial::SparsePolynomial;
pub use rustnomial::traits::{Evaluable, FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
pub use rustnomial::poly_math;
