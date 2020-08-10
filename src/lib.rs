extern crate num;
extern crate roots;

mod rustnomial;
pub use rustnomial::binomial::LinearBinomial;
pub use rustnomial::derivative::Derivable;
pub use rustnomial::err::TryAddError;
pub use rustnomial::find_roots::Roots;
pub use rustnomial::integral::{Integrable, Integral};
pub use rustnomial::monomial::Monomial;
pub use rustnomial::poly_math;
pub use rustnomial::polynomial::Polynomial;
pub use rustnomial::sparsepolynomial::SparsePolynomial;
pub use rustnomial::terms::{Degree, Term};
pub use rustnomial::traits::{Evaluable, FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
pub use rustnomial::trinomial::QuadraticTrinomial;
