extern crate num;

mod rustnomial;
pub use rustnomial::polynomial::Polynomial;
pub use rustnomial::sparsepolynomial::SparsePolynomial;
pub use rustnomial::monomial::Monomial;
pub use rustnomial::integral::{Integral, Integrable};
pub use rustnomial::traits::{GenericPolynomial, Evaluable};
pub use rustnomial::degree::{Degree, Term};
pub use rustnomial::derivative::Derivable;