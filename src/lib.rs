extern crate num;

mod rustnomial;
pub use rustnomial::degree::{Degree, Term};
pub use rustnomial::derivative::Derivable;
pub use rustnomial::integral::{Integrable, Integral};
pub use rustnomial::monomial::Monomial;
pub use rustnomial::polynomial::Polynomial;
pub use rustnomial::sparsepolynomial::SparsePolynomial;
pub use rustnomial::traits::{Evaluable, GenericPolynomial};
