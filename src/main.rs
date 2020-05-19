extern crate rustnomial;

use rustnomial::rustnomial::polynomial::Polynomial;
use std::any::Any;

fn main() {
    let a = Polynomial::new(vec![1, 2, 3, 4, 5]);
    println!("{}", a);
    println!("{:?}", 1.type_id());
    let mut ap = Polynomial::new(vec![]);
    ap.terms = vec![0; 10];
    ap.trim();
    println!("{}", ap.terms.len());
}