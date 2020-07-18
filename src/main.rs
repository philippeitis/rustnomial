extern crate rustnomial;
extern crate core;

use rustnomial::{polynomial, Polynomial};

fn main() {
    let a = polynomial!(1, 2);
    let b = polynomial!(1, 2);
    let c = polynomial!(1, 4, 4);

    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
    println!("{}", a.clone() * b);
    println!("{}", a.clone() * c);

}