extern crate rustnomial;
extern crate core;

use rustnomial::Polynomial;
use std::any::Any;
use std::io::Result;

fn main() {
    let a = Polynomial::new(vec![1, 6, 12, 8]);
    let b = Polynomial::new(vec![1, 1, 1, 1, 1]);
    let res = a.div_mod(&b);
    println!("{}", a);
    println!("{}", b);
    match res {
        Ok((c, d)) => {
            println!("Div successful.");
            println!("{}", c);
            println!("{}", d);
        }
        _ => {
            println!("Div unsuccessful.");
        }
    }


}