extern crate rustnomial;
extern crate core;

use rustnomial::{polynomial, integral, Integrable, Polynomial, derivative};

fn main() {
    let a = Polynomial::new(vec![1, -2, -24]);
    let b = Polynomial::new(vec![1, -6]);
    let g: Polynomial<i8> = Polynomial::new(vec![]);
    let h = Polynomial{ terms: vec![0, 0, 0, 0]};
    println!("{}", g);
    println!("{}", h);

    let res = a.div_mod(&b);
    println!("{}", a);
    println!("{}", b);
    match res {
        Ok((c, d)) => {
            println!("Div successful.");
            println!("{}, {}", c, c.is_zero());
            println!("{}, {}", d, d.is_zero());
        }
        _ => {
            println!("Div unsuccessful.");
        }
    }

    let mut c = polynomial!(-1.2, -2.5, 3.0);
    println!("{}", c);
    println!("{}", derivative!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
    println!("{}", integral!(10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
    println!("{}", c.eval(1.0));
    c -= c.clone();
    println!("{}", c);
    let f = Polynomial{terms: vec!{1usize, 1, 25}};
    println!("{}, {}", f, f.eval(5));
    let a = -2.0f32;
    println!("{}", a.abs());
}