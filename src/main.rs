extern crate rustnomial;

// fn k_vec_mul<N>(_lhs: &Vec<N>, _rhs: &Vec<N>) -> Vec<N>
// where
//     N: Mul<Output = N> + AddAssign + Copy + Zero,
// {
//     let _rhs_len = _rhs.len();
//     let _lhs_len = _lhs.len();
//
//     if (_rhs_len < 4) || (_lhs_len < 4) {
//         return vec_mul(_rhs, _lhs);
//     }
//
//     let n = _rhs_len.max(_lhs_len) / 2;
//
//     let (x1, x2) = _lhs.split_at(_lhs_len - n);
//     let (y1, y2) = _rhs.split_at(_rhs_len - n);
//
//     let mut a = k_vec_mul(&x1.to_vec(), &y1.to_vec());
//     let mut d = k_vec_mul(&x2.to_vec(), &y2.to_vec());
//     let mut e = k_vec_mul(&a, &d);
//     a.append(&mut e);
//     a.append(&mut d);
//     a
// }

// fn vec_mul<N>(_lhs: &Vec<N>, _rhs: &Vec<N>) -> Vec<N>
// where
//     N: Mul<Output = N> + AddAssign + Copy + Zero,
// {
//     let _rhs = &_rhs[first_nonzero_index(&_rhs)..];
//     let _lhs = &_lhs[first_nonzero_index(&_lhs)..];
//     let mut terms = vec![N::zero(); _rhs.len() + _lhs.len() - 1];
//     for (index, &rterm) in _rhs.iter().enumerate() {
//         if rterm.is_zero() {
//             continue;
//         }
//         for (&lterm, term) in _lhs.iter().zip(terms[index..].iter_mut()) {
//             *term += rterm * lterm;
//         }
//     }
//     terms
// }
//
// fn main() {
//     // let x = vec![1, 0, 0, 0, 0, 0, 0, 0];
//     // let y = vec![2, 0, 0, 0, 0, 0, 0, 0];
//     //
//     println!("{:?}", 1);
//     // println!("{:?}", k_vec_mul(&x, &y));
//
//     // let sqrt = 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128.abs();
//     // let max = sqrt * sqrt;
//     // println!("{:#032X}", sqrt);
//     // println!("{:#032X}", max);
// }

//
// use std::{time::Duration as D, thread::sleep, print as P};
// fn main() {
//     P!("\x1b[2J");
//     for a in 0.. {
//         let (mut r, a, z) = ([(0., 0); 1760], a as f32 * 0.07, f32::sin_cos);
//         for j in 0..90 {
//             for i in 0..315 {
//                 let (f, d) = z(j as f32 * 0.07);
//                 let ((c, l), h, (e, g), (n, m)) =
//                     (z(i as f32 / 50.), d + 2., z (a), z(a/2.));
//                 let (t, p) = (c * h * g - f * e, 1. / (c * h * e + f * g + 5.));
//                 let (x, y) = ((40. + 30. * p * (l * h * m - t * n)) as usize,
//                 (12. + 15. * p * (l * h * n + t * m)) as usize);
//                 let o = x + 80 * y;
//                 if 22 > y && 80 > x && p > r[o].0 {
//                     r[o] = (
//                         p,
//                         b".,-~:;=!*#$@"[(8.
//                             * ((f * e - c * d * g) * m - c * d * e - f * g - l * d * n))
//                             as usize],
//                     );
//                 }
//             }
//         }
//         P!("\x1b
//         [H");
//         r.chunks(80).for_each(|s| {
//             s.iter().for_each(|p| P!("{}", p.1 as char));
//             P!("\n")
//         });
//         sleep(D::from_millis(30));
//     }
// }

// //              use std::{time              // 14/14
//          ::Duration as D,thread          // 22/22
//         ::sleep as S,print as P};        // 25/25
//      fn main(){let z=f32::sin_cos;P!     // 31/30 (+1)
//   ("\x1b[2J");for a in 0..{let(mut r,a   // 36/36
//   )=([(0.,0);1760],a as f32*0.07);for    // 35/36 (-1)
//  j in 0..90{for i in 0..315{let (i,j)=   // 38/38
//  (i as f32/50.,j as f32*0.07);let((c,l   // 38/38
//  ),(f,d),(e,g),(n      ,m))=(z(i),z(j),  // 39/39
// z(a),z(a/2.));let       h=d+2.;let t=c*h // 40/40 (+1)
// *g-f*e;let p=1.           /(c*h*e+f*g+5. // 40/40
// );let x=(40.+30.       *p*(l*h*m-t*n))as // 40/40
//  usize;let y=(12.     +15.*p*(l*h*n+t*m  // 38/38
//  ))as usize;let o=x+80*y;if 22>y&&80>x   // 37/38 (-1)
//  &&p>r[o].0{r[o]=(p,b".,-~:;=!*#$@"[(8.  // 38/38
//   *((f*e-c*d*g)*m-c*d*e-f*g-l*d*n))as    // 36/36
//     usize]);}}}P!("\x1b[H");r.chunks(    // 33/33
//      80).for_each(|s|{s.iter().//--      // 26/30 (-4)
//        for_each(|p|P!("{}",p.1 as        // 26/26 (+3)
//        char));println!()}); S(D::        // 26/22 (+4)
//            from_millis(30));}}           // 17/14

//              use std::{time              // 14/14
//          ::Duration as D,thread          // 22/22
//         ::sleep as S,print as P};        // 25/25
//      fn main(){P!("\x1b[2J");for a       // 29/30 (-1)
//   in 0..{let(mut r,a,z)=([(0.,0);1760]   // 36/36
//   ,a as f32*0.07,f32::sin_cos);for j    // 35/36 (-1)
//  in 0..90{for i in 0..315{let(i,j)=(i//  // 38/38 (-2)
//  as f32/50.,0.07 * j as f32);let((c,l),(   // 38/38
//  f,d),(e,g),(n,m)      )=(z(i),z(j),z(a) // 39/39
// ,z(a/2.));let h=        d+2.;let(t,p)=(c // 40/40
// *h*g-f*e,1./(c            *h*e+f*g+5.)); // 40/40
// let(x,y)=((40.+        30.*p*(l*h*m-t*n) // 40/40
//  )as usize,(12.+      15.*p*(l*h*n+t*m)  // 38/38
//  )as usize);let o=x+80*y;if 22>y&&80>x   // 37/38 (-1)
//  &&p>r[o].0{r[o]=(p,b".,-~:;=!*#$@"[(8.  // 38/38
//   *((f*e-c*d*g)*m-c*d*e-f*g-l*d*n))as    // 36/36
//     usize]);}}}P!("\x1b[H");r.chunks(    // 33/33
//      80).for_each(|s|{s.iter().//--      // 26/30 (-4)
//        for_each(|p|P!("{}",p.1 as        // 26/26
//         char));P!("\n")});S(D::          // 23/22 (+1)
//           from_millis(30));}}            // 19/14 (+5)

//              use std::{time              // 14/14
//          ::Duration as D,thread          // 22/22
//         ::sleep,print as P};//---        // 25/25 (-5)
//      fn main(){P!("\x1b[2J");for a       // 29/30 (-1)
//   in 0..{let(mut r,a,z)=([(0.,0);1760]   // 36/36
//   ,a as f32*0.07,f32::sin_cos);for j in  // 37/36 (+1)
//  0..90{for i in 0..315{let(j,i)=(0.07*j  // 38/38
//  as f32,i as f32/50.);let((c,l),(f,d),(  // 38/38
//  e,g),(n,m))=(z(       i),z(j),z(a),z(a/ // 39/39
// 2.));let h=d+2.         ;let(t,p)=(c*h*g // 40/40
// -f*e,1./(c*h*e            +f*g+5.));let( // 40/40
// x,y)=((40.+ 30.*        p*(l*h*m-t*n))as // 40/40
//  usize,(12.+15.*p      *(l*h*n+t*m))as   // 37/38 (-1)
//  usize);let o=x+80*y;if 22>y&&80>x&&p>r  // 38/38
//  [o].0{r[o]=(p,b".,-~:;=!*#$@"[(8.*((f*  // 38/38
//   e-c*d*g)*m-c*d*e-f*g-l*d*n))as usize   // 36/36
//     ]);}}}P!("\x1b[H");r.chunks(80).     // 32/33 (-1)
//      for_each(|s|{s.iter().for_each      // 30/30
//        (|p|P!("{}",p.1 as char));        // 26/26
//          println!()});sleep(D::          // 22/22
//            from_millis(30));}}           // 17/14 (+3)

//
//              use std::{time              // 14/14
//          ::Duration as D,thread          // 22/22
//         ::sleep,print as P};//---        // 25/25 (-5)
//      fn main(){P!("\x1b[2J");for a       // 29/30 (-1)
//   in 0..{let(mut r,a,z)=([(0.,0);1760]   // 36/36
//   ,a as f32*0.07,f32::sin_cos);for j in  // 37/36 (+1)
//  0..90{for i in 0..315{let(f,d)=z(0.07*  // 38/38
//  j as f32);let((c,l),h,(e,g),(n,m))=(z(  // 38/38
//  i as f32/50.),d+      2.,z(a),z(a/2.)); // 39/39
// let(t,p)=(c*h*g-                         // 40/40
// -f*e,1./(c*h*e            +f*g+5.));let( // 40/40
// x,y)=((40.+ 30.*        p*(l*h*m-t*n))as // 40/40
//  usize,(12.+15.*p      *(l*h*n+t*m))as   // 37/38 (-1)
//  usize);let o=x+80*y;if 22>y&&80>x&&p>r  // 38/38
//  [o].0{r[o]=(p,b".,-~:;=!*#$@"[(8.*((f*  // 38/38
//   e-c*d*g)*m-c*d*e-f*g-l*d*n))as usize   // 36/36
//     ]);}}}P!("\x1b[H");r.chunks(80).     // 32/33 (-1)
//      for_each(|s|{s.iter().for_each      // 30/30
//        (|p|P!("{}",p.1 as char));        // 26/26
//          println!()});sleep(D::          // 22/22
//            from_millis(30));}}           // 17/14 (+3)

//              use std::{time              // 14/14
//          ::Duration as D,thread          // 22/22
//        ::sleep_ms,print as P};//         // 25/25
//      fn main(){P!("\x1b[2J");for a       // 29/30
//   in 0..{let (mut r,a,z)=([(0.,0);1760   // 36/36
//   ],a as f32*0.07,f32::sin_cos);for j    // 35/36
//  in 0..90{for i in 0..315{let(f,d)=z(j   // 37/38
//  as f32*0.07);let((c,l),h,(e,g),(n,m))=  // 38/38
//  (z(i as f32/50.)      ,d+2.,z(a),z(0.5* // 39/39
// a));let(t,p)=(c*        h*g-f*e,1./(c*h* // 40/40
// e+f*g+5.));let            (x,y)=((40.+p* // 40/40
// 30.*(l*h*m-t*n))        as usize,(12.+p* // 40/40
//  15.*(l*h*n+t*m))      as usize);let o=x // 39/38
//  +80*y;if 22>y&&x>0&&y>0&&80>x&&p>r[o].  // 38/38
//  0{r[o]=(p,b".,-~:;=!*#$@"[(8.*((f*e-c*  // 38/38
//   d*g)*m-c*d*e-f*g-l*d*n))as usize]);}   // 36/36
//     }}P!("\x1b[H");r.chunks(80).//---    // 33/33
//      for_each(|s|{s.iter().for_each      // 30/30
//        (|p|P!("{}",p.1 as char));        // 26/26
//          P!("\n")});sleep_ms(30          // 22/22
//              );}}//------//              // 14/14

use rustnomial::integral;
fn main() {
    let a = integral!(1, 2, 3);
    println!("{}", a);
}
