pub mod inc;
pub mod encode;
pub mod fdr;
pub mod mwu;
pub mod rank_test;
pub mod result;
pub mod utils;

pub use inc::Inc;
pub use encode::EncodeIndex;
pub use fdr::{FdrResult, Fdr};
pub use mwu::mann_whitney_u;
pub use result::IncResult;
