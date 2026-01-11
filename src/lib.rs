//! # llm-rs
//!
//! A small LLM interpreter written in Rust.
//!
//! This library provides functionality for interpreting and running large language models.

/// A placeholder function that demonstrates the library structure.
///
/// # Examples
///
/// ```
/// use llm_rs::hello;
///
/// let greeting = hello();
/// assert_eq!(greeting, "Hello from llm-rs!");
/// ```
pub fn hello() -> String {
    "Hello from llm-rs!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello() {
        assert_eq!(hello(), "Hello from llm-rs!");
    }
}
