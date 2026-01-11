use llm_rs;

#[test]
fn test_integration() {
    let result = llm_rs::hello();
    assert_eq!(result, "Hello from llm-rs!");
}
