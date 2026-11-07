use acpi::aml::{AmlError, Interpreter};
use aml_test_tools::handlers::null_handler::NullHandler;

#[test]
fn minimized_store_reference_cycle_returns_error_instead_of_looping() {
    let interpreter = Interpreter::new(NullHandler, 2);
    let aml = [
        // Name (TST0, One)
        0x08, b'T', b'S', b'T', b'0', 0x01,
        // Minimized libFuzzer input for a cyclic sequence of Store and RefOf operations.
        0x70, 0x70, 0x71, 0x63, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x63, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x7d, 0x7d, 0x70, 0x70, 0x7d, 0x7d, 0x70, 0x70,
    ];

    assert!(matches!(interpreter.load_table(&aml), Err(AmlError::ReferenceCycle | AmlError::BudgetExceeded)));
}
