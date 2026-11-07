use acpi::{
    Handler,
    aml::{Interpreter, namespace::AmlName, object::Object},
};
use aml_test_tools::{
    RunTestResult,
    TestResult,
    handlers::logging_handler::LoggingHandler,
    new_interpreter_with_native_regions,
    run_test_for_opcodes,
    run_test_for_string,
};
use pci_types::PciAddress;
use std::str::FromStr;

// `run_aml_test` and `run_opcodes_test` are very similar in structure, but whilst there are only
// two of them it's not worth adding complexity to make them DRY.

/// Run a test against an ASL string and check for successful execution.
///
/// The string `asl` represents a compile-able ASL string, so needs to include the `DefinitionBlock`
/// statement.
#[allow(dead_code)]
pub fn run_aml_test<H: Handler + Send + Sync + 'static>(
    asl: &'static str,
    handler: H,
) -> Interpreter<LoggingHandler<H>> {
    let result = run_aml_test_with_result(asl, handler);
    match result {
        RunTestResult::Pass(interpreter) => interpreter,
        result => panic!("Test failed with: {:?}", TestResult::from(&result)),
    }
}

/// Run a test against an ASL string and return the raw result.
///
/// The string `asl` represents a compile-able ASL string, so needs to include the `DefinitionBlock`
/// statement.
#[allow(dead_code)]
pub fn run_aml_test_with_result<H: Handler + Send + Sync + 'static>(
    asl: &'static str,
    handler: H,
) -> RunTestResult<LoggingHandler<H>> {
    run_aml_test_with_pci_devices(asl, handler, Vec::new())
}

#[allow(dead_code)]
pub fn run_aml_test_with_pci_device<H: Handler + Send + Sync + 'static>(
    asl: &'static str,
    handler: H,
    device_path: &str,
    address: PciAddress,
) -> Interpreter<LoggingHandler<H>> {
    let pci_devices = vec![(AmlName::from_str(device_path).unwrap(), address)];
    let result = run_aml_test_with_pci_devices(asl, handler, pci_devices);
    match result {
        RunTestResult::Pass(interpreter) => interpreter,
        result => panic!("Test failed with: {:?}", TestResult::from(&result)),
    }
}

fn run_aml_test_with_pci_devices<H: Handler + Send + Sync + 'static>(
    asl: &'static str,
    handler: H,
    pci_devices: Vec<(AmlName, PciAddress)>,
) -> RunTestResult<LoggingHandler<H>> {
    // Tests calling the test functions don't do much else, and we usually want logging, so initialize it here.
    let _ = pretty_env_logger::try_init();

    let logged_handler = LoggingHandler::new(handler);
    let interpreter = new_interpreter_with_native_regions(logged_handler, pci_devices);

    run_test_for_string(asl, interpreter, &None)
}

/// Evaluate an object without arguments and return its unwrapped value.
#[allow(dead_code)]
pub fn evaluate(interpreter: &Interpreter<impl Handler>, path: &str) -> Object {
    (*interpreter.evaluate(AmlName::from_str(path).unwrap(), vec![]).unwrap()).clone()
}

/// Run a test against a sequence of AML opcodes.
///
/// The provided opcodes are wrapped in a `MAIN` method and also have a table header prepended. The
/// `MAIN` function is executed as part of the test, so `opcodes` only needs to include the actual
/// opcodes to execute.
#[allow(dead_code)]
pub fn run_opcodes_test(opcodes: &[u8], handler: impl Handler + Send + Sync + 'static) {
    let result = run_opcodes_test_with_result(opcodes, handler);
    assert!(matches!(result, RunTestResult::Pass(_)), "Test failed with: {:?}", TestResult::from(&result));
}

/// Run a test against a sequence of AML opcodes and return the raw result.
///
/// The provided opcodes are wrapped in a `MAIN` method and also have a table header prepended. The
/// `MAIN` function is executed as part of the test, so `opcodes` only needs to include the actual
/// opcodes to execute.
#[allow(dead_code)]
pub fn run_opcodes_test_with_result<H: Handler + Send + Sync + 'static>(
    opcodes: &[u8],
    handler: H,
) -> RunTestResult<LoggingHandler<H>> {
    let _ = pretty_env_logger::try_init();

    let logged_handler = LoggingHandler::new(handler);
    let interpreter = new_interpreter_with_native_regions(logged_handler, Vec::new());

    run_test_for_opcodes(opcodes, interpreter, &None)
}
