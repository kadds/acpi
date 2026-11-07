#![no_main]

use aml_test_tools::handlers::null_handler::NullHandler;
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;

mod support;

use support::FuzzExecutionMeter;

const MAX_AML_BYTES: usize = 1 << 20;
const VALID_NAMESPACE_PREFIX: &[u8] = &[0x08, b'T', b'S', b'T', b'0', 0x01];

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_AML_BYTES - VALID_NAMESPACE_PREFIX.len() {
        return;
    }

    let interpreter = aml_test_tools::new_interpreter(NullHandler);
    interpreter.install_execution_meter(Arc::new(FuzzExecutionMeter::new()));
    let mut aml = Vec::with_capacity(VALID_NAMESPACE_PREFIX.len() + data.len());
    aml.extend_from_slice(VALID_NAMESPACE_PREFIX);
    aml.extend_from_slice(data);
    let _ = interpreter.load_table(&aml);
});
