#![no_main]

use acpi::aml::{object::Object, resource::resource_descriptor_list};
use libfuzzer_sys::fuzz_target;

const MAX_RESOURCE_TEMPLATE_BYTES: usize = 64 * 1024;
const VALID_IO_RESOURCE: &[u8] = &[0x47, 0x01, 0x60, 0x00, 0x60, 0x00, 0x01, 0x01];

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_RESOURCE_TEMPLATE_BYTES - VALID_IO_RESOURCE.len() - 2 {
        return;
    }

    let mut template = Vec::with_capacity(VALID_IO_RESOURCE.len() + data.len() + 2);
    template.extend_from_slice(VALID_IO_RESOURCE);
    template.extend_from_slice(data);
    template.extend_from_slice(&[0x79, 0]);
    let _ = resource_descriptor_list(Object::Buffer(template).wrap());
});
