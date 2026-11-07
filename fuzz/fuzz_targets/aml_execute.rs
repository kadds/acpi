#![no_main]

use acpi::aml::{
    AmlError, Interpreter,
    namespace::AmlName,
    op_region::{NativeRegionAccessMode, NativeRegionHandler, NativeRegionToken, OpRegion, RegionSpace},
};
use aml_test_tools::handlers::null_handler::NullHandler;
use core::sync::atomic::{AtomicU64, Ordering};
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;
use std::str::FromStr;

mod support;

use support::FuzzExecutionMeter;

const MAX_AML_BODY_BYTES: usize = 64 * 1024;

const STATIC_REGION_AND_FIELD: &[u8] = &[
    // OperationRegion (REG0, SystemMemory, 0x40000, 0x10)
    0x5b, 0x80, b'R', b'E', b'G', b'0', 0x00, 0x0c, 0x00, 0x00, 0x04, 0x00, 0x0a, 0x10,
    // Field (REG0, ByteAcc, NoLock, Preserve) { DATA, 8 }
    0x5b, 0x81, 0x0b, b'R', b'E', b'G', b'0', 0x10, b'D', b'A', b'T', b'A', 0x08,
];

struct FuzzToken {
    region: OpRegion,
    generation: u64,
}

struct FuzzNativeRegionHandler {
    generation: AtomicU64,
    expire_on_authorize: bool,
}

impl FuzzNativeRegionHandler {
    fn validate_token(&self, token: &NativeRegionToken, offset: usize, length: usize) -> Result<(), AmlError> {
        let token = token
            .downcast_ref::<FuzzToken>()
            .ok_or_else(|| AmlError::HostError("FUZZ_INVALID_TOKEN".to_string()))?;
        if token.generation != self.generation.load(Ordering::Acquire) {
            return Err(AmlError::HostError("FUZZ_REVOKED_TOKEN".to_string()));
        }
        let offset = u64::try_from(offset).map_err(|_| AmlError::BudgetExceeded)?;
        let length = u64::try_from(length).map_err(|_| AmlError::BudgetExceeded)?;
        if length == 0
            || offset.checked_add(length).is_none_or(|end| end > token.region.length)
            || token.region.base.checked_add(offset).is_none()
        {
            return Err(AmlError::HostError("FUZZ_REGION_BOUNDS".to_string()));
        }
        Ok(())
    }
}

impl NativeRegionHandler for FuzzNativeRegionHandler {
    fn authorize(
        &self,
        region: &OpRegion,
        _mode: NativeRegionAccessMode,
    ) -> Result<NativeRegionToken, AmlError> {
        if !matches!(region.space, RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig) {
            return Err(AmlError::NoHandlerForRegionAccess(region.space));
        }
        let generation = self.generation.load(Ordering::Acquire);
        let token = NativeRegionToken::new(FuzzToken {
            region: region.clone(),
            generation,
        });
        if self.expire_on_authorize {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
        Ok(token)
    }

    fn read(&self, token: &NativeRegionToken, offset: usize, length: usize) -> Result<u64, AmlError> {
        self.validate_token(token, offset, length)?;
        Ok(0)
    }

    fn write(
        &self,
        token: &NativeRegionToken,
        offset: usize,
        length: usize,
        _value: u64,
    ) -> Result<(), AmlError> {
        self.validate_token(token, offset, length)
    }

    fn invalidate_all(&self) -> Result<(), AmlError> {
        self.generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

fn package_length_bytes(length: usize) -> Option<Vec<u8>> {
    let encoded_bytes = if length < 0x40 {
        1
    } else if length < 0x1000 {
        2
    } else if length < 0x10_0000 {
        3
    } else if length < 0x1000_0000 {
        4
    } else {
        return None;
    };
    let mut encoded = Vec::new();
    encoded.try_reserve_exact(encoded_bytes).ok()?;
    if encoded_bytes == 1 {
        encoded.push(length as u8);
    } else {
        encoded.push((((encoded_bytes - 1) as u8) << 6) | (length as u8 & 0x0f));
        for shift in (4..(encoded_bytes * 8 - 4)).step_by(8) {
            encoded.push((length >> shift) as u8);
        }
    }
    Some(encoded)
}

fn append_main_method(aml: &mut Vec<u8>, body: &[u8]) -> bool {
    let mut package_bytes = 1;
    let package_length = loop {
        let Some(length) = body.len().checked_add(5 + package_bytes) else {
            return false;
        };
        let Some(encoded) = package_length_bytes(length) else {
            return false;
        };
        if encoded.len() == package_bytes {
            break encoded;
        }
        package_bytes = encoded.len();
    };
    let Some(reserve) = package_length
        .len()
        .checked_add(1 + 4 + 1)
        .and_then(|header| header.checked_add(body.len()))
    else {
        return false;
    };
    if aml.try_reserve(reserve).is_err() {
        return false;
    }
    aml.extend_from_slice(&[0x14]);
    aml.extend_from_slice(&package_length);
    aml.extend_from_slice(b"MAIN");
    aml.push(0);
    aml.extend_from_slice(body);
    true
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_AML_BODY_BYTES {
        return;
    }

    let mut body = Vec::new();
    if body.try_reserve(data.len() + 5).is_err() {
        return;
    }
    // Force one token-backed field read before executing the mutated AML body.
    body.extend_from_slice(&[0x70, b'D', b'A', b'T', b'A', 0x60]);
    body.extend_from_slice(data);
    body.extend_from_slice(&[0xa4, 0x00]);

    let mut aml = Vec::new();
    if aml.try_reserve(STATIC_REGION_AND_FIELD.len()).is_err() {
        return;
    }
    aml.extend_from_slice(STATIC_REGION_AND_FIELD);
    if !append_main_method(&mut aml, &body) {
        return;
    }

    let interpreter = Interpreter::new(NullHandler, 2);
    let meter = Arc::new(FuzzExecutionMeter::new());
    interpreter.install_execution_meter(meter);
    interpreter.install_native_region_handler(Box::new(FuzzNativeRegionHandler {
        generation: AtomicU64::new(1),
        expire_on_authorize: data.first().is_some_and(|byte| byte & 1 != 0),
    }));
    if interpreter.load_table(&aml).is_err() || interpreter.initialize_native_regions().is_err() {
        return;
    }

    let _ = interpreter.evaluate(AmlName::from_str("\\MAIN").unwrap(), Vec::new());
    let _ = interpreter.deinitialize_native_regions();
});
