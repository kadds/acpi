//! Verify that AML Global Lock operations use the host-owned lock protocol.

use acpi::{
    Handle,
    Handler,
    RawPhysicalMapping,
    aml::{
        AmlError,
        Interpreter,
        op_region::{NativeRegionAccessMode, NativeRegionHandler, NativeRegionToken, OpRegion},
    },
};
use aml_test_tools::new_interpreter_with_native_regions;
use lock_api::RawReentrantMutex;
use parking_lot::{RawMutex, RawThreadId};
use pci_types::PciAddress;
use std::{
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

#[test]
fn global_lock_operations_use_the_host_protocol_and_are_reentrant() {
    let interpreter = new_interpreter_with_native_regions(LockHandler::new(), Vec::new());
    interpreter.acquire_global_lock(0).expect("first global-lock acquire");
    interpreter.acquire_global_lock(0).expect("recursive global-lock acquire");
    assert!(interpreter.has_global_lock_dependency());
    interpreter.release_global_lock().expect("release recursive acquire");
    interpreter.release_global_lock().expect("release hardware acquire");
    assert!(matches!(
        interpreter.release_global_lock(),
        Err(AmlError::HostError(message)) if message == "NOT_ACQUIRED_GLOBAL_LOCK"
    ));
}

#[test]
fn aml_acquire_release_executes_through_the_global_lock_handler() {
    let interpreter = new_interpreter_with_native_regions(LockHandler::new(), Vec::new());
    let aml = [
        // Method (MAIN, 0) {
        //     Acquire (\\_GL_, 0); Release (\\_GL_); Return (0x2a)
        // }
        0x14, 0x19, b'M', b'A', b'I', b'N', 0x00, 0x5b, 0x23, 0x5c, b'_', b'G', b'L', b'_', 0x00, 0x00, 0x5b, 0x27,
        0x5c, b'_', b'G', b'L', b'_', 0xa4, 0x0a, 0x2a,
    ];
    interpreter.load_table(&aml).expect("Global Lock AML should parse");
    let result = interpreter
        .evaluate(acpi::aml::namespace::AmlName::from_str("\\MAIN").unwrap(), Vec::new())
        .expect("AML Acquire/Release should use the host Global Lock");
    assert_eq!(result.as_integer().unwrap(), 0x2a);
}

#[test]
fn aml_acquire_returns_false_on_success_and_true_on_timeout() {
    for (timeout, expected) in [(false, 0), (true, u64::MAX)] {
        let callbacks = Arc::new(GlobalLockCallbacks::new(timeout));
        let interpreter = interpreter_with_callbacks(Arc::clone(&callbacks));
        let aml = method_table(
            *b"MAIN",
            &[
                0xa4, // Return
                0x5b, 0x23, // Acquire
                0x5c, b'_', b'G', b'L', b'_', // \_GL_
                0x00, 0x00, // timeout = 0 ms
            ],
        );
        interpreter.load_table(&aml).expect("Acquire expression should parse");

        let result = interpreter
            .evaluate(acpi::aml::namespace::AmlName::from_str("\\MAIN").unwrap(), Vec::new())
            .expect("Acquire returns its timeout result to AML");
        assert_eq!(result.as_integer().expect("Acquire result is an integer Boolean"), expected);
        assert_eq!(callbacks.acquire_count.load(Ordering::Relaxed), 1);
        assert_eq!(
            callbacks.release_count.load(Ordering::Relaxed),
            usize::from(!timeout),
            "a successful lock held at method return must be released"
        );
    }
}

#[test]
fn uncertain_global_lock_acquire_poison_prevents_following_aml_work() {
    let callbacks = Arc::new(GlobalLockCallbacks::unknown());
    let interpreter = interpreter_with_callbacks(Arc::clone(&callbacks));
    let aml = method_table(
        *b"MAIN",
        &[
            0xa4, // Return
            0x5b, 0x23, // Acquire
            0x5c, b'_', b'G', b'L', b'_', // \_GL_
            0x00, 0x00,
        ],
    );
    interpreter.load_table(&aml).expect("Acquire expression should parse");
    let path = acpi::aml::namespace::AmlName::from_str("\\MAIN").unwrap();

    assert!(matches!(interpreter.evaluate(path.clone(), Vec::new()), Err(AmlError::GlobalLockOutcomeUnknown)));
    assert!(matches!(interpreter.evaluate(path, Vec::new()), Err(AmlError::GlobalLockOutcomeUnknown)));
    assert_eq!(callbacks.acquire_count.load(Ordering::Relaxed), 1);
    assert_eq!(callbacks.release_count.load(Ordering::Relaxed), 0);
}

#[test]
fn aml_method_error_releases_global_lock_acquisitions_from_that_method() {
    let callbacks = Arc::new(GlobalLockCallbacks::new(false));
    let interpreter = interpreter_with_callbacks(Arc::clone(&callbacks));
    let aml = method_table(
        *b"MAIN",
        &[
            0x5b, 0x23, // Acquire
            0x5c, b'_', b'G', b'L', b'_', // \_GL_
            0x00, 0x00, // timeout = 0 ms
            0x78, // Divide (1, 0, NullName, NullName)
            0x0a, 0x01, // ByteConst(1)
            0x00, // Zero
            0x00, // NullName target for remainder
            0x00, // NullName target for quotient
            0xa4, 0x00, // Return(Zero)
        ],
    );
    interpreter.load_table(&aml).expect("erroring method should parse");

    let result = interpreter.evaluate(acpi::aml::namespace::AmlName::from_str("\\MAIN").unwrap(), Vec::new());
    assert!(matches!(result, Err(AmlError::DivisionByZero)));
    assert_eq!(callbacks.acquire_count.load(Ordering::Relaxed), 1);
    assert_eq!(
        callbacks.release_count.load(Ordering::Relaxed),
        1,
        "method failure must unwind locks acquired by the failed method"
    );
}

fn interpreter_with_callbacks(callbacks: Arc<GlobalLockCallbacks>) -> Interpreter<LockHandler> {
    let interpreter = Interpreter::new(LockHandler::new(), 2);
    interpreter.install_native_region_handler(Box::new(RecordingNativeRegionHandler { callbacks }));
    interpreter
}

fn method_table(name: [u8; 4], body: &[u8]) -> Vec<u8> {
    let package_length = 1 + 4 + 1 + body.len();
    assert!(package_length < 0x40, "test AML uses one-byte PkgLength");
    let mut aml = Vec::with_capacity(package_length + 1);
    aml.extend_from_slice(&[0x14, package_length as u8]);
    aml.extend_from_slice(&name);
    aml.push(0);
    aml.extend_from_slice(body);
    aml
}

struct GlobalLockCallbacks {
    timeout: bool,
    unknown: bool,
    acquire_count: AtomicUsize,
    release_count: AtomicUsize,
}

impl GlobalLockCallbacks {
    fn new(timeout: bool) -> Self {
        Self { timeout, unknown: false, acquire_count: AtomicUsize::new(0), release_count: AtomicUsize::new(0) }
    }

    fn unknown() -> Self {
        Self {
            timeout: false,
            unknown: true,
            acquire_count: AtomicUsize::new(0),
            release_count: AtomicUsize::new(0),
        }
    }
}

struct RecordingNativeRegionHandler {
    callbacks: Arc<GlobalLockCallbacks>,
}

impl NativeRegionHandler for RecordingNativeRegionHandler {
    fn authorize(&self, region: &OpRegion, _mode: NativeRegionAccessMode) -> Result<NativeRegionToken, AmlError> {
        Ok(NativeRegionToken::new(region.clone()))
    }

    fn read(&self, _token: &NativeRegionToken, _offset: usize, _length: usize) -> Result<u64, AmlError> {
        Err(AmlError::HostError("UNUSED_TEST_REGION_READ".to_string()))
    }

    fn write(
        &self,
        _token: &NativeRegionToken,
        _offset: usize,
        _length: usize,
        _value: u64,
    ) -> Result<(), AmlError> {
        Err(AmlError::HostError("UNUSED_TEST_REGION_WRITE".to_string()))
    }

    fn acquire_global_lock(&self, _timeout_us: u64) -> Result<(), AmlError> {
        self.callbacks.acquire_count.fetch_add(1, Ordering::Relaxed);
        if self.callbacks.timeout {
            Err(AmlError::MutexAcquireTimeout)
        } else if self.callbacks.unknown {
            Err(AmlError::GlobalLockOutcomeUnknown)
        } else {
            Ok(())
        }
    }

    fn release_global_lock(&self) -> Result<(), AmlError> {
        self.callbacks.release_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn execution_thread_id(&self) -> Option<u64> {
        Some(1)
    }
}

#[test]
fn locked_field_reads_under_the_global_lock() {
    let interpreter = new_interpreter_with_native_regions(LockHandler::new(), Vec::new());
    let aml = [
        // OperationRegion (REG0, SystemMemory, 0, 1)
        0x5b, 0x80, b'R', b'E', b'G', b'0', 0x00, 0x0c, 0, 0, 0, 0, 0x0c, 1, 0, 0, 0,
        // Field (REG0, AnyAcc, Lock, Preserve) { DATA, 8 }
        0x5b, 0x81, 0x0b, b'R', b'E', b'G', b'0', 0x10, b'D', b'A', b'T', b'A', 0x08,
        // Method (MAIN, 0) { Return (DATA) }
        0x14, 0x0b, b'M', b'A', b'I', b'N', 0x00, 0xa4, b'D', b'A', b'T', b'A',
    ];
    interpreter.load_table(&aml).expect("locked field AML should parse");
    interpreter.initialize_native_regions().expect("native region setup");

    let value = interpreter
        .evaluate(acpi::aml::namespace::AmlName::from_str("\\MAIN").unwrap(), vec![])
        .expect("LockRule=Lock field access should use the host Global Lock");
    assert_eq!(value.as_integer().expect("locked field returns integer"), 0);
    assert!(interpreter.has_global_lock_dependency());
}

#[derive(Clone)]
struct LockHandler {
    // Use RawReentrantMutex for two reasons:
    // 1. We need a raw mutex because we don't want to hold onto the RAII MutexGuard - we'll control
    //    locking and unlocking manually
    // 2. We need to handle reentrancy, so RawMutex by itself is insufficient.
    mutex: Arc<RawReentrantMutex<RawMutex, RawThreadId>>,
}

impl LockHandler {
    const MUTEX_HANDLE: Handle = Handle(1);

    pub fn new() -> Self {
        Self { mutex: Arc::new(RawReentrantMutex::INIT) }
    }
}

impl Handler for LockHandler {
    fn create_mutex(&self) -> Handle {
        // Don't add complexity for a simple test handler - we only need one mutex.
        Self::MUTEX_HANDLE
    }

    fn acquire(&self, mutex: Handle, timeout: u16) -> Result<(), AmlError> {
        assert_eq!(mutex, Self::MUTEX_HANDLE);

        match timeout {
            0 => self.mutex.try_lock(),
            0xffff => {
                self.mutex.lock();
                true
            }
            _ => self.mutex.try_lock_for(Duration::from_millis(timeout as u64)),
        }
        .ok_or(AmlError::MutexAcquireTimeout)
    }

    fn release(&self, mutex: Handle) {
        assert_eq!(mutex, Self::MUTEX_HANDLE);
        assert!(self.mutex.is_owned_by_current_thread());

        // Safety: We've just checked that it's this thread that owns the mutex, so it's safe to
        // unlock.
        unsafe {
            self.mutex.unlock();
        }
    }

    unsafe fn map_physical_region<T>(&self, _physical_address: usize, _size: usize) -> RawPhysicalMapping<T> {
        unimplemented!()
    }

    unsafe fn unmap_physical_region<T>(&self, _region: RawPhysicalMapping<T>) {
        // Do nothing
    }

    fn read_u8(&self, _address: usize) -> u8 {
        0
    }

    fn read_u16(&self, _address: usize) -> u16 {
        unimplemented!()
    }

    fn read_u32(&self, _address: usize) -> u32 {
        unimplemented!()
    }

    fn read_u64(&self, _address: usize) -> u64 {
        unimplemented!()
    }

    fn write_u8(&self, _address: usize, _value: u8) {
        unimplemented!()
    }

    fn write_u16(&self, _address: usize, _value: u16) {
        unimplemented!()
    }

    fn write_u32(&self, _address: usize, _value: u32) {
        unimplemented!()
    }

    fn write_u64(&self, _address: usize, _value: u64) {
        unimplemented!()
    }

    fn read_io_u8(&self, _port: u16) -> u8 {
        unimplemented!()
    }

    fn read_io_u16(&self, _port: u16) -> u16 {
        unimplemented!()
    }

    fn read_io_u32(&self, _port: u16) -> u32 {
        unimplemented!()
    }

    fn write_io_u8(&self, _port: u16, _value: u8) {
        unimplemented!()
    }

    fn write_io_u16(&self, _port: u16, _value: u16) {
        unimplemented!()
    }

    fn write_io_u32(&self, _port: u16, _value: u32) {
        unimplemented!()
    }

    fn read_pci_u8(&self, _address: PciAddress, _offset: u16) -> u8 {
        unimplemented!()
    }

    fn read_pci_u16(&self, _address: PciAddress, _offset: u16) -> u16 {
        unimplemented!()
    }

    fn read_pci_u32(&self, _address: PciAddress, _offset: u16) -> u32 {
        unimplemented!()
    }

    fn write_pci_u8(&self, _address: PciAddress, _offset: u16, _value: u8) {
        unimplemented!()
    }

    fn write_pci_u16(&self, _address: PciAddress, _offset: u16, _value: u16) {
        unimplemented!()
    }

    fn write_pci_u32(&self, _address: PciAddress, _offset: u16, _value: u32) {
        unimplemented!()
    }

    fn nanos_since_boot(&self) -> u64 {
        unimplemented!()
    }

    fn stall(&self, _microseconds: u64) {
        unimplemented!()
    }

    fn sleep(&self, _milliseconds: u64) {
        unimplemented!()
    }
}
