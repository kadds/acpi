use acpi::aml::{
    AmlError,
    ExecutionMeter,
    Interpreter,
    namespace::{AmlName, NamespaceLevelKind},
    object::Object,
    op_region::{OpRegion, RegionSpace},
};
use aml_test_tools::{
    RunTestResult,
    TestFailureReason,
    handlers::{
        native_region_handler::TestNativeRegionHandler,
        std_test_handler::{
            Command,
            construct_std_handler,
            create_mutex,
            read_io_u8,
            read_pci_u8,
            read_u8,
            write_io_u8,
            write_pci_u8,
            write_u8,
        },
    },
    new_interpreter,
    run_test_for_string,
};
use pci_types::PciAddress;
use std::{
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

struct AllocationMeter {
    buffers: Arc<AtomicU64>,
}

impl ExecutionMeter for AllocationMeter {
    fn charge_opcode(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn enter_method(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn leave_method(&self) {}

    fn loop_iteration(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn reserve_buffer(&self, bytes: u64) -> Result<(), AmlError> {
        self.buffers.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    fn reserve_working_allocation_scoped(
        &self,
        _heap_bytes: u64,
        _package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<u64, AmlError> {
        self.reserve_buffer(buffer_bytes)?;
        Ok(0)
    }
}

struct LimitedBufferMeter {
    buffers: AtomicU64,
    max_buffer_bytes: u64,
}

impl ExecutionMeter for LimitedBufferMeter {
    fn charge_opcode(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn enter_method(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn leave_method(&self) {}

    fn loop_iteration(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn reserve_buffer(&self, bytes: u64) -> Result<(), AmlError> {
        let current = self.buffers.load(Ordering::Relaxed);
        let Some(next) = current.checked_add(bytes) else {
            return Err(AmlError::HostError("TEST_BUFFER_BUDGET".to_string()));
        };
        if next > self.max_buffer_bytes {
            return Err(AmlError::HostError("TEST_BUFFER_BUDGET".to_string()));
        }
        self.buffers.store(next, Ordering::Relaxed);
        Ok(())
    }

    fn rollback_buffer_reservation(&self, bytes: u64) {
        self.buffers.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn reserve_working_allocation_scoped(
        &self,
        _heap_bytes: u64,
        _package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<u64, AmlError> {
        self.reserve_buffer(buffer_bytes)?;
        Ok(0)
    }

    fn release_working_allocation_scoped(
        &self,
        _heap_bytes: u64,
        _package_bytes: u64,
        buffer_bytes: u64,
        _generation: u64,
    ) {
        self.rollback_buffer_reservation(buffer_bytes);
    }

    fn release_persistent_allocation(
        &self,
        _heap_bytes: u64,
        _package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<(), AmlError> {
        self.rollback_buffer_reservation(buffer_bytes);
        Ok(())
    }
}

struct BoundedCallMeter {
    depth: AtomicU64,
    max_depth: u64,
}

impl ExecutionMeter for BoundedCallMeter {
    fn charge_opcode(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn enter_method(&self) -> Result<(), AmlError> {
        let depth = self.depth.fetch_add(1, Ordering::Relaxed) + 1;
        if depth <= self.max_depth {
            Ok(())
        } else {
            self.depth.fetch_sub(1, Ordering::Relaxed);
            Err(AmlError::HostError("TEST_CALL_DEPTH".to_string()))
        }
    }

    fn leave_method(&self) {
        self.depth.fetch_sub(1, Ordering::Relaxed);
    }

    fn loop_iteration(&self) -> Result<(), AmlError> {
        Ok(())
    }
}

mod test_infra;

#[test]
fn native_region_registration_is_once_per_scope_and_space() {
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    let registrations = Arc::new(AtomicU64::new(0));
    let registrations_for_method = Arc::clone(&registrations);
    {
        let mut namespace = interpreter.namespace.lock();
        namespace.add_level(AmlName::from_str("\\TEST").unwrap(), NamespaceLevelKind::Device).unwrap();
        namespace
            .insert(
                AmlName::from_str("\\TEST._REG").unwrap(),
                Object::native_method(2, move |args| {
                    assert_eq!(args[0].as_integer().unwrap(), 0);
                    assert!(matches!(args[1].as_integer().unwrap(), 0 | 1));
                    registrations_for_method.fetch_add(1, Ordering::Relaxed);
                    Ok(Object::Integer(0).wrap())
                })
                .wrap(),
            )
            .unwrap();
        namespace
            .insert(
                AmlName::from_str("\\TEST.MEM0").unwrap(),
                Object::OpRegion(OpRegion {
                    space: RegionSpace::SystemMemory,
                    base: 0x40000,
                    length: 0x10,
                    parent_device_path: AmlName::from_str("\\TEST").unwrap(),
                })
                .wrap(),
            )
            .unwrap();
        namespace
            .insert(
                AmlName::from_str("\\TEST.MEM1").unwrap(),
                Object::OpRegion(OpRegion {
                    space: RegionSpace::SystemMemory,
                    base: 0x50000,
                    length: 0x10,
                    parent_device_path: AmlName::from_str("\\TEST").unwrap(),
                })
                .wrap(),
            )
            .unwrap();
    }
    interpreter.initialize_native_regions().expect("native region registration");
    assert_eq!(registrations.load(Ordering::Relaxed), 1);
    interpreter.initialize_native_regions().expect("idempotent region registration");
    assert_eq!(registrations.load(Ordering::Relaxed), 1);
    interpreter.deinitialize_native_regions().expect("native region de-registration");
    assert_eq!(registrations.load(Ordering::Relaxed), 2);
}

#[test]
fn dynamic_region_redefinition_invalidates_cached_tokens() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "DYNREG", 1) {
        Method(MAIN, 0, NotSerialized) {
            OperationRegion(MEM0, SystemMemory, 0x40000, 0x10)
            Return(Zero)
        }
    }"#;

    let raw_handler = construct_std_handler(vec![create_mutex()]);
    let native_handler = TestNativeRegionHandler::new(raw_handler.clone(), Vec::new());
    let invalidations = native_handler.invalidation_count();
    let interpreter = Interpreter::new(raw_handler, 2);
    interpreter.install_native_region_handler(Box::new(native_handler));

    let interpreter = match run_test_for_string(AML, interpreter, &None) {
        RunTestResult::Pass(interpreter) => interpreter,
        result => panic!("dynamic OperationRegion setup failed: {result:?}"),
    };
    let invalidations_after_first_definition = invalidations.load(Ordering::Relaxed);
    interpreter
        .evaluate(AmlName::from_str("\\MAIN").unwrap(), Vec::new())
        .expect("dynamic OperationRegion should be redefined");
    assert!(
        invalidations.load(Ordering::Relaxed) > invalidations_after_first_definition,
        "redefining a connected scope/space must invalidate its previous region token"
    );
}

#[test]
fn disconnect_callback_cannot_access_a_region_being_deactivated() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "REGDISC", 1) {
        OperationRegion(MEM0, SystemMemory, 0x40000, 0x10)
        Field(MEM0, ByteAcc, NoLock, Preserve) { DATA, 8 }

        Method(_REG, 2, NotSerialized) {
            If (LEqual(Arg1, Zero)) { Return(DATA) }
            Return(Zero)
        }

        Method(MAIN, 0, NotSerialized) { Return(Zero) }
    }"#;

    let interpreter = test_infra::run_aml_test(AML, construct_std_handler(vec![create_mutex()]));
    assert_eq!(
        interpreter.deinitialize_native_regions(),
        Err(AmlError::NoHandlerForRegionAccess(RegionSpace::SystemMemory)),
        "_REG(..., 0) must observe the region as disconnected"
    );
}

#[test]
fn create_qword_field_rejects_bit_offset_overflow_without_panicking() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 2, "RSACPI", "QWOFLOW", 1) {
        Name(BUF0, Buffer(0x08) { 0, 0, 0, 0, 0, 0, 0, 0 })
        Method(MAIN, 0, NotSerialized) {
            CreateQWordField(BUF0, 0xFFFFFFFFFFFFFFFF, QWRD)
            Return(Zero)
        }
    }"#;

    let result = run_test_for_string(AML, new_interpreter(construct_std_handler(vec![create_mutex()])), &None);
    assert!(
        matches!(result, RunTestResult::Failed(_, TestFailureReason::ParseFail(AmlError::IndexOutOfBounds))),
        "overflowing CreateQWordField offset should return IndexOutOfBounds: {result:?}"
    );
}

#[test]
fn buffer_field_write_outside_buffer_returns_error_without_panicking() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "BFOOBND", 1) {
        Name(BUF0, Buffer(0x01) { 0 })
        Method(MAIN, 0, NotSerialized) {
            CreateByteField(BUF0, 1, BYTE)
            Store(1, BYTE)
            Return(Zero)
        }
    }"#;

    let result = run_test_for_string(AML, new_interpreter(construct_std_handler(vec![create_mutex()])), &None);
    assert!(
        matches!(result, RunTestResult::Failed(_, TestFailureReason::ParseFail(AmlError::IndexOutOfBounds))),
        "out-of-range BufferField write should return IndexOutOfBounds: {result:?}"
    );
}

#[test]
fn compatibility_pic_is_single_use_and_rejects_replay() {
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    let calls = Arc::new(AtomicU64::new(0));
    let calls_for_method = Arc::clone(&calls);
    interpreter
        .namespace
        .lock()
        .insert(
            AmlName::from_str("\\_PIC").unwrap(),
            Object::native_method(1, move |args| {
                assert_eq!(args[0].as_integer().unwrap(), 1);
                calls_for_method.fetch_add(1, Ordering::Relaxed);
                Ok(Object::Integer(0).wrap())
            })
            .wrap(),
        )
        .unwrap();

    interpreter
        .evaluate_pic_once(vec![Object::Integer(1).wrap()])
        .expect("first _PIC evaluation")
        .expect("_PIC should be present");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(interpreter.evaluate_pic_once(vec![Object::Integer(1).wrap()]).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn compatibility_pic_reentry_from_aml_is_rejected_before_recursive_execution() {
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    // Method (_PIC, 1) { Return (_PIC (Arg0)) }
    interpreter
        .load_table(&[0x14, 0x0c, b'_', b'P', b'I', b'C', 0x01, 0xa4, b'_', b'P', b'I', b'C', 0x68])
        .expect("recursive _PIC method should load");
    interpreter.install_execution_meter(Arc::new(BoundedCallMeter { depth: AtomicU64::new(0), max_depth: 8 }));

    let error = interpreter
        .evaluate_pic_once(vec![Object::Integer(1).wrap()])
        .expect_err("recursive _PIC must fail closed");
    assert!(format!("{error:?}").contains("NOT_SUPPORTED_PIC_REENTRY"), "unexpected _PIC error: {error:?}");
}

#[test]
fn locked_field_access_is_rejected_before_region_access() {
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    let aml = [
        // OperationRegion (REG0, SystemMemory, 0, 0x1000)
        0x5b, 0x80, b'R', b'E', b'G', b'0', 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x00,
        // Field (REG0, AnyAcc, Lock, Preserve) { DATA, 8 }
        0x5b, 0x81, 0x0b, b'R', b'E', b'G', b'0', 0x10, b'D', b'A', b'T', b'A', 0x08,
        // Method (MAIN, 0) { Return (DATA) }
        0x14, 0x0b, b'M', b'A', b'I', b'N', 0x00, 0xa4, b'D', b'A', b'T', b'A',
    ];
    interpreter.load_table(&aml).expect("locked field AML should parse");

    let error = interpreter
        .evaluate(AmlName::from_str("\\MAIN").unwrap(), vec![])
        .expect_err("LockRule=Lock field access must fail closed without FACS lock support");
    assert!(format!("{error:?}").contains("NOT_SUPPORTED_GLOBAL_LOCK"));
    assert!(interpreter.has_global_lock_dependency());
}

#[test]
fn execution_meter_observes_runtime_buffer_allocation() {
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    let buffers = Arc::new(AtomicU64::new(0));
    interpreter.install_execution_meter(Arc::new(AllocationMeter { buffers: Arc::clone(&buffers) }));

    // Name (BUF0, Buffer (0x04) { 1, 2, 3, 4 })
    interpreter
        .load_table(&[0x08, b'B', b'U', b'F', b'0', 0x11, 0x07, 0x0a, 0x04, 1, 2, 3, 4])
        .expect("buffer AML should load");
    assert_eq!(buffers.load(Ordering::Relaxed), 4);
}

#[test]
fn oversized_buffer_field_read_is_rejected_before_materializing_output() {
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    interpreter
        .install_execution_meter(Arc::new(LimitedBufferMeter { buffers: AtomicU64::new(0), max_buffer_bytes: 4 }));

    // BUF0 is four bytes. MAIN creates a one-megabit BufferField and returns
    // it, which would otherwise materialize a 128 KiB temporary Buffer.
    interpreter
        .load_table(&[
            0x08, b'B', b'U', b'F', b'0', 0x11, 0x07, 0x0a, 0x04, 1, 2, 3, 4, 0x14, 0x1b, b'M', b'A', b'I', b'N',
            0x00, 0x5b, 0x13, b'B', b'U', b'F', b'0', 0x00, 0x0c, 0x00, 0x00, 0x10, 0x00, b'F', b'L', b'D', b'0',
            0xa4, b'F', b'L', b'D', b'0',
        ])
        .expect("buffer field method should load");

    let error = interpreter
        .evaluate(AmlName::from_str("\\MAIN").unwrap(), vec![])
        .expect_err("field materialization must respect the runtime buffer budget");
    assert!(format!("{error:?}").contains("TEST_BUFFER_BUDGET"));
}

#[test]
fn logical_buffer_string_conversion_is_metered_and_released() {
    let aml = [
        // Name (BUF0, Buffer (0x04) { 'A', 'B', 'C', 'D' })
        0x08, b'B', b'U', b'F', b'0', 0x11, 0x07, 0x0a, 0x04, b'A', b'B', b'C', b'D',
        // Method (MAIN, 0) { Return (LEqual (BUF0, "ABCD")) }
        0x14, 0x12, b'M', b'A', b'I', b'N', 0x00, 0xa4, 0x93, b'B', b'U', b'F', b'0', 0x0d, b'A', b'B', b'C', b'D',
        0x00,
    ];
    let rejected_interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    let rejected_meter = Arc::new(LimitedBufferMeter { buffers: AtomicU64::new(0), max_buffer_bytes: 3 });
    rejected_interpreter.load_table(&aml).expect("logical conversion AML should parse");
    rejected_interpreter.install_execution_meter(rejected_meter.clone());
    let error = rejected_interpreter
        .evaluate(AmlName::from_str("\\MAIN").unwrap(), vec![])
        .expect_err("logical temporary buffer must obey the buffer budget");
    assert!(format!("{error:?}").contains("TEST_BUFFER_BUDGET"));
    assert_eq!(rejected_meter.buffers.load(Ordering::Relaxed), 0);

    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    let meter = Arc::new(LimitedBufferMeter { buffers: AtomicU64::new(0), max_buffer_bytes: 4 });
    interpreter.load_table(&aml).expect("logical conversion AML should parse");
    interpreter.install_execution_meter(meter.clone());
    let result = interpreter
        .evaluate(AmlName::from_str("\\MAIN").unwrap(), vec![])
        .expect("bounded logical buffer conversion should succeed");
    assert!(matches!(*result, Object::Integer(u64::MAX)));
    assert_eq!(meter.buffers.load(Ordering::Relaxed), 0);
}

#[test]
fn test_region_in_top_level() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "OPREG", 1) {
    OperationRegion(MEM, SystemMemory, 0x40000, 0x1000)
    Field(MEM, ByteAcc, NoLock, Preserve) {
        DATA, 8
    }

    Method(RDWR, 0, NotSerialized) {
        DATA = 0xA5
        Return (DATA)
    }

    Method(MAIN, 0, NotSerialized) {
        Local0 = RDWR()
        // Remember that returning zero indicates success.
        Return (Local0 != 0xA5)
    }
}
"#;

    const EXPECTED_COMMANDS: &[Command] = &[
        create_mutex(),
        // DATA = 0xA5
        write_u8(0x40000, 0xA5),
        read_u8(0x40000, 0xA5),
    ];

    let handler = construct_std_handler(EXPECTED_COMMANDS.to_vec());
    test_infra::run_aml_test(AML, handler);
}

#[test]
fn test_region_in_device() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "OPREG", 1) {
    Device(TEST) {
        Name (_HID, EisaId ("PNP0C01")) // Arbitrary choice.

        OperationRegion(MEM, SystemMemory, 0x40000, 0x1000)
        Field(MEM, ByteAcc, NoLock, Preserve) {
            DATA, 8
        }

        Method(RDWR, 0, NotSerialized) {
            DATA = 0xA5
            Return (DATA)
        }
    }

    Method(MAIN, 0, NotSerialized) {
        Local0 = ^TEST.RDWR()
        // Remember that returning zero indicates success.
        Return (Local0 != 0xA5)
    }
}
"#;

    const EXPECTED_COMMANDS: &[Command] = &[
        create_mutex(),
        // DATA = 0xA5
        write_u8(0x40000, 0xA5),
        read_u8(0x40000, 0xA5),
    ];

    let handler = construct_std_handler(EXPECTED_COMMANDS.to_vec());
    test_infra::run_aml_test(AML, handler);
}

#[test]
fn test_region_in_method() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "OPREG", 1) {
    // iasl is not a fan of this function - it generates a bunch of warnings, which for testing
    // purposes can be ignored.
    Method(RDWR, 1, NotSerialized) {
        OperationRegion(MEM, SystemMemory, Arg0, 0x1000)
        Field(MEM, ByteAcc, NoLock, Preserve) {
            DATA, 8
        }

        DATA = 0xA5
        Return (DATA)
    }

    Method(MAIN, 0, NotSerialized) {
        Local0 = RDWR(0x40000)
        // Remember that returning zero indicates success.
        if (Local0 != 0xA5) {
            Return (One)
        }

        Local0 = RDWR(0x50000)
        Return (Local0 != 0xA5)
    }
}
"#;

    const EXPECTED_COMMANDS: &[Command] = &[
        create_mutex(),
        // Base set to 0x40000
        write_u8(0x40000, 0xA5),
        read_u8(0x40000, 0xA5),
        // Base set to 0x50000
        write_u8(0x50000, 0xA5),
        read_u8(0x50000, 0xA5),
    ];

    let handler = construct_std_handler(EXPECTED_COMMANDS.to_vec());
    test_infra::run_aml_test(AML, handler);
}

#[test]
#[ignore]
fn test_buffer_field_implicit_conv() {
    // This test covers https://github.com/rust-osdev/acpi/issues/273
    const AML: &str = r#"DefinitionBlock ("", "SSDT", 2, "AMD", "AmdTable", 0x00000002)
{
    Name (ADAT, Buffer (0x0010)
    {
        /* 0000 */  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // ........
        /* 0008 */  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // ........

    })
    OperationRegion (A053, SystemMemory, (DerefOf (ADAT [(0x04)])), 0x1000) // <--- this line failed
    Field (A053, ByteAcc, NoLock, Preserve)
    {
        Offset (0x18),
        A, 8
    }
}"#;

    const EXPECTED_COMMANDS: &[Command] = &[create_mutex()];

    let handler = construct_std_handler(EXPECTED_COMMANDS.to_vec());
    test_infra::run_aml_test(AML, handler);
}

#[test]
fn test_region_in_pci_device() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "OPREG", 1) {
    Device(TEST) {
        Name (_ADR, 0x00020001) // Arbitrary choice.
        Name (_BBN, 3)
        Method (_SEG, 0, NotSerialized) { Return (4) }

        OperationRegion(MEM, PCI_Config, 0x40, 0x20)
        Field(MEM, ByteAcc, NoLock, Preserve) {
        Offset (0x10),
            DATA, 8
        }

        Method(RDWR, 0, NotSerialized) {
            DATA = 0xA5
            Return (DATA)
        }
    }

    Method(MAIN, 0, NotSerialized) {
        Local0 = ^TEST.RDWR()
        // Remember that returning zero indicates success.
        Return (Local0 != 0xA5)
    }
}
"#;

    let address: PciAddress = PciAddress::new(4, 3, 2, 1);
    let expected_commands: &[Command] = &[
        create_mutex(),
        // DATA = 0xA5
        write_pci_u8(address, 0x50, 0xA5),
        read_pci_u8(address, 0x50, 0xA5),
    ];

    let handler = construct_std_handler(expected_commands.to_vec());
    test_infra::run_aml_test_with_pci_device(AML, handler, "\\TEST", address);
}

#[test]
fn built_in_native_regions_never_fall_back_to_raw_handler() {
    const AML: &str = r#"DefinitionBlock("", "DSDT", 1, "RSACPI", "NATREG", 1) {
    Device(TEST) {
        Name (_ADR, 0x00020001)
        Name (_BBN, 3)
        Method (_SEG, 0, NotSerialized) { Return (4) }

        OperationRegion(MEM, SystemMemory, 0x40000, 0x1000)
        Field(MEM, ByteAcc, NoLock, Preserve) { MEMD, 8 }

        OperationRegion(IO0, SystemIO, 0x40, 0x10)
        Field(IO0, ByteAcc, NoLock, Preserve) { IOD0, 8 }

        OperationRegion(PCI, PCI_Config, 0x40, 0x20)
        Field(PCI, ByteAcc, NoLock, Preserve) {
            Offset (0x10),
            PCID, 8
        }

        Method(RDWR, 0, NotSerialized) {
            MEMD = 0xA5
            IOD0 = 0xA5
            PCID = 0xA5
            Return (MEMD != 0xA5 || IOD0 != 0xA5 || PCID != 0xA5)
        }
    }

    Method(MAIN, 0, NotSerialized) { Return (^TEST.RDWR()) }
}
"#;

    let address = PciAddress::new(4, 3, 2, 1);
    let raw_handler = construct_std_handler(vec![create_mutex()]);
    let native_commands = vec![
        write_u8(0x40000, 0xA5),
        write_io_u8(0x40, 0xA5),
        write_pci_u8(address, 0x50, 0xA5),
        read_u8(0x40000, 0xA5),
        read_io_u8(0x40, 0xA5),
        read_pci_u8(address, 0x50, 0xA5),
    ];
    let native_handler = construct_std_handler(native_commands);
    let interpreter = Interpreter::new(raw_handler, 2);
    interpreter.install_native_region_handler(Box::new(TestNativeRegionHandler::new(
        native_handler,
        vec![(AmlName::from_str("\\TEST").unwrap(), address)],
    )));

    // The raw Handler permits only interpreter construction. If any built-in
    // region operation reaches its address-only callbacks, CheckCommandHandler
    // panics; the independent native Handler accounts for all expected I/O.
    let result = run_test_for_string(AML, interpreter, &None);
    assert!(matches!(&result, RunTestResult::Pass(_)), "native dispatch used the raw Handler: {result:?}");
}
