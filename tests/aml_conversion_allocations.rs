use acpi::aml::{
    AmlError,
    ExecutionMeter,
    IntegerSize,
    namespace::AmlName,
    object::Object,
    resource::{Resource, resource_descriptor_list_metered},
};
use aml_test_tools::{
    handlers::std_test_handler::{construct_std_handler, create_mutex},
    new_interpreter,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

struct CountingAllocator;

std::thread_local! {
    static FAIL_ALLOCATION_SIZE: Cell<usize> = const { Cell::new(0) };
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: This allocator forwards every operation to `System`; thread-local
// state scopes fault injection and allocation tracking to the calling test.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FAIL_ALLOCATION_SIZE.with(|size| size.get() == layout.size()) {
            FAIL_ALLOCATION_SIZE.with(|size| size.set(0));
            return std::ptr::null_mut();
        }
        TRACK_ALLOCATIONS.with(|tracking| {
            if tracking.get() {
                ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` are forwarded unchanged to the system allocator.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TRACK_ALLOCATIONS.with(|tracking| {
            if tracking.get() {
                ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
        // SAFETY: `pointer`, `layout`, and `new_size` are forwarded unchanged to the system allocator.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

struct NoopExecutionMeter;

impl ExecutionMeter for NoopExecutionMeter {
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
}

struct LimitedAmlHeapMeter {
    reserved_bytes: AtomicU64,
    maximum_bytes: u64,
}

impl LimitedAmlHeapMeter {
    fn reserve(&self, bytes: u64) -> Result<(), AmlError> {
        let mut current = self.reserved_bytes.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(bytes).ok_or(AmlError::BudgetExceeded)?;
            if next > self.maximum_bytes {
                return Err(AmlError::BudgetExceeded);
            }
            match self.reserved_bytes.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, bytes: u64) {
        let _ = self
            .reserved_bytes
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| Some(current.saturating_sub(bytes)));
    }
}

impl ExecutionMeter for LimitedAmlHeapMeter {
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

    fn reserve_package(&self, bytes: u64) -> Result<(), AmlError> {
        self.reserve(bytes)
    }

    fn reserve_buffer(&self, bytes: u64) -> Result<(), AmlError> {
        self.reserve(bytes)
    }

    fn rollback_buffer_reservation(&self, bytes: u64) {
        self.release(bytes);
    }

    fn reserve_heap(&self, bytes: u64) -> Result<(), AmlError> {
        self.reserve(bytes)
    }

    fn release_heap(&self, bytes: u64) -> Result<(), AmlError> {
        self.release(bytes);
        Ok(())
    }

    fn reserve_namespace_object(&self, bytes: u64) -> Result<(), AmlError> {
        self.reserve(bytes)
    }

    fn reserve_working_allocation_scoped(
        &self,
        heap_bytes: u64,
        package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<u64, AmlError> {
        let bytes = heap_bytes
            .checked_add(package_bytes)
            .and_then(|total| total.checked_add(buffer_bytes))
            .ok_or(AmlError::BudgetExceeded)?;
        self.reserve(bytes)?;
        Ok(0)
    }

    fn release_working_allocation_scoped(
        &self,
        heap_bytes: u64,
        package_bytes: u64,
        buffer_bytes: u64,
        _generation: u64,
    ) {
        self.release(heap_bytes.saturating_add(package_bytes).saturating_add(buffer_bytes));
    }

    fn release_persistent_allocation(
        &self,
        heap_bytes: u64,
        package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<(), AmlError> {
        self.release(heap_bytes.saturating_add(package_bytes).saturating_add(buffer_bytes));
        Ok(())
    }
}

#[test]
fn scalar_integer_conversions_do_not_allocate() {
    let value = Object::String(String::from("0Xff"));
    ALLOCATION_COUNT.with(|count| count.set(0));

    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    let result = value.to_integer(IntegerSize::FourBytes);
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));

    assert_eq!(result, Ok(255));
    assert_eq!(ALLOCATION_COUNT.with(Cell::get), 0);

    let buffer = Object::Buffer(vec![0x78, 0x56, 0x34, 0x12, 0xef, 0xcd, 0xab, 0x90]).wrap();
    let field = Object::BufferField { buffer, offset: 0, length: 1 << 20 };
    ALLOCATION_COUNT.with(|count| count.set(0));

    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    let result = field.to_integer(IntegerSize::FourBytes);
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));

    assert_eq!(result, Ok(0x1234_5678));
    assert_eq!(ALLOCATION_COUNT.with(Cell::get), 0);
}

#[test]
fn buffer_field_allocator_exhaustion_returns_typed_budget_error() {
    let buffer = Object::Buffer(vec![0; 32]).wrap();
    let field = Object::BufferField { buffer, offset: 0, length: 128 };
    FAIL_ALLOCATION_SIZE.with(|size| size.set(16));

    assert!(matches!(field.read_buffer_field(IntegerSize::EightBytes), Err(AmlError::BudgetExceeded)));
}

#[test]
fn aml_method_allocation_exhaustion_propagates_without_allocating_an_error() {
    use std::str::FromStr;

    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    interpreter
        .load_table(&[
            // Name (BUF0, Buffer (0x04) { 1, 2, 3, 4 })
            0x08, b'B', b'U', b'F', b'0', 0x11, 0x07, 0x0a, 0x04, 1, 2, 3, 4,
            // Method (MAIN, 0) { Return (BufferField (BUF0, 0, 0x00100000)) }
            0x14, 0x1b, b'M', b'A', b'I', b'N', 0x00, 0x5b, 0x13, b'B', b'U', b'F', b'0', 0x00, 0x0c, 0x00, 0x00,
            0x10, 0x00, b'F', b'L', b'D', b'0', 0xa4, b'F', b'L', b'D', b'0',
        ])
        .expect("buffer field AML should load");

    // The buffer-field result is 128 KiB; fail exactly that fallible request,
    // leaving interpreter bookkeeping allocations unaffected.
    FAIL_ALLOCATION_SIZE.with(|size| size.set(128 * 1024));
    let result = interpreter.evaluate(AmlName::from_str("\\MAIN").unwrap(), vec![]);

    assert!(matches!(result, Err(AmlError::BudgetExceeded)));
}

#[test]
fn aml_method_body_allocator_exhaustion_returns_budget_error() {
    const METHOD_BODY_BYTES: usize = 37;
    const METHOD_PACKAGE_BYTES: usize = 1 + 4 + 1 + METHOD_BODY_BYTES;

    let mut table = vec![0x14, METHOD_PACKAGE_BYTES as u8, b'M', b'A', b'I', b'N', 0];
    table.extend(std::iter::repeat_n(0xa3, METHOD_BODY_BYTES));
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));

    FAIL_ALLOCATION_SIZE.with(|size| size.set(METHOD_BODY_BYTES));
    let result = interpreter.load_table(&table);

    assert!(matches!(result, Err(AmlError::BudgetExceeded)));
    assert_eq!(FAIL_ALLOCATION_SIZE.with(Cell::get), 0, "the method-body allocation was injected");
}

#[test]
fn malformed_varpackage_count_stops_at_the_aml_heap_budget() {
    let interpreter = new_interpreter(construct_std_handler(vec![create_mutex()]));
    interpreter.install_execution_meter(Arc::new(LimitedAmlHeapMeter {
        reserved_bytes: AtomicU64::new(0),
        maximum_bytes: 1024 * 1024,
    }));
    let aml = [
        // Name (TST0, One), followed by the minimized malformed VarPackage AML input.
        0x08, b'T', b'S', b'T', b'0', 0x01, 0x13, 0x54, 0x00, 0x92, 0x00, b'S', b'S', b'D', 0x01,
    ];

    assert!(matches!(interpreter.load_table(&aml), Err(AmlError::BudgetExceeded)));
}

#[test]
fn metered_resource_output_vector_allocator_exhaustion_returns_budget_error() {
    let template = Object::Buffer(vec![
        0x47, 0x01, 0x60, 0x00, 0x60, 0x00, 0x01, 0x01, // IO Port
        0x79, 0x00, // EndTag
    ])
    .wrap();
    let meter = NoopExecutionMeter;
    let initial_resource_capacity = 4;
    let allocation_bytes = core::mem::size_of::<Resource>() * initial_resource_capacity;

    FAIL_ALLOCATION_SIZE.with(|size| size.set(allocation_bytes));
    let result = resource_descriptor_list_metered(template, &meter);

    assert!(matches!(result, Err(AmlError::BudgetExceeded)));
    assert_eq!(FAIL_ALLOCATION_SIZE.with(Cell::get), 0, "the resource-vector allocation was injected");
}

#[test]
fn metered_resource_source_allocator_exhaustion_returns_budget_error() {
    const SOURCE_BYTES: usize = 211;
    let source = "S".repeat(SOURCE_BYTES);
    let payload_length = 45 + SOURCE_BYTES;
    let mut template = vec![0x8a, payload_length as u8, (payload_length >> 8) as u8];
    template.extend_from_slice(&[0, 0x0c, 0]);
    template.extend_from_slice(&[0; 40]);
    assert_eq!(template.len(), 46);
    template.push(0); // Resource Source Index
    template.extend_from_slice(source.as_bytes());
    template.push(0); // Resource Source terminator
    template.extend_from_slice(&[0x79, 0]); // EndTag
    let template = Object::Buffer(template).wrap();
    let meter = NoopExecutionMeter;

    FAIL_ALLOCATION_SIZE.with(|size| size.set(SOURCE_BYTES));
    let result = resource_descriptor_list_metered(template, &meter);

    assert!(matches!(result, Err(AmlError::BudgetExceeded)));
    assert_eq!(FAIL_ALLOCATION_SIZE.with(Cell::get), 0, "the Resource Source allocation was injected");
}

#[test]
fn metered_gpio_pin_allocator_exhaustion_returns_budget_error() {
    const PIN_COUNT: usize = 8;
    let source_name_offset = 23 + PIN_COUNT * core::mem::size_of::<u16>();
    let total_length = source_name_offset + 1;
    let payload_length = (total_length - 3) as u16;
    let mut template = vec![0; total_length];
    template[0] = 0x8c;
    template[1..3].copy_from_slice(&payload_length.to_le_bytes());
    template[3] = 1; // Revision.
    template[4] = 1; // GPIO I/O connection.
    template[14..16].copy_from_slice(&(23_u16).to_le_bytes());
    template[17..19].copy_from_slice(&(source_name_offset as u16).to_le_bytes());
    for index in 0..PIN_COUNT {
        let pin_offset = 23 + index * core::mem::size_of::<u16>();
        template[pin_offset..pin_offset + 2].copy_from_slice(&(index as u16).to_le_bytes());
    }
    let template = Object::Buffer(template).wrap();
    let meter = NoopExecutionMeter;

    FAIL_ALLOCATION_SIZE.with(|size| size.set(PIN_COUNT * core::mem::size_of::<u16>()));
    let result = resource_descriptor_list_metered(template, &meter);

    assert!(matches!(result, Err(AmlError::BudgetExceeded)));
    assert_eq!(FAIL_ALLOCATION_SIZE.with(Cell::get), 0, "the GPIO pin allocation was injected");
}
