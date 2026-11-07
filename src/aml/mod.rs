/*
 * TODO:
 *  - Field reads supporting custom handlers
 *  - Locked fields
 *  - Bank and index fields
 *  - Run `_REG` on supported op region handlers
 *  - Count operations performed and time
 *  - Correct DefStore / DefCopyObject behaviour
 *  - Load and LoadTable
 *  - DefDataRegion
 *  - Notify
 *
 *  - Method recursion depth?
 *  - Loop timeouts
 *  - Fuzzing and guarantee panic-free interpretation
 */

mod interrupt_model_used;
mod match_expr;

pub mod namespace;
pub mod object;
pub mod op_region;
pub mod pci_routing;
pub mod resource;
pub mod allocator;

use crate::{AcpiError, AmlTable, Handle, Handler, PhysicalMapping, platform::AcpiPlatform, sdt::SdtHeader};
use crate::aml::allocator::{Arc, BTreeMap, Box, String, Vec};
use alloc::{format, string::ToString};
use bit_field::BitField;
use core::{
    fmt,
    mem,
    ptr,
    slice,
    str::FromStr,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
};
pub use interrupt_model_used::InterruptModelUsed;
use log::{error, info, trace, warn};
use match_expr::{MatchOp, do_match};
use namespace::{AmlName, NameComponent, NameSeg, Namespace, NamespaceLevel, NamespaceLevelKind};
use object::{
    DeviceStatus,
    FieldFlags,
    FieldUnit,
    FieldUnitKind,
    FieldUpdateRule,
    MeteredObject,
    MethodFlags,
    Object,
    ObjectHeapReservation,
    ObjectToken,
    ObjectType,
    ReferenceKind,
    WrappedObject,
};
use op_region::{
    NativeRegionAccessMode,
    NativeRegionHandler,
    NativeRegionRegistrations,
    OpRegion,
    RegionHandler,
    RegionSpace,
};
use spinning_top::Spinlock;

/// Helper macro to extract an expected set of [`Argument`]s from the given [`OpInFlight`]. Use
/// like:
/// ``` ignore,rust
/// extract_args!(op => [Argument::Object(source), Argument::Object(target)]);
/// extract_args!(op[0..2] => [Argument::Object(source), Argument::Namespace(name)]);
/// ```
macro_rules! extract_args {
    ($op:ident => $args:tt) => {
        let $args = &$op.arguments[..] else {
            return Err(AmlError::InternalError(format!(
                "Operation has invalid argument types: {}, in {}:{}",
                stringify!($args),
                file!(),
                line!(),
            )));
        };
    };
    ($op:ident[$x:expr] => $args:tt) => {
        let $args = &$op.arguments[$x] else {
            return Err(AmlError::InternalError(format!(
                "Operation has invalid argument types: {}, in {}:{}",
                stringify!($args),
                file!(),
                line!(),
            )));
        };
    };
}

/// `BaseInterpreter` implements a virtual machine for the dynamic AML bytecode. It can be used by a
/// host operating system to load tables containing AML bytecode (generally the DSDT and SSDTs) and
/// will then manage the AML namespace and all objects created during the life of the system.
///
/// [`Interpreter`] and [`NoSendInterpreter`] provide sensible partial specialisations of
/// `BaseInterpreter` for the multi-threaded and single-threaded use cases.
pub struct BaseInterpreter<H, R>
where
    H: Handler,
    R: RegionHandler + ?Sized,
{
    handler: H,
    pub namespace: Spinlock<Namespace>,
    pub object_token: Spinlock<ObjectToken>,
    integer_size: IntegerSize,
    region_handlers: Spinlock<BTreeMap<RegionSpace, Box<R>>>,
    native_region_handler: Spinlock<Option<Arc<dyn NativeRegionHandler>>>,
    execution_meter: Spinlock<Option<Arc<dyn ExecutionMeter>>>,

    global_lock_mutex: Handle,
    global_lock_dependency: AtomicBool,
    global_lock_poisoned: AtomicBool,
    global_lock_owner: Spinlock<Option<(u64, u32)>>,
    routing_methods_enabled: AtomicBool,
    native_region_registration_active: AtomicBool,
    native_region_dynamic_registration_active: AtomicBool,
    native_region_registrations: Spinlock<NativeRegionRegistrations>,
    /// Compatibility `_PIC` lifecycle: 0 = not evaluated, 1 = active,
    /// 2 = completed.  The host owns the only transition into the active
    /// state; AML cannot invoke `_PIC` recursively or after completion.
    pic_evaluation_state: AtomicU8,
    pic_method: Spinlock<Option<WrappedObject>>,
    pic_method_sealed: AtomicBool,
}

/// An Interpreter that does not enforce a requirement to be Send. This might be useful for
/// uni-processor kernels, or operating systems where this crate runs in a single-threaded user-mode
/// process.
pub type NoSendInterpreter<H> = BaseInterpreter<H, dyn RegionHandler>;

/// An Interpreter that is known to be Send + Sync. The version you should prefer by default.
pub type Interpreter<H> = BaseInterpreter<H, dyn RegionHandler + Send + Sync>;

struct MethodMeterGuard<'a, H, R>
where
    H: Handler,
    R: RegionHandler + ?Sized,
{
    interpreter: &'a BaseInterpreter<H, R>,
    active: usize,
}

impl<'a, H, R> MethodMeterGuard<'a, H, R>
where
    H: Handler,
    R: RegionHandler + ?Sized,
{
    fn new(interpreter: &'a BaseInterpreter<H, R>) -> Self {
        Self { interpreter, active: 0 }
    }

    fn enter(&mut self) -> Result<(), AmlError> {
        self.interpreter.enter_method()?;
        self.active += 1;
        Ok(())
    }

    fn leave(&mut self) {
        if self.active != 0 {
            self.active -= 1;
            self.interpreter.leave_method();
        }
    }
}

impl<H, R> Drop for MethodMeterGuard<'_, H, R>
where
    H: Handler,
    R: RegionHandler + ?Sized,
{
    fn drop(&mut self) {
        while self.active != 0 {
            self.active -= 1;
            self.interpreter.leave_method();
        }
    }
}

/// Host-owned execution accounting for the AML ordering domain.
pub trait ExecutionMeter: Send + Sync {
    fn charge_opcode(&self) -> Result<(), AmlError>;
    fn enter_method(&self) -> Result<(), AmlError>;
    fn leave_method(&self);
    fn loop_iteration(&self) -> Result<(), AmlError>;
    fn check_deadline(&self) -> Result<(), AmlError> {
        Ok(())
    }
    fn remaining_deadline_ns(&self) -> Result<Option<u64>, AmlError> {
        self.check_deadline()?;
        Ok(None)
    }
    fn reserve_package(&self, _bytes: u64) -> Result<(), AmlError> {
        Ok(())
    }
    fn reserve_buffer(&self, _bytes: u64) -> Result<(), AmlError> {
        Ok(())
    }
    fn rollback_buffer_reservation(&self, _bytes: u64) {}
    fn reserve_heap(&self, _bytes: u64) -> Result<(), AmlError> {
        Ok(())
    }
    fn commit_working_heap(&self, _bytes: u64) -> Result<(), AmlError> {
        Ok(())
    }
    fn release_heap(&self, _bytes: u64) -> Result<(), AmlError> {
        Ok(())
    }
    fn reserve_working_heap(&self, bytes: u64) -> Result<(), AmlError> {
        self.reserve_heap(bytes)
    }
    fn reserve_working_heap_scoped(&self, bytes: u64) -> Result<u64, AmlError> {
        self.reserve_working_heap(bytes)?;
        Ok(0)
    }
    fn release_working_heap(&self, bytes: u64) {
        let _ = self.release_heap(bytes);
    }
    fn release_working_heap_scoped(&self, bytes: u64, _generation: u64) {
        self.release_working_heap(bytes);
    }
    fn commit_working_heap_scoped(&self, bytes: u64, _generation: u64) -> Result<(), AmlError> {
        self.commit_working_heap(bytes)
    }
    fn reserve_working_allocation_scoped(
        &self,
        heap_bytes: u64,
        _package_bytes: u64,
        _buffer_bytes: u64,
    ) -> Result<u64, AmlError> {
        self.reserve_working_heap_scoped(heap_bytes)
    }
    fn release_working_allocation_scoped(
        &self,
        heap_bytes: u64,
        _package_bytes: u64,
        _buffer_bytes: u64,
        generation: u64,
    ) {
        self.release_working_heap_scoped(heap_bytes, generation);
    }
    fn commit_working_allocation_scoped(
        &self,
        heap_bytes: u64,
        _package_bytes: u64,
        _buffer_bytes: u64,
        generation: u64,
    ) -> Result<(), AmlError> {
        self.commit_working_heap_scoped(heap_bytes, generation)
    }
    fn release_persistent_allocation(
        &self,
        heap_bytes: u64,
        _package_bytes: u64,
        _buffer_bytes: u64,
    ) -> Result<(), AmlError> {
        self.release_heap(heap_bytes)
    }
    fn reserve_namespace_object(&self, _heap_bytes: u64) -> Result<(), AmlError> {
        Ok(())
    }
    fn release_namespace_object(&self, _heap_bytes: u64) {}
}

struct NamespaceObjectHeapReservation {
    meter: Arc<dyn ExecutionMeter>,
    heap_bytes: u64,
}

impl fmt::Debug for NamespaceObjectHeapReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("NamespaceObjectHeapReservation").field("heap_bytes", &self.heap_bytes).finish()
    }
}

impl ObjectHeapReservation for NamespaceObjectHeapReservation {
    fn commit_to_persistent(&mut self) -> Result<(), AmlError> {
        Ok(())
    }
}

impl Drop for NamespaceObjectHeapReservation {
    fn drop(&mut self) {
        self.meter.release_namespace_object(self.heap_bytes);
    }
}

struct MeteredObjectHeapReservation {
    meter: Arc<dyn ExecutionMeter>,
    persistent: bool,
    generation: u64,
    heap_bytes: u64,
    package_bytes: u64,
    buffer_bytes: u64,
}

impl fmt::Debug for MeteredObjectHeapReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MeteredObjectHeapReservation")
            .field("persistent", &self.persistent)
            .field("generation", &self.generation)
            .field("heap_bytes", &self.heap_bytes)
            .field("package_bytes", &self.package_bytes)
            .field("buffer_bytes", &self.buffer_bytes)
            .finish()
    }
}

impl ObjectHeapReservation for MeteredObjectHeapReservation {
    fn commit_to_persistent(&mut self) -> Result<(), AmlError> {
        if !self.persistent {
            self.meter.commit_working_allocation_scoped(
                self.heap_bytes,
                self.package_bytes,
                self.buffer_bytes,
                self.generation,
            )?;
            self.persistent = true;
        }
        Ok(())
    }
}

impl Drop for MeteredObjectHeapReservation {
    fn drop(&mut self) {
        if self.persistent {
            let _ =
                self.meter.release_persistent_allocation(self.heap_bytes, self.package_bytes, self.buffer_bytes);
        } else {
            self.meter.release_working_allocation_scoped(
                self.heap_bytes,
                self.package_bytes,
                self.buffer_bytes,
                self.generation,
            );
        }
    }
}

fn reserve_metered_object_allocation(
    meter: Option<Arc<dyn ExecutionMeter>>,
    heap_bytes: u64,
    package_bytes: u64,
    buffer_bytes: u64,
) -> Result<Option<MeteredObjectHeapReservation>, AmlError> {
    let Some(meter) = meter else {
        return Ok(None);
    };
    let generation = meter.reserve_working_allocation_scoped(heap_bytes, package_bytes, buffer_bytes)?;
    Ok(Some(MeteredObjectHeapReservation {
        meter,
        persistent: false,
        generation,
        heap_bytes,
        package_bytes,
        buffer_bytes,
    }))
}

fn wrapped_object_cell_heap_bytes() -> Result<u64, AmlError> {
    object::wrapped_object_inner_heap_bytes()
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(AmlError::BudgetExceeded)
}

fn boxed_metered_reservation_heap_bytes() -> Result<u64, AmlError> {
    mem::size_of::<MeteredObjectHeapReservation>()
        .checked_add(24)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(AmlError::BudgetExceeded)
}

fn reserve_boxed_metered_object_allocation(
    meter: Option<Arc<dyn ExecutionMeter>>,
    heap_bytes: u64,
    package_bytes: u64,
    buffer_bytes: u64,
) -> Result<Option<Box<dyn ObjectHeapReservation>>, AmlError> {
    let heap_bytes =
        heap_bytes.checked_add(boxed_metered_reservation_heap_bytes()?).ok_or(AmlError::BudgetExceeded)?;
    Ok(reserve_metered_object_allocation(meter, heap_bytes, package_bytes, buffer_bytes)?
        .map(|reservation| allocator::box_new(reservation) as Box<dyn ObjectHeapReservation>))
}

#[doc(hidden)]
pub fn wrap_metered_object(
    object: Object,
    meter: Option<Arc<dyn ExecutionMeter>>,
) -> Result<WrappedObject, AmlError> {
    let cell_heap_reservation =
        reserve_boxed_metered_object_allocation(meter, wrapped_object_cell_heap_bytes()?, 0, 0)?;
    Ok(object.wrap_metered(cell_heap_reservation, None))
}

struct TemporaryBufferReservation {
    reservation: Option<MeteredObjectHeapReservation>,
}

impl Drop for TemporaryBufferReservation {
    fn drop(&mut self) {
        drop(self.reservation.take());
    }
}

fn reserve_temporary_buffer(
    meter: Option<Arc<dyn ExecutionMeter>>,
    bytes: usize,
) -> Result<TemporaryBufferReservation, AmlError> {
    let buffer_bytes = u64::try_from(bytes).map_err(|_| AmlError::BudgetExceeded)?;
    let heap_bytes = bytes
        .checked_add(if bytes == 0 { 0 } else { 24 })
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| AmlError::BudgetExceeded)?;
    let reservation = reserve_metered_object_allocation(meter, heap_bytes, 0, buffer_bytes)?;
    Ok(TemporaryBufferReservation { reservation })
}

/// Estimate persistent registration storage and the largest per-callback
/// scratch required by the native OperationRegion lifecycle.
#[doc(hidden)]
pub fn native_region_registration_heap_bytes(
    registration_capacity: usize,
    path_clone_bytes: u64,
    maximum_path_clone_bytes: u64,
) -> Option<(u64, u64)> {
    let registration_entry_bytes =
        registration_capacity.checked_mul(core::mem::size_of::<(AmlName, RegionSpace)>())?;
    let vector_arc_header_bytes = core::mem::size_of::<Vec<(AmlName, RegionSpace)>>()
        .checked_add(2_usize.checked_mul(core::mem::size_of::<usize>())?)?;
    let persistent_bytes = u64::try_from(vector_arc_header_bytes.checked_add(registration_entry_bytes)?)
        .ok()?
        .checked_add(path_clone_bytes)?;
    let callback_bytes = native_region_callback_scratch_heap_bytes(maximum_path_clone_bytes)?;
    Some((persistent_bytes.checked_add(callback_bytes)?, persistent_bytes))
}

fn native_region_callback_scratch_heap_bytes(maximum_path_clone_bytes: u64) -> Option<u64> {
    let resolved_path_bytes =
        maximum_path_clone_bytes.checked_add(u64::try_from(mem::size_of::<NameComponent>()).ok()?)?;
    let callback_path_bytes = resolved_path_bytes.checked_mul(6)?;
    let argument_count = 2_usize;
    let argument_vector_bytes = core::mem::size_of::<Vec<WrappedObject>>()
        .checked_add(argument_count.checked_mul(core::mem::size_of::<WrappedObject>())?)?;
    let object_allocation_bytes =
        core::mem::size_of::<Object>().checked_add(2_usize.checked_mul(core::mem::size_of::<usize>())?)?;
    let callback_argument_bytes = argument_count.checked_mul(object_allocation_bytes)?;
    let callback_bytes = callback_path_bytes
        .checked_add(u64::try_from(argument_vector_bytes.checked_add(callback_argument_bytes)?).ok()?)?
        .checked_add(512)?;
    Some(callback_bytes)
}

fn traverse_namespace_with_meter<F>(
    namespace: &Namespace,
    execution_meter: Option<&Arc<dyn ExecutionMeter>>,
    mut callback: F,
) -> Result<(), AmlError>
where
    F: FnMut(&AmlName, &NamespaceLevel) -> Result<bool, AmlError>,
{
    match execution_meter {
        Some(meter) => namespace.traverse_metered(meter.as_ref(), &mut callback),
        None => namespace.traverse(&mut callback),
    }
}

/// The value returned by the `Revision` opcode.
const INTERPRETER_REVISION: u64 = 1;

/// The maximum number of times a name used as a package element may resolve to another such name
/// before we give up. Real firmware doesn't chain these at all, but broken tables must not be able
/// to make us loop forever.
const MAX_NAME_PATH_INDIRECTIONS: usize = 8;

// The following constants are public as they are referenced from the AmlError docs.
/// How many nested function calls should we allow? This is a fairly arbitrary figure that could be
/// changed with experience if it proves too small.
///
/// The interpreter doesn't allocate a stack frame on the processor stack for each AML method call,
/// so we're not worried about overflowing the kernel or process stack. This is a basic way to
/// ensure the AML is not stuck in an infinite recursion (which would deadlock the caller)
pub const MAX_STACK_DEPTH: usize = 1000;

/// The maximum length of time a single loop should be allowed to execute for, in nanoseconds.
///
/// This is a fairly arbitrary limit to protect against infinite loops and the risk of system
/// deadlock.
pub const LOOP_TIMEOUT_NS: u64 = 10_000_000_000;
const METHOD_LOCK_SCAN_DEADLINE_INTERVAL: usize = 256;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MethodGlobalLockFacts {
    acquires_global_lock: bool,
    contains_acquire: bool,
}

fn charge_method_lock_scan(meter: Option<&dyn ExecutionMeter>, scanned_steps: &mut usize) -> Result<(), AmlError> {
    if let Some(meter) = meter {
        meter.loop_iteration()?;
    }
    *scanned_steps = scanned_steps.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
    if *scanned_steps % METHOD_LOCK_SCAN_DEADLINE_INTERVAL == 0 {
        if let Some(meter) = meter {
            meter.check_deadline()?;
        }
    }
    Ok(())
}

fn method_global_lock_facts(
    code: &[u8],
    meter: Option<&dyn ExecutionMeter>,
) -> Result<MethodGlobalLockFacts, AmlError> {
    let mut facts = MethodGlobalLockFacts::default();
    let mut scanned_steps = 0_usize;
    for (index, opcode) in code.windows(2).enumerate() {
        charge_method_lock_scan(meter, &mut scanned_steps)?;
        if opcode != [0x5b, 0x23] {
            continue;
        }
        facts.contains_acquire = true;
        let mut name_start = index.checked_add(2).ok_or(AmlError::BudgetExceeded)?;
        while code.get(name_start) == Some(&0x5e) {
            charge_method_lock_scan(meter, &mut scanned_steps)?;
            name_start = name_start.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
        }
        if code.get(name_start) == Some(&0x5c) {
            charge_method_lock_scan(meter, &mut scanned_steps)?;
            name_start = name_start.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
        }
        facts.acquires_global_lock |= code.get(name_start..name_start.saturating_add(4)) == Some(&b"_GL_"[..]);
    }
    if let Some(meter) = meter {
        meter.check_deadline()?;
    }
    Ok(facts)
}

#[cfg(test)]
mod global_lock_static_tests {
    use super::{METHOD_LOCK_SCAN_DEADLINE_INTERVAL, MethodGlobalLockFacts, method_global_lock_facts};
    use crate::aml::{AmlError, ExecutionMeter};
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct ScanMeter {
        maximum_iterations: usize,
        iterations: AtomicUsize,
        deadline_checks: AtomicUsize,
        reject_deadline: bool,
    }

    impl ExecutionMeter for ScanMeter {
        fn charge_opcode(&self) -> Result<(), AmlError> {
            Ok(())
        }

        fn enter_method(&self) -> Result<(), AmlError> {
            Ok(())
        }

        fn leave_method(&self) {}

        fn loop_iteration(&self) -> Result<(), AmlError> {
            let iteration = self.iterations.fetch_add(1, Ordering::Relaxed);
            if iteration >= self.maximum_iterations {
                return Err(AmlError::BudgetExceeded);
            }
            Ok(())
        }

        fn check_deadline(&self) -> Result<(), AmlError> {
            self.deadline_checks.fetch_add(1, Ordering::Relaxed);
            if self.reject_deadline { Err(AmlError::DeadlineExceeded) } else { Ok(()) }
        }
    }

    #[test]
    fn detects_rooted_global_lock_acquire_operand() {
        assert_eq!(
            method_global_lock_facts(&[0x5b, 0x23, 0x5c, b'_', b'G', b'L', b'_', 0x00], None),
            Ok(MethodGlobalLockFacts { acquires_global_lock: true, contains_acquire: true })
        );
        assert_eq!(
            method_global_lock_facts(&[0x5b, 0x23, 0x5c, b'M', b'U', b'T', b'0', 0x00], None),
            Ok(MethodGlobalLockFacts { acquires_global_lock: false, contains_acquire: true })
        );
    }

    #[test]
    fn global_lock_scan_consumes_loop_fuel() {
        let code = crate::aml_vec![0xa3; 1024];
        let meter = ScanMeter {
            maximum_iterations: 8,
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            reject_deadline: false,
        };

        assert_eq!(method_global_lock_facts(&code, Some(&meter)), Err(AmlError::BudgetExceeded));
        assert_eq!(meter.iterations.load(Ordering::Relaxed), 9);
        assert_eq!(meter.deadline_checks.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn global_lock_scan_checks_deadline_during_long_method_bodies() {
        let code = crate::aml_vec![0xa3; 1024];
        let meter = ScanMeter {
            maximum_iterations: usize::MAX,
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            reject_deadline: true,
        };

        assert_eq!(method_global_lock_facts(&code, Some(&meter)), Err(AmlError::DeadlineExceeded));
        assert_eq!(meter.iterations.load(Ordering::Relaxed), METHOD_LOCK_SCAN_DEADLINE_INTERVAL);
        assert_eq!(meter.deadline_checks.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn global_lock_scan_charges_qualified_name_prefixes() {
        let mut code = crate::aml_vec![0x5b, 0x23];
        code.extend(core::iter::repeat_n(0x5e, 16));
        code.extend_from_slice(&[0x5c, b'_', b'G', b'L', b'_', 0x00]);
        let meter = ScanMeter {
            maximum_iterations: 2,
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            reject_deadline: false,
        };

        assert_eq!(method_global_lock_facts(&code, Some(&meter)), Err(AmlError::BudgetExceeded));
        assert_eq!(meter.iterations.load(Ordering::Relaxed), 3);
    }
}

impl<H, R> BaseInterpreter<H, R>
where
    H: Handler,
    R: RegionHandler + ?Sized,
{
    /// Construct a new [`Interpreter`]. This does not load any tables - if you have an
    /// [`crate::AcpiTables`] already, construct an [`AcpiPlatform`] first and then use
    /// [`Interpreter::new_from_platform`]
    pub fn new(handler: H, dsdt_revision: u8) -> Self {
        info!("Initializing AML interpreter v{}", env!("CARGO_PKG_VERSION"));

        let global_lock_mutex = handler.create_mutex();

        BaseInterpreter {
            handler,
            namespace: Spinlock::new(Namespace::new(global_lock_mutex)),
            object_token: Spinlock::new(unsafe { ObjectToken::create_interpreter_token() }),
            integer_size: IntegerSize::from_revision(dsdt_revision),
            region_handlers: Spinlock::new(allocator::btree_map_new()),
            native_region_handler: Spinlock::new(None),
            execution_meter: Spinlock::new(None),
            global_lock_mutex,
            global_lock_dependency: AtomicBool::new(false),
            global_lock_poisoned: AtomicBool::new(false),
            global_lock_owner: Spinlock::new(None),
            routing_methods_enabled: AtomicBool::new(false),
            native_region_registration_active: AtomicBool::new(false),
            native_region_dynamic_registration_active: AtomicBool::new(false),
            native_region_registrations: Spinlock::new(NativeRegionRegistrations::default()),
            pic_evaluation_state: AtomicU8::new(0),
            pic_method: Spinlock::new(None),
            pic_method_sealed: AtomicBool::new(false),
        }
    }

    /// Construct a new [`Interpreter`] with the given [`AcpiPlatform`].
    pub fn new_from_platform(platform: &AcpiPlatform<H>) -> Result<BaseInterpreter<H, R>, AcpiError> {
        fn load_table<H, R>(interpreter: &BaseInterpreter<H, R>, table: AmlTable) -> Result<(), AcpiError>
        where
            H: Handler,
            R: RegionHandler + ?Sized,
        {
            let mapping = unsafe {
                PhysicalMapping::<_, SdtHeader>::new(
                    table.phys_address,
                    table.length as usize,
                    &interpreter.handler,
                )
            };
            let stream = unsafe {
                slice::from_raw_parts(
                    ptr::from_ref(&*mapping).byte_add(size_of::<SdtHeader>()) as *const u8,
                    table.length as usize - mem::size_of::<SdtHeader>(),
                )
            };
            interpreter.load_table(stream).map_err(AcpiError::Aml)?;
            Ok(())
        }

        // FACS is mutable firmware state. Its Global Lock is accessed only
        // through the host-owned typed callback; the interpreter never maps it.
        let dsdt = platform.tables.dsdt()?;
        let interpreter = BaseInterpreter::new(platform.handler.clone(), dsdt.revision);

        if let Err(err) = load_table(&interpreter, dsdt) {
            error!("Error while loading DSDT: {:?}. Continuing; this may cause downstream errors.", err);
        }

        for (i, ssdt) in platform.tables.ssdts().enumerate() {
            if let Err(err) = load_table(&interpreter, ssdt) {
                error!("Error while loading SSDT{}: {:?}. Continuing.", i, err);
            }
        }

        Ok(interpreter)
    }

    /// Load the supplied byte stream as an AML table. This should be only the encoded AML stream -
    /// not the header at the start of a table. If you've used [`Interpreter::new_from_platform`],
    /// you'll likely not need to load any tables manually.
    pub fn load_table(&self, stream: &[u8]) -> Result<(), AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let context = unsafe { MethodContext::new_from_table(stream, meter)? };
        self.do_execute_method(context)?;
        Ok(())
    }

    /// Evaluate an object at the given path in the namespace. If the object is a method, this
    /// invokes the method with the given set of arguments.
    pub fn evaluate(&self, path: AmlName, args: Vec<WrappedObject>) -> Result<WrappedObject, AmlError> {
        trace!("Invoking AML method: {}", path);

        let object = self.namespace.lock().get_absolute(&path)?;
        self.evaluate_object(path, object, args)
    }

    fn evaluate_object(
        &self,
        path: AmlName,
        object: WrappedObject,
        args: Vec<WrappedObject>,
    ) -> Result<WrappedObject, AmlError> {
        validate_compatibility_method_entry(
            &path,
            self.pic_evaluation_state.load(Ordering::Acquire),
            self.routing_methods_enabled.load(Ordering::Acquire),
        )?;
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let method_uses_global_lock = match &*object {
            Object::Method { code, .. } => method_global_lock_facts(code, meter.as_deref())?.acquires_global_lock,
            _ => false,
        };
        let device_requires_global_lock = self.namespace.lock().path_requires_global_lock(&path);
        if method_uses_global_lock || device_requires_global_lock {
            self.global_lock_dependency.store(true, Ordering::Release);
        }
        match &*object {
            Object::Method { .. } => {
                let execute_method = || {
                    let path = MeteredNameString::account_owned_name(path, meter.clone())?;
                    let namespace_path_reservation = path.reserve_clone_scratch()?;
                    self.namespace.lock().add_level(path.try_clone()?, NamespaceLevelKind::MethodLocals)?;
                    drop(namespace_path_reservation);
                    let context = MethodContext::new_from_method(object, args, path, meter)?;
                    self.do_execute_method(context)
                };
                if device_requires_global_lock {
                    self.with_global_lock(true, execute_method)
                } else {
                    execute_method()
                }
            }
            Object::NativeMethod { f, .. } => f(&args, meter),
            _ => Ok(object),
        }
    }

    /// Preserve the firmware `_PIC` method and atomically replace its public
    /// namespace entry with a host guard.  Only `evaluate_pic_once` may invoke
    /// the preserved method; AML calls during `_REG` or nested AML execution
    /// see the guard instead.
    pub fn seal_pic_method(&self, guard: WrappedObject) -> Result<(), AmlError> {
        if self.pic_method_sealed.load(Ordering::Acquire) {
            return Err(AmlError::HostError("NOT_SUPPORTED_PIC_REENTRY".to_string()));
        }
        let path = AmlName::from_str("\\_PIC")?;
        let mut namespace = self.namespace.lock();
        let original = match namespace.get(path.clone()) {
            Ok(object) => {
                if !matches!(&*object, Object::Method { .. } | Object::NativeMethod { .. }) {
                    return Err(AmlError::ObjectNotOfExpectedType {
                        expected: ObjectType::Method,
                        got: object.typ(),
                    });
                }
                Some(object.clone())
            }
            Err(AmlError::ObjectDoesNotExist(_)) => None,
            Err(error) => return Err(error),
        };
        namespace.insert_immutable(path, guard)?;
        *self.pic_method.lock() = original;
        self.pic_method_sealed.store(true, Ordering::Release);
        Ok(())
    }

    pub fn evaluate_if_present(
        &self,
        path: AmlName,
        args: Vec<WrappedObject>,
    ) -> Result<Option<WrappedObject>, AmlError> {
        match self.evaluate(path.clone(), args) {
            Ok(result) => Ok(Some(result)),
            Err(AmlError::ObjectDoesNotExist(not_present)) => {
                if path == not_present {
                    Ok(None)
                } else {
                    Err(AmlError::ObjectDoesNotExist(not_present))
                }
            }
            Err(other) => Err(other),
        }
    }

    /// Evaluate the compatibility `_PIC` method exactly once.  This is kept
    /// separate from `evaluate_if_present` so ordinary AML evaluation cannot
    /// accidentally turn the compatibility handshake into a replayable
    /// method call.
    pub fn evaluate_pic_once(&self, args: Vec<WrappedObject>) -> Result<Option<WrappedObject>, AmlError> {
        if self.pic_evaluation_state.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(AmlError::HostError("NOT_SUPPORTED_PIC_REENTRY".to_string()));
        }
        let path = AmlName::from_str("\\_PIC")?;
        let result = if self.pic_method_sealed.load(Ordering::Acquire) {
            match self.pic_method.lock().clone() {
                Some(method) => self.evaluate_object(path, method, args).map(Some),
                None => Ok(None),
            }
        } else {
            self.evaluate_if_present(path, args)
        };
        self.pic_evaluation_state.store(2, Ordering::Release);
        result
    }

    pub fn install_region_handler(&self, space: RegionSpace, handler: Box<R>) {
        let mut handlers = self.region_handlers.lock();
        assert!(handlers.get(&space).is_none(), "Tried to install handler for same space twice!");
        handlers.insert(space, handler);
    }

    /// Install the single capability-aware dispatch boundary for built-in
    /// SystemMemory, SystemIO, and PCI_Config regions.
    ///
    /// These spaces must not be implemented by the address-only `Handler`
    /// callbacks.  Keeping one boundary also makes the handler responsible
    /// for resolving and invalidating NaOS region tokens consistently.
    pub fn install_native_region_handler(&self, handler: Box<dyn NativeRegionHandler>) {
        let mut installed = self.native_region_handler.lock();
        assert!(installed.is_none(), "Tried to install native region handler twice!");
        *installed = Some(Arc::from(handler));
    }

    /// Install execution accounting for the interpreter's real dispatch loop.
    pub fn install_execution_meter(&self, meter: Arc<dyn ExecutionMeter>) {
        let mut installed = self.execution_meter.lock();
        assert!(installed.is_none(), "Tried to install execution meter twice!");
        *installed = Some(meter);
    }

    /// Whether parsing or executing AML has observed a dependency on the
    /// firmware Global Lock or a locked FieldUnit.
    pub fn has_global_lock_dependency(&self) -> bool {
        self.global_lock_dependency.load(Ordering::Acquire)
    }

    /// Detect device `_GLK` flags in the parsed namespace and statically
    /// encoded `Acquire (\_GL_, ...)` references in method bodies. This avoids
    /// scanning arbitrary table bytes, where the same byte sequence may occur
    /// in a string or buffer payload. Dynamic aliases are detected on execute.
    pub fn try_detect_global_lock_namespace_dependency(&self) -> Result<bool, AmlError> {
        if self.has_global_lock_dependency() {
            return Ok(true);
        }
        let global_lock_flag_name = NameSeg::from_str("_GLK").map_err(|_| AmlError::InvalidName(None))?;
        let mut found = false;
        let execution_meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let namespace = self.namespace.lock();
        traverse_namespace_with_meter(&namespace, execution_meter.as_ref(), |_, level| {
            found |= level.values.contains_key(&global_lock_flag_name);
            Ok(!found)
        })?;
        if found {
            self.global_lock_dependency.store(true, Ordering::Release);
        }
        Ok(self.has_global_lock_dependency())
    }

    /// Compatibility wrapper for callers that require fail-closed detection.
    /// Bounded hosts should use the fallible method so budget exhaustion is
    /// reported instead of being indistinguishable from a real dependency.
    pub fn detect_global_lock_namespace_dependency(&self) -> bool {
        match self.try_detect_global_lock_namespace_dependency() {
            Ok(dependency) => dependency,
            Err(_) => {
                self.global_lock_dependency.store(true, Ordering::Release);
                true
            }
        }
    }

    /// Allow `_PRT` evaluation only after the host has frozen the compatibility
    /// profile and completed the one ordered `_PIC` decision.
    pub fn enable_routing_methods(&self) {
        self.routing_methods_enabled.store(true, Ordering::Release);
    }

    fn charge_opcode(&self) -> Result<(), AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        if let Some(meter) = meter {
            meter.charge_opcode()?;
        }
        Ok(())
    }

    fn enter_method(&self) -> Result<(), AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        if let Some(meter) = meter {
            meter.enter_method()?;
        }
        Ok(())
    }

    fn leave_method(&self) {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        if let Some(meter) = meter {
            meter.leave_method();
        }
    }

    fn charge_loop_iteration(&self) -> Result<(), AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        if let Some(meter) = meter {
            meter.loop_iteration()?;
        }
        Ok(())
    }

    fn reserve_temporary_buffer(&self, bytes: usize) -> Result<TemporaryBufferReservation, AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        reserve_temporary_buffer(meter, bytes)
    }

    fn reserve_object_heap(
        &self,
        heap_bytes: u64,
        package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<Option<Box<dyn ObjectHeapReservation>>, AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        reserve_boxed_metered_object_allocation(meter, heap_bytes, package_bytes, buffer_bytes)
    }

    fn wrap_object(&self, object: Object) -> Result<WrappedObject, AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        wrap_metered_object(object, meter)
    }

    fn wrap_object_with_heap_reservation(
        &self,
        object: Object,
        heap_reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> Result<WrappedObject, AmlError> {
        let cell_heap_reservation = self.reserve_object_heap(wrapped_object_cell_heap_bytes()?, 0, 0)?;
        Ok(object.wrap_metered(cell_heap_reservation, heap_reservation))
    }

    fn read_buffer_field(&self, object: &Object) -> Result<WrappedObject, AmlError> {
        let heap_reservation = match object {
            Object::BufferField { length, .. } if *length > self.integer_size as usize => {
                let bytes = length.div_ceil(8);
                let bytes = u64::try_from(bytes).map_err(|_| AmlError::BudgetExceeded)?;
                let heap_bytes = bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?;
                self.reserve_object_heap(heap_bytes, 0, bytes)?
            }
            _ => None,
        };

        self.wrap_object_with_heap_reservation(object.read_buffer_field(self.integer_size)?, heap_reservation)
    }

    fn try_clone_object(&self, object: &Object) -> Result<MeteredObject, AmlError> {
        let (object, heap_reservation) = match object {
            Object::Buffer(bytes) => {
                let buffer_bytes = u64::try_from(bytes.len()).map_err(|_| AmlError::BudgetExceeded)?;
                let heap_bytes = buffer_bytes
                    .checked_add(if bytes.is_empty() { 0 } else { 24 })
                    .ok_or(AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                (Object::Buffer(try_vec_from_slice(bytes)?), reservation)
            }
            Object::String(value) => {
                let heap_bytes = value
                    .len()
                    .checked_add(if value.is_empty() { 0 } else { 24 })
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                let mut copy = String::new();
                copy.try_reserve_exact(value.len()).map_err(|_| AmlError::BudgetExceeded)?;
                copy.push_str(value);
                (Object::String(copy), reservation)
            }
            Object::Package(elements) => {
                let package_bytes = elements
                    .len()
                    .checked_mul(mem::size_of::<WrappedObject>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| AmlError::BudgetExceeded)?;
                let heap_bytes = package_bytes
                    .checked_add(if elements.is_empty() { 0 } else { 24 })
                    .ok_or(AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, package_bytes, 0)?;
                (Object::Package(try_vec_from_slice(elements)?), reservation)
            }
            Object::Method { code, flags } => {
                let heap_bytes = code
                    .len()
                    .checked_add(if code.is_empty() { 0 } else { 24 })
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                (Object::Method { code: try_vec_from_slice(code)?, flags: *flags }, reservation)
            }
            Object::NamePath { name, scope } => {
                let heap_bytes = aml_name_clone_heap_bytes(name)?
                    .checked_add(aml_name_clone_heap_bytes(scope)?)
                    .ok_or_else(|| AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                (Object::NamePath { name: name.try_clone()?, scope: scope.try_clone()? }, reservation)
            }
            Object::OpRegion(region) => {
                let heap_bytes = aml_name_clone_heap_bytes(&region.parent_device_path)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                (
                    Object::OpRegion(OpRegion {
                        space: region.space,
                        base: region.base,
                        length: region.length,
                        parent_device_path: region.parent_device_path.try_clone()?,
                    }),
                    reservation,
                )
            }
            _ => (object.clone(), None),
        };
        Ok(MeteredObject { object, heap_reservation })
    }

    fn try_implicit_cast_copy(&self, target_type: ObjectType, source: &Object) -> Result<MeteredObject, AmlError> {
        let integer_bytes = match source {
            Object::Integer(value) => Some(value.to_le_bytes()),
            _ => None,
        };
        let source_bytes = match source {
            Object::Integer(_) => integer_bytes.as_ref().map_or(&[][..], |bytes| &bytes[..]),
            Object::String(value) => value.as_bytes(),
            Object::Buffer(bytes) => bytes.as_slice(),
            _ => {
                return Err(AmlError::InvalidImplicitCast { from: target_type, to: source.typ() });
            }
        };

        match target_type {
            ObjectType::Integer => {
                let mut bytes = [0_u8; 8];
                let length = usize::min(bytes.len(), source_bytes.len());
                bytes[..length].copy_from_slice(&source_bytes[..length]);
                Ok(MeteredObject { object: Object::Integer(u64::from_le_bytes(bytes)), heap_reservation: None })
            }
            ObjectType::Buffer => {
                let buffer_bytes = u64::try_from(source_bytes.len()).map_err(|_| AmlError::BudgetExceeded)?;
                let heap_bytes = buffer_bytes
                    .checked_add(if source_bytes.is_empty() { 0 } else { 24 })
                    .ok_or(AmlError::BudgetExceeded)?;
                let heap_reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                Ok(MeteredObject { object: Object::Buffer(try_vec_from_slice(source_bytes)?), heap_reservation })
            }
            ObjectType::String => {
                let output_capacity = source_bytes.len().checked_mul(3).ok_or_else(|| AmlError::BudgetExceeded)?;
                let allocation_bytes = output_capacity
                    .checked_add(if output_capacity == 0 { 0 } else { 24 })
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| AmlError::BudgetExceeded)?;
                let heap_reservation = self.reserve_object_heap(allocation_bytes, 0, 0)?;
                Ok(MeteredObject {
                    object: Object::String(try_lossy_string_prefix(source_bytes)?),
                    heap_reservation,
                })
            }
            _ => Err(AmlError::InvalidImplicitCast { from: target_type, to: source.typ() }),
        }
    }

    fn reserve_namespace_object(
        &self,
        name: &AmlName,
        payload_bytes: u64,
    ) -> Result<Option<Box<dyn ObjectHeapReservation>>, AmlError> {
        const NAMESPACE_OBJECT_OVERHEAD: u64 = 128;
        let name_bytes = name
            .encoded_len()
            .and_then(|length| u64::try_from(length).ok())
            .ok_or_else(|| AmlError::BudgetExceeded)?;
        let component_heap_bytes = name.heap_allocation_upper_bound().ok_or_else(|| AmlError::BudgetExceeded)?;
        let heap_bytes = NAMESPACE_OBJECT_OVERHEAD
            .checked_add(name_bytes)
            .and_then(|bytes| bytes.checked_add(component_heap_bytes))
            .and_then(|bytes| bytes.checked_add(payload_bytes))
            .ok_or_else(|| AmlError::BudgetExceeded)?;
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        if let Some(meter) = meter {
            meter.reserve_namespace_object(heap_bytes)?;
            Ok(Some(allocator::box_new(NamespaceObjectHeapReservation { meter, heap_bytes })))
        } else {
            Ok(None)
        }
    }

    fn resolve_name(&self, name: &AmlName, scope: &AmlName) -> Result<MeteredNameString, AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        match meter {
            Some(meter) => name.resolve_metered(scope, meter),
            None => Ok(MeteredNameString::from_owned_name(name.resolve(scope)?)),
        }
    }

    fn evaluate_if_present_metered(
        &self,
        path: MeteredNameString,
        args: Vec<WrappedObject>,
    ) -> Result<Option<WrappedObject>, AmlError> {
        let evaluation_path = path.try_clone_metered()?;
        let (evaluation_path, _evaluation_path_heap_reservation) = evaluation_path.into_parts();
        let object = self.namespace.lock().get_absolute_if_present(&evaluation_path)?;
        match object {
            Some(object) => self.evaluate_object(evaluation_path, object, args).map(Some),
            None => Ok(None),
        }
    }

    #[doc(hidden)]
    pub fn evaluate_if_present_at_scope(
        &self,
        name: &str,
        scope: &AmlName,
        args: Vec<WrappedObject>,
    ) -> Result<Option<WrappedObject>, AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let name = MeteredNameString::parse_name_metered(name, meter)?;
        let path = self.resolve_name(&name, scope)?;
        self.evaluate_if_present_metered(path, args)
    }

    fn insert_metered_name(
        &self,
        path: MeteredNameString,
        object: WrappedObject,
        namespace_reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> Result<(), AmlError> {
        let (path, _path_heap_reservation) = path.into_parts();
        let token = self.object_token.lock();
        object.commit_heap_reservation(&token)?;
        self.namespace.lock().insert_if_mutable_with_reservation(path, object, namespace_reservation).map(|_| ())
    }

    fn insert_metered_name_if_mutable(
        &self,
        path: MeteredNameString,
        object: WrappedObject,
        namespace_reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> Result<bool, AmlError> {
        let (path, _path_heap_reservation) = path.into_parts();
        let token = self.object_token.lock();
        object.commit_heap_reservation(&token)?;
        self.namespace.lock().insert_if_mutable_with_reservation(path, object, namespace_reservation)
    }

    fn search_name(
        &self,
        name: &AmlName,
        scope: &AmlName,
    ) -> Result<(MeteredNameString, WrappedObject), AmlError> {
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let namespace = self.namespace.lock();
        match meter {
            Some(meter) => {
                let (resolved, object, _) = namespace.search_with_flags_metered(name, scope, meter)?;
                Ok((resolved, object))
            }
            None => {
                let (resolved, object, _) = namespace.search_with_flags(name, scope)?;
                Ok((MeteredNameString::from_owned_name(resolved), object))
            }
        }
    }

    /// Initialize the namespace - this should be called after all tables have been loaded and
    /// operation region handlers registered. Specifically, it will call relevant `_STA`, `_INI`,
    /// and `_REG` methods.
    pub fn initialize_namespace(&self) {
        /*
         * This should match the initialization order of ACPICA and uACPI.
         */
        if let Err(err) = self.evaluate_if_present(AmlName::from_str("\\_INI").unwrap(), crate::aml_vec![]) {
            warn!("Invoking \\_INI failed: {:?}", err);
        }
        if let Err(err) = self.evaluate_if_present(AmlName::from_str("\\_SB._INI").unwrap(), crate::aml_vec![]) {
            warn!("Invoking \\_SB._INI failed: {:?}", err);
        }

        if let Err(err) = self.initialize_native_regions() {
            warn!("Invoking native OperationRegion _REG activation failed: {:?}", err);
        }

        /*
         * We can now initialize each device in the namespace. For each device, we evaluate `_STA`,
         * which indicates if the device is present and functional. If this method does not exist,
         * we assume the device should be initialized.
         *
         * We then evaluate `_INI` for the device. This can dynamically populate objects such as
         * `_ADR`, `_CID`, `_HID`, `_SUN`, and `_UID`, and so is necessary before further
         * operation.
         */
        let mut num_devices_initialized = 0;
        let execution_meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let (device_candidates, _candidate_heap_reservation) = {
            let namespace = self.namespace.lock();
            let candidate_bound = match execution_meter.as_deref() {
                Some(meter) => namespace.device_like_candidate_heap_upper_bound_metered(meter).ok(),
                None => namespace.device_like_candidate_heap_upper_bound(),
            };
            let Some((candidate_count, candidate_heap_bytes)) = candidate_bound else {
                warn!("Unable to bound namespace initialization candidates");
                return;
            };
            let mut candidate_heap_reservation = NameStringHeapReservation::new(execution_meter.clone());
            if let Err(error) = candidate_heap_reservation.reserve_bytes(candidate_heap_bytes) {
                warn!("Unable to reserve namespace initialization candidates: {:?}", error);
                return;
            }
            let mut candidates = allocator::vec_new();
            if candidates.try_reserve_exact(candidate_count).is_err() {
                warn!("Unable to allocate namespace initialization candidates");
                return;
            }
            let snapshot_result =
                traverse_namespace_with_meter(&namespace, execution_meter.as_ref(), |path, level| {
                    if matches!(
                        level.kind,
                        NamespaceLevelKind::Device
                            | NamespaceLevelKind::Processor
                            | NamespaceLevelKind::ThermalZone
                            | NamespaceLevelKind::PowerResource
                    ) {
                        candidates.push((path.try_clone()?, level.kind));
                    }
                    Ok(true)
                });
            if let Err(error) = snapshot_result {
                warn!("Unable to snapshot namespace initialization candidates: {:?}", error);
                return;
            }
            (candidates, candidate_heap_reservation)
        };
        let mut skipped_subtree: Option<&AmlName> = None;
        for (path, _) in &device_candidates {
            if let Some(meter) = execution_meter.as_deref()
                && let Err(error) = meter.loop_iteration()
            {
                warn!("Unable to continue namespace initialization candidates: {:?}", error);
                return;
            }
            if let Some(parent) = skipped_subtree {
                if path.is_descendant_of(parent) {
                    continue;
                }
                skipped_subtree = None;
            }

            let should_initialize = match self
                .resolve_name(&AmlName::from_str("_STA").unwrap(), path)
                .and_then(|sta_path| self.evaluate_if_present_metered(sta_path, crate::aml_vec![]))
            {
                Ok(Some(result)) => match result.as_integer() {
                    Ok(result) => {
                        let status = DeviceStatus(result);
                        status.present() && status.functioning()
                    }
                    Err(error) => {
                        warn!("Invalid _STA result for {}: {:?}", path, error);
                        false
                    }
                },
                Ok(None) => true,
                Err(err) => {
                    warn!("Failed to evaluate _STA for device {}: {:?}", path, err);
                    false
                }
            };

            if !should_initialize {
                skipped_subtree = Some(path);
                continue;
            }

            num_devices_initialized += 1;
            if let Err(err) = self
                .resolve_name(&AmlName::from_str("_INI").unwrap(), path)
                .and_then(|ini_path| self.evaluate_if_present_metered(ini_path, crate::aml_vec![]))
            {
                warn!("Failed to evaluate _INI for device {}: {:?}", path, err);
            }
        }

        info!("Initialized {} devices", num_devices_initialized);
    }

    /// Activate the built-in native OperationRegion spaces after all AML
    /// tables have been loaded.  `_REG` is a namespace-side lifecycle hook,
    /// not a hardware authority: the actual region access still enters the
    /// installed capability-aware native handler on every read/write.
    ///
    /// Firmware commonly declares several regions in one device.  ACPI
    /// connects a handler per `(scope, address space)`, so deduplicate those
    /// pairs and invoke `_REG(space, 1)` exactly once for each supported
    /// scope.  Unsupported address spaces remain unregistered and therefore
    /// fail closed when an AML method attempts to use them.
    fn rollback_native_region_registrations(&self, registrations: &[(AmlName, RegionSpace)]) -> Option<AmlError> {
        let mut first_error = None;
        for (scope, space) in registrations.iter().rev() {
            if let Err(error) = self.evaluate_native_region_registration(scope, *space, false) {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        first_error
    }

    pub fn initialize_native_regions(&self) -> Result<(), AmlError> {
        if self.native_region_registration_active.load(Ordering::Acquire) {
            return Ok(());
        }

        let execution_meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let (registration_capacity, path_clone_bytes, maximum_path_clone_bytes) = {
            let namespace = self.namespace.lock();
            let mut registration_capacity = 0_usize;
            let mut path_clone_bytes = 0_u64;
            let mut maximum_path_clone_bytes = 0_u64;
            traverse_namespace_with_meter(&namespace, execution_meter.as_ref(), |_, level| {
                for (_, object) in level.values.values() {
                    let Object::OpRegion(region) = &**object else {
                        continue;
                    };
                    if !matches!(
                        region.space,
                        RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig
                    ) {
                        continue;
                    }
                    registration_capacity =
                        registration_capacity.checked_add(1).ok_or_else(|| AmlError::BudgetExceeded)?;
                    let bytes = region
                        .parent_device_path
                        .clone_heap_upper_bound()
                        .ok_or_else(|| AmlError::BudgetExceeded)?;
                    path_clone_bytes =
                        path_clone_bytes.checked_add(bytes).ok_or_else(|| AmlError::BudgetExceeded)?;
                    maximum_path_clone_bytes = maximum_path_clone_bytes.max(bytes);
                }
                Ok(true)
            })?;
            (registration_capacity, path_clone_bytes, maximum_path_clone_bytes)
        };
        let (working_heap_bytes, persistent_heap_bytes) = native_region_registration_heap_bytes(
            registration_capacity,
            path_clone_bytes,
            maximum_path_clone_bytes,
        )
        .ok_or_else(|| AmlError::BudgetExceeded)?;
        let mut working_heap_reservation = NameStringHeapReservation::new(execution_meter.clone());
        working_heap_reservation.reserve_bytes(working_heap_bytes)?;

        let mut registrations: Vec<(AmlName, RegionSpace)> = allocator::vec_new();
        registrations.try_reserve_exact(registration_capacity).map_err(|_| AmlError::BudgetExceeded)?;
        let mut observed_region_count = 0_usize;
        let mut observed_path_clone_bytes = 0_u64;
        let namespace = self.namespace.lock();
        traverse_namespace_with_meter(&namespace, execution_meter.as_ref(), |_, level| {
            for (_, object) in level.values.values() {
                let Object::OpRegion(region) = &**object else {
                    continue;
                };
                if !matches!(
                    region.space,
                    RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig
                ) {
                    continue;
                }
                observed_region_count =
                    observed_region_count.checked_add(1).ok_or_else(|| AmlError::BudgetExceeded)?;
                let path_bytes =
                    region.parent_device_path.clone_heap_upper_bound().ok_or_else(|| AmlError::BudgetExceeded)?;
                observed_path_clone_bytes =
                    observed_path_clone_bytes.checked_add(path_bytes).ok_or_else(|| AmlError::BudgetExceeded)?;
                if observed_region_count > registration_capacity
                    || observed_path_clone_bytes > path_clone_bytes
                    || path_bytes > maximum_path_clone_bytes
                {
                    return Err(AmlError::BudgetExceeded);
                }
                if !registrations
                    .iter()
                    .any(|(scope, space)| *scope == region.parent_device_path && *space == region.space)
                {
                    if registrations.len() == registration_capacity {
                        return Err(AmlError::BudgetExceeded);
                    }
                    registrations.push((region.parent_device_path.try_clone()?, region.space));
                }
            }
            Ok(true)
        })?;
        drop(namespace);

        let registrations = allocator::arc_new(registrations);
        self.native_region_registrations.lock().install(Arc::clone(&registrations))?;
        let registration_count = registrations.len();
        let activation = (|| {
            for index in 0..registration_count {
                self.native_region_registrations.lock().mark_connected(index + 1)?;
                let (scope, space) = &registrations[index];
                self.evaluate_native_region_registration(scope, *space, true)?;
            }
            if execution_meter.is_some() {
                working_heap_reservation.commit_bytes_to_persistent(persistent_heap_bytes)?;
            }
            Ok(())
        })();
        if let Err(error) = activation {
            let attempted_count = self.native_region_registrations.lock().connected_count();
            self.native_region_registrations.lock().mark_connected(0)?;
            let mut cleanup_error = self.invalidate_native_region_tokens().err();
            if let Some(rollback_error) =
                self.rollback_native_region_registrations(&registrations[..attempted_count])
            {
                if cleanup_error.is_none() {
                    cleanup_error = Some(rollback_error);
                }
            }
            let (retired, retired_dynamic, reserved_heap_bytes) = self.native_region_registrations.lock().clear();
            drop(retired);
            drop(retired_dynamic);
            drop(registrations);
            if reserved_heap_bytes != 0 {
                if let Some(meter) = execution_meter.as_ref() {
                    if let Err(release_error) = meter.release_heap(reserved_heap_bytes) {
                        if cleanup_error.is_none() {
                            cleanup_error = Some(release_error);
                        }
                    }
                }
            }
            self.native_region_registration_active.store(false, Ordering::Release);
            if let Some(cleanup_error) = cleanup_error {
                return Err(AmlError::InternalError(format!(
                    "_REG rollback/token invalidation failed after {error:?}: {cleanup_error:?}"
                )));
            }
            return Err(error);
        }
        self.native_region_registrations.lock().set_reserved_heap_bytes(if execution_meter.is_some() {
            persistent_heap_bytes
        } else {
            0
        });
        self.native_region_registration_active.store(true, Ordering::Release);
        Ok(())
    }

    /// Deactivate the same built-in region registrations that were admitted
    /// by `initialize_native_regions`.  This is an explicit lifecycle edge:
    /// callers must not silently replay AML after a failed recovery, but a
    /// controlled teardown can still notify firmware with `_REG(..., 0)`.
    pub fn deinitialize_native_regions(&self) -> Result<(), AmlError> {
        let was_active = self.native_region_registration_active.load(Ordering::Acquire);
        let execution_meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let (registrations, connected_count, dynamic_registrations, maximum_path_clone_bytes) = {
            let registration_state = self.native_region_registrations.lock();
            let (registrations, connected_count) = registration_state.snapshot();
            let dynamic_registrations = registration_state.dynamic_snapshot();
            let maximum_path_clone_bytes = if connected_count == 0 && dynamic_registrations.is_none() {
                0
            } else {
                registration_state.maximum_path_clone_heap_upper_bound().ok_or_else(|| AmlError::BudgetExceeded)?
            };
            (registrations, connected_count, dynamic_registrations, maximum_path_clone_bytes)
        };
        if was_active && registrations.as_ref().is_none_or(|entries| connected_count != entries.len()) {
            return Err(AmlError::InternalError("active native region registrations are incomplete".to_string()));
        }
        let working_heap_bytes = if was_active {
            native_region_callback_scratch_heap_bytes(maximum_path_clone_bytes)
                .ok_or_else(|| AmlError::BudgetExceeded)?
        } else {
            0
        };
        let mut _working_heap_reservation = NameStringHeapReservation::new(execution_meter.clone());
        _working_heap_reservation.reserve_bytes(working_heap_bytes)?;
        self.native_region_registration_active.store(false, Ordering::Release);
        self.native_region_registrations.lock().disconnect_all();
        let mut first_error = self.invalidate_native_region_tokens().err();
        if let Some(registrations) = dynamic_registrations.as_deref() {
            for (scope, space) in registrations.iter().rev() {
                if let Err(error) = self.evaluate_native_region_registration(scope, *space, false) {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        if let Some(registrations) = registrations.as_deref() {
            for (scope, space) in registrations.iter().take(connected_count).rev() {
                if let Err(error) = self.evaluate_native_region_registration(scope, *space, false) {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        drop(dynamic_registrations);
        drop(registrations);
        let (retired, retired_dynamic, reserved_heap_bytes) = self.native_region_registrations.lock().clear();
        drop(retired);
        drop(retired_dynamic);
        if reserved_heap_bytes != 0 {
            if let Some(meter) = execution_meter.as_ref() {
                if let Err(error) = meter.release_heap(reserved_heap_bytes) {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn evaluate_native_region_registration(
        &self,
        scope: &AmlName,
        space: RegionSpace,
        connected: bool,
    ) -> Result<(), AmlError> {
        let space_number = match space {
            RegionSpace::SystemMemory => 0,
            RegionSpace::SystemIO => 1,
            RegionSpace::PciConfig => 2,
            RegionSpace::EmbeddedControl => 3,
            RegionSpace::SmBus => 4,
            RegionSpace::SystemCmos => 5,
            RegionSpace::PciBarTarget => 6,
            RegionSpace::Ipmi => 7,
            RegionSpace::GeneralPurposeIo => 8,
            RegionSpace::GenericSerialBus => 9,
            RegionSpace::Pcc => 10,
            RegionSpace::Oem(value) => u64::from(value),
        };
        let path = scope.try_resolve_child(NameSeg::from_bytes(*b"_REG")?)?;
        let connection_status = if connected { 1 } else { 0 };
        let mut args = allocator::vec_new();
        args.try_reserve_exact(2).map_err(|_| AmlError::BudgetExceeded)?;
        args.push(self.wrap_object(Object::Integer(space_number))?);
        args.push(self.wrap_object(Object::Integer(connection_status))?);
        self.evaluate_if_present(path, args).map(|_| ())
    }

    fn invalidate_native_region_tokens(&self) -> Result<(), AmlError> {
        let handler = self.native_region_handler.lock().as_ref().map(Arc::clone);
        if let Some(handler) = handler {
            handler.invalidate_all()?;
        }
        Ok(())
    }

    fn prepare_dynamic_native_region_redefinition(&self, space: RegionSpace) -> Result<(), AmlError> {
        if matches!(space, RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig)
            && self.native_region_registration_active.load(Ordering::Acquire)
        {
            self.invalidate_native_region_tokens()?;
        }
        Ok(())
    }

    fn register_dynamic_native_region(&self, scope: &AmlName, space: RegionSpace) -> Result<(), AmlError> {
        if !matches!(space, RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig)
            || !self.native_region_registration_active.load(Ordering::Acquire)
        {
            return Ok(());
        }
        if self
            .native_region_dynamic_registration_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(AmlError::HostError("NOT_SUPPORTED_REGION_REGISTRATION_REENTRY".to_string()));
        }
        let result = self.register_dynamic_native_region_inner(scope, space);
        self.native_region_dynamic_registration_active.store(false, Ordering::Release);
        result
    }

    fn register_dynamic_native_region_inner(&self, scope: &AmlName, space: RegionSpace) -> Result<(), AmlError> {
        if self.native_region_handler.lock().is_none() {
            return Err(AmlError::NoHandlerForRegionAccess(space));
        }

        let old_registrations = {
            let registrations = self.native_region_registrations.lock();
            if registrations.contains_registration(scope, space) {
                return Ok(());
            }
            registrations.dynamic_snapshot()
        };
        let new_count = old_registrations
            .as_ref()
            .map_or(Some(1), |entries| entries.len().checked_add(1))
            .ok_or(AmlError::BudgetExceeded)?;
        let mut path_clone_bytes = 0_u64;
        let mut maximum_path_clone_bytes = 0_u64;
        if let Some(entries) = old_registrations.as_deref() {
            for (existing_scope, _) in entries {
                let bytes = existing_scope.clone_heap_upper_bound().ok_or(AmlError::BudgetExceeded)?;
                path_clone_bytes = path_clone_bytes.checked_add(bytes).ok_or(AmlError::BudgetExceeded)?;
                maximum_path_clone_bytes = maximum_path_clone_bytes.max(bytes);
            }
        }
        let new_path_clone_bytes = scope.clone_heap_upper_bound().ok_or(AmlError::BudgetExceeded)?;
        path_clone_bytes = path_clone_bytes.checked_add(new_path_clone_bytes).ok_or(AmlError::BudgetExceeded)?;
        maximum_path_clone_bytes = maximum_path_clone_bytes.max(new_path_clone_bytes);
        let (working_heap_bytes, persistent_heap_bytes) =
            native_region_registration_heap_bytes(new_count, path_clone_bytes, maximum_path_clone_bytes)
                .ok_or(AmlError::BudgetExceeded)?;
        let execution_meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        let accounted_persistent_heap_bytes = if execution_meter.is_some() { persistent_heap_bytes } else { 0 };
        let mut working_heap_reservation = NameStringHeapReservation::new(execution_meter.clone());
        working_heap_reservation.reserve_bytes(working_heap_bytes)?;

        let mut new_registrations = allocator::vec_new();
        new_registrations.try_reserve_exact(new_count).map_err(|_| AmlError::BudgetExceeded)?;
        if let Some(entries) = old_registrations.as_deref() {
            for (existing_scope, existing_space) in entries {
                new_registrations.push((existing_scope.try_clone()?, *existing_space));
            }
        }
        new_registrations.push((scope.try_clone()?, space));
        drop(old_registrations);

        let new_registrations = Some(allocator::arc_new(new_registrations));
        let (old_registrations, old_heap_bytes) = self
            .native_region_registrations
            .lock()
            .replace_dynamic(new_registrations, accounted_persistent_heap_bytes)?;

        let mut activation_error = self.evaluate_native_region_registration(scope, space, true).err();
        let mut persistent_heap_committed = false;
        if activation_error.is_none() && execution_meter.is_some() {
            if let Err(error) = working_heap_reservation.commit_bytes_to_persistent(persistent_heap_bytes) {
                activation_error = Some(error);
            } else {
                persistent_heap_committed = true;
            }
        }
        if activation_error.is_none() {
            activation_error = self.invalidate_native_region_tokens().err();
        }
        if let Some(error) = activation_error {
            let (failed_registrations, _) = self
                .native_region_registrations
                .lock()
                .replace_dynamic(old_registrations.clone(), old_heap_bytes)?;
            drop(failed_registrations);
            drop(old_registrations);
            let mut cleanup_error = self.invalidate_native_region_tokens().err();
            if let Err(disconnect_error) = self.evaluate_native_region_registration(scope, space, false) {
                if cleanup_error.is_none() {
                    cleanup_error = Some(disconnect_error);
                }
            }
            if persistent_heap_committed {
                if let Some(meter) = execution_meter.as_ref() {
                    if let Err(release_error) = meter.release_heap(accounted_persistent_heap_bytes) {
                        if cleanup_error.is_none() {
                            cleanup_error = Some(release_error);
                        }
                    }
                }
            }
            if let Some(cleanup_error) = cleanup_error {
                return Err(AmlError::InternalError(format!(
                    "dynamic _REG rollback/token invalidation failed after {error:?}: {cleanup_error:?}"
                )));
            }
            return Err(error);
        }

        drop(old_registrations);
        if old_heap_bytes != 0 {
            if let Some(meter) = execution_meter.as_ref() {
                meter.release_heap(old_heap_bytes)?;
            }
        }
        Ok(())
    }

    fn native_region_is_connected(&self, region: &OpRegion) -> bool {
        matches!(region.space, RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig)
            && self.native_region_registrations.lock().contains(&region.parent_device_path, region.space)
    }

    fn native_execution_thread_id(&self) -> Option<u64> {
        self.native_region_handler
            .lock()
            .as_ref()
            .and_then(|handler| handler.execution_thread_id())
            .filter(|thread_id| *thread_id != 0)
    }

    fn global_lock_depth_for_thread(&self, thread_id: u64) -> u32 {
        self.global_lock_owner
            .lock()
            .as_ref()
            .filter(|(owner_thread_id, _)| *owner_thread_id == thread_id)
            .map_or(0, |(_, depth)| *depth)
    }

    fn unwind_global_lock_depth(&self, thread_id: u64, entry_depth: u32) -> Result<(), AmlError> {
        loop {
            let depth = self.global_lock_depth_for_thread(thread_id);
            if depth == entry_depth {
                return Ok(());
            }
            if depth < entry_depth {
                self.global_lock_poisoned.store(true, Ordering::Release);
                return Err(AmlError::GlobalLockOutcomeUnknown);
            }
            self.release_global_lock()?;
        }
    }

    fn is_known_global_lock_acquire_failure(error: &AmlError) -> bool {
        matches!(error, AmlError::MutexAcquireTimeout)
            || matches!(error, AmlError::HostError(message) if message == "NOT_SUPPORTED_GLOBAL_LOCK")
    }

    pub fn acquire_global_lock(&self, timeout: u16) -> Result<(), AmlError> {
        self.global_lock_dependency.store(true, Ordering::Release);
        if self.global_lock_poisoned.load(Ordering::Acquire) {
            return Err(AmlError::GlobalLockOutcomeUnknown);
        }
        let handler = self
            .native_region_handler
            .lock()
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| AmlError::HostError("NOT_SUPPORTED_GLOBAL_LOCK".to_string()))?;
        let thread_id = handler
            .execution_thread_id()
            .filter(|thread_id| *thread_id != 0)
            .ok_or_else(|| AmlError::HostError("NOT_SUPPORTED_GLOBAL_LOCK".to_string()))?;
        {
            let mut owner = self.global_lock_owner.lock();
            if let Some((owner_thread_id, depth)) = owner.as_mut() {
                if *owner_thread_id == thread_id {
                    *depth = depth.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
                    return Ok(());
                }
            }
        }
        let mut timeout_us = if timeout == u16::MAX { u64::MAX } else { u64::from(timeout) * 1_000 };
        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        if let Some(meter) = meter.as_ref() {
            if let Some(remaining_ns) = meter.remaining_deadline_ns()? {
                timeout_us = timeout_us.min(remaining_ns / 1_000);
            }
        }
        let acquire_result = handler.acquire_global_lock(timeout_us);
        if acquire_result.as_ref().is_err_and(|error| !Self::is_known_global_lock_acquire_failure(error)) {
            self.global_lock_poisoned.store(true, Ordering::Release);
        }
        if let Some(meter) = meter.as_ref() {
            if let Err(deadline_error) = meter.check_deadline() {
                if acquire_result.is_ok() {
                    if handler.release_global_lock().is_err() {
                        self.global_lock_poisoned.store(true, Ordering::Release);
                        return Err(AmlError::GlobalLockOutcomeUnknown);
                    }
                }
                return Err(deadline_error);
            }
        }
        match acquire_result {
            Ok(()) => {}
            Err(AmlError::MutexAcquireTimeout) => return Err(AmlError::MutexAcquireTimeout),
            Err(AmlError::HostError(message)) if message == "NOT_SUPPORTED_GLOBAL_LOCK" => {
                return Err(AmlError::HostError(message));
            }
            Err(_) => {
                self.global_lock_poisoned.store(true, Ordering::Release);
                return Err(AmlError::GlobalLockOutcomeUnknown);
            }
        }
        let mut owner = self.global_lock_owner.lock();
        if owner.is_some() {
            drop(owner);
            if handler.release_global_lock().is_err() {
                self.global_lock_poisoned.store(true, Ordering::Release);
                return Err(AmlError::GlobalLockOutcomeUnknown);
            }
            return Err(AmlError::InternalError("global lock owner changed during acquisition".to_string()));
        }
        *owner = Some((thread_id, 1));
        Ok(())
    }

    pub fn release_global_lock(&self) -> Result<(), AmlError> {
        self.global_lock_dependency.store(true, Ordering::Release);
        let handler = self
            .native_region_handler
            .lock()
            .as_ref()
            .map(Arc::clone)
            .ok_or(AmlError::GlobalLockOutcomeUnknown)?;
        let thread_id = handler
            .execution_thread_id()
            .filter(|thread_id| *thread_id != 0)
            .ok_or(AmlError::GlobalLockOutcomeUnknown)?;
        {
            let mut owner = self.global_lock_owner.lock();
            match owner.as_mut() {
                Some((owner_thread_id, depth)) if *owner_thread_id == thread_id && *depth > 1 => {
                    *depth -= 1;
                    return Ok(());
                }
                Some((owner_thread_id, _)) if *owner_thread_id == thread_id => *owner = None,
                _ => return Err(AmlError::HostError("NOT_ACQUIRED_GLOBAL_LOCK".to_string())),
            }
        }
        if handler.release_global_lock().is_err() {
            self.global_lock_poisoned.store(true, Ordering::Release);
            return Err(AmlError::GlobalLockOutcomeUnknown);
        }
        Ok(())
    }

    fn do_execute_method(&self, context: MethodContext) -> Result<WrappedObject, AmlError> {
        if matches!(
            &context.current_block.kind,
            BlockKind::Method { method_scope } if method_scope.is_rooted_single_segment(*b"_PIC")
        ) && self.pic_evaluation_state.compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire).is_err()
        {
            return Err(AmlError::HostError("NOT_SUPPORTED_PIC_REENTRY".to_string()));
        }
        let thread_id = self.native_execution_thread_id();
        let entry_lock_depth = thread_id.map_or(0, |thread_id| self.global_lock_depth_for_thread(thread_id));
        let mut meter = MethodMeterGuard::new(self);
        meter.enter()?;
        let result = self.do_execute_method_loop(context, &mut meter);
        let cleanup =
            thread_id.map_or(Ok(()), |thread_id| self.unwind_global_lock_depth(thread_id, entry_lock_depth));
        match (result, cleanup) {
            (result, Ok(())) => result,
            (_, Err(_)) => Err(AmlError::GlobalLockOutcomeUnknown),
        }
    }

    fn do_execute_method_loop(
        &self,
        mut context: MethodContext,
        meter: &mut MethodMeterGuard<'_, H, R>,
    ) -> Result<WrappedObject, AmlError> {
        /*
         * This is the main loop that executes operations. Every op is handled at the top-level of
         * the loop to prevent pathological stack growth from nested operations.
         *
         * The loop has three main stages:
         *   1) Check if any in-flight operations are ready to be executed (i.e. have collected all
         *      their arguments). An operation completing may contribute the last required argument
         *      of the one above, so this is repeated for as many operations as are ready to be
         *      retired.
         *   2) Look at the next opcode in the stream. If we've run out of opcodes in the current
         *      block, run logic to determine where in the stream we should move to next. Special
         *      logic at this level handles things like moving in/out of package definitions, and
         *      performing control flow.
         *   3) When the next opcode is determined, use it to interpret the next portion of the
         *      stream. If that is data, the correct number of bytes can be consumed and
         *      contributed to the current in-flight operation. If it's an opcode, a new in-flight
         *      operation is started, and we go round the loop again.
         *
         * This scheme is what allows the interpreter to use a loop that somewhat resembles a
         * traditional fast bytecode VM, but also provides enough flexibility to handle the
         * quirkier parts of the AML grammar, particularly the left-to-right encoding of operands.
         */
        let mut context_stack: Vec<MethodContext> = allocator::vec_new();
        let mut context_stack_heap_reservation =
            NameStringHeapReservation::new(self.execution_meter.lock().as_ref().map(Arc::clone));

        loop {
            if let Some(error) = context.take_allocation_error() {
                return Err(error);
            }
            self.charge_opcode()?;
            /*
             * First, see if we've gathered enough arguments to complete some in-flight operations.
             */
            while let Some(op) = context.in_flight.pop_if(|op| op.arguments.len() == op.expected_arguments) {
                if let Some(error) = context.take_allocation_error() {
                    return Err(error);
                }
                match op.op {
                    Opcode::Add
                    | Opcode::Subtract
                    | Opcode::Multiply
                    | Opcode::Divide
                    | Opcode::ShiftLeft
                    | Opcode::ShiftRight
                    | Opcode::Mod
                    | Opcode::Nand
                    | Opcode::And
                    | Opcode::Or
                    | Opcode::Nor
                    | Opcode::Xor => self.do_binary_maths(&mut context, op)?,
                    Opcode::Not | Opcode::FindSetLeftBit | Opcode::FindSetRightBit => {
                        self.do_unary_maths(&mut context, op)?;
                    }
                    Opcode::Increment | Opcode::Decrement => {
                        let [Argument::Object(operand)] = &op.arguments[..] else {
                            return Err(AmlError::InternalError(
                                "increment/decrement operand was not an object".to_string(),
                            ));
                        };
                        let operand = operand.clone().unwrap_reference()?;
                        let token = self.object_token.lock();

                        let Object::Integer(operand) = (unsafe { operand.gain_mut(&token) }) else {
                            Err(AmlError::ObjectNotOfExpectedType {
                                expected: ObjectType::Integer,
                                got: operand.typ(),
                            })?
                        };

                        let new_value = match op.op {
                            Opcode::Increment => operand.wrapping_add(1),
                            Opcode::Decrement => operand.wrapping_sub(1),
                            _ => unreachable!(),
                        };

                        *operand = new_value;
                        context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(new_value))?));
                        context.retire_op(op);
                    }
                    Opcode::LAnd
                    | Opcode::LOr
                    | Opcode::LNot
                    | Opcode::LNotEqual
                    | Opcode::LLessEqual
                    | Opcode::LGreaterEqual
                    | Opcode::LEqual
                    | Opcode::LGreater
                    | Opcode::LLess => self.do_logical_op(&mut context, op)?,
                    Opcode::ToBuffer => self.do_to_buffer(&mut context, op)?,
                    Opcode::ToInteger => self.do_to_integer(&mut context, op)?,
                    Opcode::ToString => self.do_to_string(&mut context, op)?,
                    Opcode::ToDecimalString | Opcode::ToHexString => {
                        self.do_to_dec_hex_string(&mut context, op)?
                    }
                    Opcode::Mid => self.do_mid(&mut context, op)?,
                    Opcode::Concat => self.do_concat(&mut context, op)?,
                    Opcode::ConcatRes => {
                        extract_args!(op => [
                            Argument::Object(source1),
                            Argument::Object(source2),
                            Argument::Object(target)
                        ]);
                        let source1 = source1.as_buffer()?;
                        let source2 = source2.as_buffer()?;
                        let result = {
                            fn strip_end_tag(buf: &[u8]) -> &[u8] {
                                if buf.len() >= 2 && buf[buf.len() - 2] == 0x79 {
                                    buf.split_at(buf.len() - 2).0
                                } else {
                                    buf
                                }
                            }
                            let source1 = strip_end_tag(source1);
                            let source2 = strip_end_tag(source2);
                            let result_bytes = source1
                                .len()
                                .checked_add(source2.len())
                                .and_then(|bytes| bytes.checked_add(2))
                                .ok_or(AmlError::HostError("AML_BUFFER_SIZE_OVERFLOW".to_string()))?;
                            let buffer_bytes =
                                u64::try_from(result_bytes).map_err(|_| AmlError::BudgetExceeded)?;
                            let heap_bytes = buffer_bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?;
                            let reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                            let mut buffer = try_vec_with_capacity(result_bytes)?;
                            buffer.extend_from_slice(source1);
                            buffer.extend_from_slice(source2);
                            // Add a new end-tag
                            buffer.push(0x79);
                            // Don't calculate the new real checksum - just use 0
                            buffer.push(0x00);
                            self.wrap_object_with_heap_reservation(Object::Buffer(buffer), reservation)?
                        };
                        // TODO: use potentially-updated result for return value here
                        self.do_store(target.clone(), result.clone())?;
                        context.contribute_arg(Argument::Object(result));
                        context.retire_op(op);
                    }
                    Opcode::Reset => {
                        extract_args!(op => [Argument::Object(sync_object)]);
                        let sync_object = sync_object.clone().unwrap_reference()?;

                        if let Object::Event(ref counter) = *sync_object {
                            counter.store(0, Ordering::Release);
                        } else {
                            return Err(AmlError::InvalidOperationOnObject {
                                op: Operation::ResetEvent,
                                typ: sync_object.typ(),
                            });
                        }
                    }
                    Opcode::Signal => {
                        extract_args!(op => [Argument::Object(sync_object)]);
                        let sync_object = sync_object.clone().unwrap_reference()?;

                        if let Object::Event(ref counter) = *sync_object {
                            counter.fetch_add(1, Ordering::AcqRel);
                        } else {
                            return Err(AmlError::InvalidOperationOnObject {
                                op: Operation::SignalEvent,
                                typ: sync_object.typ(),
                            });
                        }
                    }
                    Opcode::Wait => {
                        extract_args!(op => [Argument::Object(sync_object), Argument::Object(timeout)]);
                        let sync_object = sync_object.clone().unwrap_reference()?;
                        let timeout = u64::min(timeout.as_integer()?, 0xffff);

                        if let Object::Event(ref counter) = *sync_object {
                            /*
                             * `Wait` returns a non-zero value if a timeout occurs and the event
                             * was not signaled, and zero if it was. Timeout is specified in
                             * milliseconds, should relinquish processor control (we use
                             * `Handler::sleep` to do so) and a value of `0xffff` specifies that
                             * the operation should wait indefinitely.
                             */
                            let mut remaining_sleep = timeout;
                            let mut timed_out = true;

                            'signaled: while remaining_sleep > 0 {
                                loop {
                                    /*
                                     * Try to decrement the counter. If it's zero after a load, we
                                     * haven't been signalled and should wait for a bit. If it's
                                     * non-zero, we were signalled and should stop waiting.
                                     */
                                    let value = counter.load(Ordering::Acquire);
                                    if value == 0 {
                                        break;
                                    }
                                    if counter
                                        .compare_exchange(value, value - 1, Ordering::AcqRel, Ordering::Acquire)
                                        .is_ok()
                                    {
                                        timed_out = false;
                                        break 'signaled;
                                    }
                                }

                                let to_sleep = u64::min(timeout, 10);
                                if timeout < 0xffff {
                                    remaining_sleep = remaining_sleep.saturating_sub(to_sleep);
                                }
                                self.handler.sleep(to_sleep);
                            }

                            context.contribute_arg(Argument::Object(
                                self.wrap_object(Object::Integer(if timed_out { u64::MAX } else { 0 }))?,
                            ));
                        } else {
                            return Err(AmlError::InvalidOperationOnObject {
                                op: Operation::WaitEvent,
                                typ: sync_object.typ(),
                            });
                        }
                    }
                    Opcode::Notify => {
                        // TODO: may need special handling on the node to get path?
                        extract_args!(op => [Argument::Namestring(name), Argument::Object(value)]);
                        let value = value.as_integer()?;

                        info!("Notify {:?} with value {}", name, value);
                        // TODO: support
                        return Err(AmlError::LibUnimplemented);
                    }
                    Opcode::FromBCD => self.do_from_bcd(&mut context, op)?,
                    Opcode::ToBCD => self.do_to_bcd(&mut context, op)?,
                    Opcode::Name => {
                        extract_args!(op => [Argument::Namestring(name), Argument::Object(object)]);
                        let name = context.resolve_name(&name, &context.current_scope)?;
                        let reservation = self.reserve_namespace_object(&name, 0)?;
                        self.insert_metered_name(name, object.clone(), reservation)?;
                        context.retire_op(op);
                    }
                    Opcode::Fatal => {
                        extract_args!(op => [Argument::ByteData(typ), Argument::DWordData(code), Argument::Object(arg)]);
                        let arg = arg.as_integer()?;
                        self.handler.handle_fatal_error(*typ, *code, arg);
                        context.retire_op(op);
                        return Err(AmlError::FatalErrorEncountered);
                    }
                    Opcode::OpRegion => {
                        extract_args!(op => [
                            Argument::Namestring(name),
                            Argument::ByteData(region_space),
                            Argument::Object(region_offset),
                            Argument::Object(region_length),
                        ]);
                        let region_offset = region_offset.clone().unwrap_transparent_reference()?;
                        let region_length = region_length.clone().unwrap_transparent_reference()?;

                        let name = context.resolve_name(&name, &context.current_scope)?;
                        let parent_device_path = MeteredNameString::clone_name_metered(
                            &context.current_scope,
                            context.execution_meter.clone(),
                        )?;
                        let (parent_device_path, parent_path_heap_reservation) = parent_device_path.into_parts();
                        let namespace_reservation = self.reserve_namespace_object(&name, 0)?;
                        let region_space = RegionSpace::from(*region_space);
                        let region = Object::OpRegion(OpRegion {
                            space: region_space,
                            base: region_offset.as_integer()?,
                            length: region_length.as_integer()?,
                            parent_device_path,
                        })
                        .wrap();
                        region.attach_heap_reservation(allocator::box_new(parent_path_heap_reservation))?;
                        self.prepare_dynamic_native_region_redefinition(region_space)?;
                        if self.insert_metered_name_if_mutable(name, region, namespace_reservation)? {
                            self.register_dynamic_native_region(&context.current_scope, region_space)?;
                        }
                        context.retire_op(op);
                    }
                    Opcode::DataRegion => {
                        extract_args!(op => [
                            Argument::Namestring(name),
                            Argument::Object(signature),
                            Argument::Object(oem_id),
                            Argument::Object(oem_table_id),
                        ]);
                        let _signature = signature.as_string()?;
                        let _oem_id = oem_id.as_string()?;
                        let _oem_table_id = oem_table_id.as_string()?;

                        // TODO: once this is integrated into the rest of the crate, load the table
                        log::warn!(
                            "DefDataRegion encountered in AML! We don't actually support these - produced region will be incorrect"
                        );

                        let name = context.resolve_name(&name, &context.current_scope)?;
                        let parent_device_path = MeteredNameString::clone_name_metered(
                            &context.current_scope,
                            context.execution_meter.clone(),
                        )?;
                        let (parent_device_path, parent_path_heap_reservation) = parent_device_path.into_parts();
                        let namespace_reservation = self.reserve_namespace_object(&name, 0)?;
                        let region = Object::OpRegion(OpRegion {
                            space: RegionSpace::SystemMemory,
                            base: 0,
                            length: 0,
                            parent_device_path,
                        })
                        .wrap();
                        region.attach_heap_reservation(allocator::box_new(parent_path_heap_reservation))?;
                        self.prepare_dynamic_native_region_redefinition(RegionSpace::SystemMemory)?;
                        if self.insert_metered_name_if_mutable(name, region, namespace_reservation)? {
                            self.register_dynamic_native_region(
                                &context.current_scope,
                                RegionSpace::SystemMemory,
                            )?;
                        }
                        context.retire_op(op);
                    }
                    Opcode::Buffer => {
                        extract_args!(op => [
                            Argument::TrackedPc(start_pc),
                            Argument::PkgLength(pkg_length),
                            Argument::Object(buffer_size),
                        ]);
                        let buffer_size =
                            buffer_size.clone().unwrap_transparent_reference()?.as_integer()? as usize;

                        let package_consumed = context
                            .current_block
                            .pc
                            .checked_sub(*start_pc)
                            .ok_or_else(|| AmlError::HostError("AML_BUFFER_PACKAGE_RANGE".to_string()))?;
                        let buffer_len = pkg_length
                            .checked_sub(package_consumed)
                            .ok_or_else(|| AmlError::HostError("AML_BUFFER_PACKAGE_LENGTH".to_string()))?;
                        let data_start = context.current_block.pc;
                        let data_end = data_start
                            .checked_add(buffer_len)
                            .filter(|end| *end <= context.current_block.stream().len())
                            .ok_or(AmlError::RunOutOfStream)?;
                        let buffer_bytes = u64::try_from(buffer_size).map_err(|_| AmlError::BudgetExceeded)?;
                        let heap_bytes = buffer_bytes
                            .checked_add(if buffer_size == 0 { 0 } else { 24 })
                            .ok_or(AmlError::BudgetExceeded)?;
                        let reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                        let mut buffer = try_vec_with_capacity(buffer_size)?;
                        buffer.resize(buffer_size, 0);

                        /*
                         * Copy the supplied elements into the buffer, avoiding a pathological case
                         * where more elements are supplied than the buffer's size
                         */
                        let to_copy = usize::min(buffer_size, buffer_len);
                        buffer[0..to_copy]
                            .copy_from_slice(&context.current_block.stream()[data_start..data_start + to_copy]);
                        context.current_block.pc = data_end;

                        context.contribute_arg(Argument::Object(
                            self.wrap_object_with_heap_reservation(Object::Buffer(buffer), reservation)?,
                        ));
                        context.retire_op(op);
                    }
                    Opcode::Package => {
                        let package_bytes = (op.expected_arguments as u64)
                            .checked_mul(mem::size_of::<WrappedObject>() as u64)
                            .ok_or(AmlError::HostError("AML_PACKAGE_SIZE_OVERFLOW".to_string()))?;
                        let heap_bytes = package_bytes
                            .checked_add(if op.expected_arguments == 0 { 0 } else { 24 })
                            .ok_or(AmlError::BudgetExceeded)?;
                        let reservation = self.reserve_object_heap(heap_bytes, package_bytes, 0)?;
                        let mut elements = try_vec_with_capacity(op.expected_arguments)?;
                        for arg in &op.arguments {
                            let Argument::Object(object) = arg else {
                                return Err(AmlError::InternalError(
                                    "Invalid argument type produced for package element".to_string(),
                                ));
                            };
                            elements.push(object.clone());
                        }

                        /*
                         * We can end up completing a package's in-flight op in two circumstances:
                         *    - If the correct number of elements are supplied, we end up here
                         *      first, and then later in the block's finishing logic.
                         *    - If less elements are supplied, we end up in the block's finishing
                         *      logic to add some `Uninitialized`s, then go round again to complete
                         *      the in-flight operation.
                         *
                         * To make these consistent, we always remove the block here, making sure
                         * we've finished it as a sanity check.
                         */
                        if context.current_block.kind != BlockKind::Package
                            || !matches!(context.peek(), Err(AmlError::RunOutOfStream))
                        {
                            return Err(AmlError::InvalidPackageLength);
                        }
                        context.current_block = context.pop_block()?;
                        context.contribute_arg(Argument::Object(
                            self.wrap_object_with_heap_reservation(Object::Package(elements), reservation)?,
                        ));
                        context.retire_op(op);
                    }
                    Opcode::VarPackage => {
                        extract_args!(op[0..1] => [Argument::Object(total_elements)]);
                        let total_elements =
                            total_elements.clone().unwrap_transparent_reference()?.as_integer()? as usize;

                        let package_bytes = (total_elements as u64)
                            .checked_mul(mem::size_of::<WrappedObject>() as u64)
                            .ok_or(AmlError::HostError("AML_PACKAGE_SIZE_OVERFLOW".to_string()))?;
                        let heap_bytes = package_bytes
                            .checked_add(if total_elements == 0 { 0 } else { 24 })
                            .ok_or(AmlError::BudgetExceeded)?;
                        let reservation = self.reserve_object_heap(heap_bytes, package_bytes, 0)?;

                        let mut elements = try_vec_with_capacity(total_elements)?;
                        for arg in &op.arguments[1..] {
                            let Argument::Object(object) = arg else {
                                return Err(AmlError::InternalError(
                                    "Invalid argument type produced for package element".to_string(),
                                ));
                            };
                            elements.push(object.clone());
                        }

                        /*
                         * As above, we always remove the block here after the in-flight op has
                         * been retired.
                         */
                        if context.current_block.kind != BlockKind::VarPackage
                            || !matches!(context.peek(), Err(AmlError::RunOutOfStream))
                        {
                            return Err(AmlError::InvalidPackageLength);
                        }
                        context.current_block = context.pop_block()?;
                        context.contribute_arg(Argument::Object(
                            self.wrap_object_with_heap_reservation(Object::Package(elements), reservation)?,
                        ));
                        context.retire_op(op);
                    }
                    Opcode::If => {
                        extract_args!(op => [
                            Argument::TrackedPc(start_pc),
                            Argument::PkgLength(then_length),
                            Argument::Object(predicate),
                        ]);
                        let predicate = predicate.as_integer()?;
                        let remaining_then_length = context.remaining_package_length(*start_pc, *then_length)?;

                        if predicate > 0 {
                            context.start_new_block(BlockKind::IfThenBranch, remaining_then_length)?;
                        } else {
                            context.current_block.pc = context
                                .current_block
                                .pc
                                .checked_add(remaining_then_length)
                                .ok_or(AmlError::InvalidPackageLength)?;

                            /*
                             * Skip over the prolog to the else branch if present. Also handle if
                             * there are no more bytes to peek - the `If` op could be the last op
                             * in a block.
                             */
                            const DEF_ELSE_OP: u8 = 0xa1;
                            match context.peek() {
                                Ok(DEF_ELSE_OP) => {
                                    context.next()?;
                                    let _else_length = context.pkglength()?;
                                }
                                Ok(_) => (),
                                Err(AmlError::RunOutOfStream) => (),
                                Err(other) => Err(other)?,
                            }
                        }
                        context.retire_op(op);
                    }
                    opcode @ Opcode::CreateBitField
                    | opcode @ Opcode::CreateByteField
                    | opcode @ Opcode::CreateWordField
                    | opcode @ Opcode::CreateDWordField
                    | opcode @ Opcode::CreateQWordField => {
                        extract_args!(op => [Argument::Object(buffer), Argument::Object(index)]);
                        let name = context.namestring()?;
                        let index = index.as_integer()?;
                        let (bit_offset, length) = match opcode {
                            Opcode::CreateBitField => (index, 1),
                            Opcode::CreateByteField => {
                                (index.checked_mul(8).ok_or(AmlError::IndexOutOfBounds)?, 8)
                            }
                            Opcode::CreateWordField => {
                                (index.checked_mul(8).ok_or(AmlError::IndexOutOfBounds)?, 16)
                            }
                            Opcode::CreateDWordField => {
                                (index.checked_mul(8).ok_or(AmlError::IndexOutOfBounds)?, 32)
                            }
                            Opcode::CreateQWordField => {
                                (index.checked_mul(8).ok_or(AmlError::IndexOutOfBounds)?, 64)
                            }
                            _ => unreachable!(),
                        };
                        let offset = usize::try_from(bit_offset).map_err(|_| AmlError::IndexOutOfBounds)?;
                        offset.checked_add(length).ok_or(AmlError::IndexOutOfBounds)?;
                        let name = context.resolve_name(&name, &context.current_scope)?;
                        let reservation = self.reserve_namespace_object(&name, 0)?;
                        self.insert_metered_name(
                            name,
                            Object::BufferField { buffer: buffer.clone(), offset, length }.wrap(),
                            reservation,
                        )?;
                        context.retire_op(op);
                    }
                    Opcode::CreateField => {
                        extract_args!(op => [Argument::Object(buffer), Argument::Object(bit_index), Argument::Object(num_bits)]);
                        let name = context.namestring()?;
                        let bit_index = bit_index.as_integer()?;
                        let num_bits = num_bits.as_integer()?;
                        let offset = usize::try_from(bit_index).map_err(|_| AmlError::IndexOutOfBounds)?;
                        let length = usize::try_from(num_bits).map_err(|_| AmlError::IndexOutOfBounds)?;
                        offset.checked_add(length).ok_or(AmlError::IndexOutOfBounds)?;
                        let name = context.resolve_name(&name, &context.current_scope)?;
                        let reservation = self.reserve_namespace_object(&name, 0)?;

                        self.insert_metered_name(
                            name,
                            Object::BufferField { buffer: buffer.clone(), offset, length }.wrap(),
                            reservation,
                        )?;
                        context.retire_op(op);
                    }
                    Opcode::CopyObject => {
                        extract_args!(op => [Argument::Object(object), Argument::Object(target)]);
                        self.do_copy_object(target.clone(), object.clone())?;
                        context.retire_op(op);
                    }
                    Opcode::Store => {
                        extract_args!(op => [Argument::Object(object), Argument::Object(target)]);
                        self.do_store(target.clone(), object.clone())?;
                        context.retire_op(op);
                    }
                    Opcode::RefOf => {
                        extract_args!(op => [Argument::Object(object)]);
                        let reference = self.wrap_object(Object::Reference {
                            kind: ReferenceKind::RefOf,
                            inner: object.clone(),
                        })?;
                        context.contribute_arg(Argument::Object(reference));
                        context.retire_op(op);
                    }
                    Opcode::CondRefOf => {
                        extract_args!(op => [Argument::Object(object), Argument::Object(target)]);
                        let result = if let Object::Reference { kind: ReferenceKind::Unresolved, .. } = **object {
                            Object::Integer(0)
                        } else {
                            let reference = self.wrap_object(Object::Reference {
                                kind: ReferenceKind::RefOf,
                                inner: object.clone(),
                            })?;
                            self.do_store(target.clone(), reference)?;
                            Object::Integer(u64::MAX)
                        };
                        context.contribute_arg(Argument::Object(self.wrap_object(result)?));
                        context.retire_op(op);
                    }
                    Opcode::DerefOf => {
                        extract_args!(op => [Argument::Object(object)]);
                        let result = self.do_deref_of(object.clone(), &context.current_scope)?;
                        context.contribute_arg(Argument::Object(result));
                        context.retire_op(op);
                    }
                    Opcode::Load => {
                        extract_args!(op => [Argument::Namestring(object), Argument::Object(result)]);
                        // TODO: read the AML from the object and load it
                        warn!("Ignoring unsupported DefLoad operation (object={}, result = {})", object, result);
                        context.retire_op(op);
                        return Err(AmlError::LibUnimplemented);
                    }
                    Opcode::LoadTable => {
                        extract_args!(op => [
                            Argument::Object(signature),
                            Argument::Object(oem_id),
                            Argument::Object(oem_table_id),
                            Argument::Object(root_path),
                            Argument::Object(parameter_path),
                            Argument::Object(parameter_data),
                        ]);
                        // TODO: search for the table in the RSDT/XSDT and load the contained AML
                        warn!(
                            "Ignoring unsupported DefLoadTable operation (signature = {}, oem_id = {}, oem_table_id = {}, root_path = {}, parameter_path = {}, parameter_data = {})",
                            signature, oem_id, oem_table_id, root_path, parameter_path, parameter_data
                        );
                        context.retire_op(op);
                        return Err(AmlError::LibUnimplemented);
                    }
                    Opcode::Sleep => {
                        extract_args!(op => [Argument::Object(msec)]);
                        self.handler.sleep(msec.as_integer()?);
                        context.retire_op(op);
                    }
                    Opcode::Stall => {
                        extract_args!(op => [Argument::Object(usec)]);
                        self.handler.stall(usec.as_integer()?);
                        context.retire_op(op);
                    }
                    Opcode::Acquire => {
                        extract_args!(op => [Argument::Object(mutex)]);
                        let mutex = mutex.clone().unwrap_reference()?;
                        let Object::Mutex { mutex, sync_level: _ } = *mutex else {
                            Err(AmlError::InvalidOperationOnObject { op: Operation::Acquire, typ: mutex.typ() })?
                        };
                        let timeout = context.next_u16()?;

                        // TODO: should we do something with the sync level??
                        let acquire_result = if mutex == self.global_lock_mutex {
                            self.global_lock_dependency.store(true, Ordering::Release);
                            self.acquire_global_lock(timeout)
                        } else {
                            self.handler.acquire(mutex, timeout)
                        };
                        let timed_out = match acquire_result {
                            Ok(()) => false,
                            Err(AmlError::MutexAcquireTimeout) => true,
                            Err(error) => return Err(error),
                        };

                        context.retire_op(op);
                        context.contribute_arg(Argument::Object(
                            self.wrap_object(Object::Integer(if timed_out { u64::MAX } else { 0 }))?,
                        ));
                    }
                    Opcode::Release => {
                        extract_args!(op => [Argument::Object(mutex)]);
                        let mutex = mutex.clone().unwrap_reference()?;
                        let Object::Mutex { mutex, sync_level: _ } = *mutex else {
                            Err(AmlError::InvalidOperationOnObject { op: Operation::Release, typ: mutex.typ() })?
                        };

                        // TODO: should we do something with the sync level??
                        if mutex == self.global_lock_mutex {
                            self.release_global_lock()?;
                        } else {
                            self.handler.release(mutex);
                        }

                        context.retire_op(op);
                    }
                    Opcode::InternalMethodCall => {
                        if context_stack.len() >= MAX_STACK_DEPTH {
                            return Err(AmlError::MethodStackExceeded);
                        }

                        extract_args!(op[0..2] => [Argument::Object(method), Argument::Namestring(method_scope)]);
                        validate_compatibility_method_entry(
                            &method_scope,
                            self.pic_evaluation_state.load(Ordering::Acquire),
                            self.routing_methods_enabled.load(Ordering::Acquire),
                        )?;
                        let argument_count = op.arguments.len().saturating_sub(2);
                        let execution_meter = self.execution_meter.lock().as_ref().map(Arc::clone);
                        let _argument_heap_reservation =
                            reserve_argument_scratch(execution_meter.clone(), argument_count)?;
                        let mut args = allocator::vec_new();
                        args.try_reserve_exact(argument_count).map_err(|_| AmlError::BudgetExceeded)?;
                        for arg in &op.arguments[2..] {
                            let Argument::Object(arg) = arg else {
                                return Err(AmlError::InternalError(
                                    "AML method argument was not resolved to an object".to_string(),
                                ));
                            };
                            args.push(arg.clone());
                        }

                        if let Object::Method { .. } = **method {
                            context_stack_heap_reservation
                                .reserve_vec_capacity(&mut context_stack, MAX_AML_CONTEXT_STACK_DEPTH)?;
                            let namespace_scope_reservation = method_scope.reserve_clone_scratch()?;
                            self.namespace
                                .lock()
                                .add_level(method_scope.try_clone()?, NamespaceLevelKind::MethodLocals)?;
                            drop(namespace_scope_reservation);

                            let new_context = MethodContext::new_from_method(
                                method.clone(),
                                args,
                                method_scope.try_clone_metered()?,
                                execution_meter,
                            )?;
                            meter.enter()?;
                            let old_context = mem::replace(&mut context, new_context);
                            context_stack.push(old_context);
                            context.retire_op(op);
                        } else if let Object::NativeMethod { ref f, .. } = **method {
                            let result = f(&args, context.execution_meter.clone())?;
                            context.contribute_arg(Argument::Object(result));
                        } else {
                            return Err(AmlError::InternalError(
                                "internal method call target is not a method".to_string(),
                            ));
                        }
                    }
                    Opcode::Return => {
                        extract_args!(op => [Argument::Object(object)]);
                        let object = object.clone().unwrap_transparent_reference()?;
                        if let Some(method_scope) = context.method_scope() {
                            let method_scope = MeteredNameString::clone_name_metered(
                                method_scope,
                                context.execution_meter.clone(),
                            )?;
                            let (method_scope, _method_scope_heap_reservation) = method_scope.into_parts();
                            self.namespace.lock().remove_level(method_scope)?;
                        }

                        if let Some(last) = context_stack.pop() {
                            meter.leave();
                            context = last;
                            context.contribute_arg(Argument::Object(object.clone()));
                            context.retire_op(op);
                        } else {
                            /*
                             * If this is the top-most context, this is a `Return` from the actual
                             * method.
                             */
                            return Ok(object.clone());
                        }
                    }
                    Opcode::ObjectType => {
                        extract_args!(op => [Argument::Object(object)]);
                        let object_type = self.object_type(object.clone())?;
                        context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(object_type))?));
                        context.retire_op(op);
                    }
                    Opcode::SizeOf => self.do_size_of(&mut context, op)?,
                    Opcode::Index => self.do_index(&mut context, op)?,
                    Opcode::BankField => {
                        extract_args!(op => [
                            Argument::TrackedPc(start_pc),
                            Argument::PkgLength(pkg_length),
                            Argument::Namestring(region_name),
                            Argument::Namestring(bank_name),
                            Argument::Object(bank_value),
                        ]);
                        let bank_value = bank_value.as_integer()?;
                        let field_flags = context.next()?;

                        let (region, bank) = {
                            let namespace = self.namespace.lock();
                            let (_, region) = context.search(&namespace, region_name)?;
                            let (_, bank) = context.search(&namespace, bank_name)?;
                            (region, bank)
                        };

                        let kind = FieldUnitKind::Bank { region, bank, bank_value };
                        self.parse_field_list(&mut context, kind, *start_pc, *pkg_length, field_flags)?;
                        context.retire_op(op);
                    }
                    Opcode::While => {
                        /*
                         * We've just evaluated the predicate for an iteration of a while loop. If
                         * false, skip over the rest of the loop, otherwise carry on.
                         */
                        extract_args!(op => [Argument::Object(predicate)]);
                        let predicate = predicate.clone().unwrap_transparent_reference()?.as_integer()?;

                        if predicate == 0 {
                            // Exit from the while loop by skipping out of the current block
                            context.current_block = context.pop_block()?;
                            context.retire_op(op);
                        }
                    }
                    Opcode::Match => {
                        extract_args!(op => [
                            Argument::Object(search_pkg),
                            Argument::MatchOp(opcode_a),
                            Argument::Object(operand_a),
                            Argument::MatchOp(opcode_b),
                            Argument::Object(operand_b),
                            Argument::Object(start_index)]
                        );

                        let Object::Package(ref pkg) = **search_pkg else {
                            return Err(AmlError::ObjectNotOfExpectedType {
                                expected: ObjectType::Package,
                                got: search_pkg.typ(),
                            });
                        };

                        let result = do_match(
                            pkg,
                            opcode_a,
                            operand_a.as_integer()?,
                            opcode_b,
                            operand_b.as_integer()?,
                            start_index.as_integer()?,
                        );

                        context.contribute_arg(Argument::Object(Object::Integer(result).wrap()));
                        context.retire_op(op);
                    }
                    _ => {
                        return Err(AmlError::InternalError(
                            "unexpected operation entered the in-flight execution stack".to_string(),
                        ));
                    }
                }
            }

            if let Some(error) = context.take_allocation_error() {
                return Err(error);
            }

            /*
             * Now that we've retired as many in-flight operations as we have arguments for, move
             * forward in the AML stream.
             */
            let opcode = match context.opcode() {
                Ok(opcode) => opcode,
                Err(AmlError::RunOutOfStream) => {
                    /*
                     * We've reached the end of the current block. What we should do about this
                     * depends on what type of block it was.
                     */
                    match context.current_block.kind {
                        BlockKind::Table => {
                            break Ok(self.wrap_object(Object::Uninitialized)?);
                        }
                        BlockKind::Method { method_scope } => {
                            self.namespace.lock().remove_level(method_scope)?;

                            if let Some(prev_context) = context_stack.pop() {
                                context = prev_context;
                                continue;
                            } else {
                                /*
                                 * If there is no explicit `Return` op, the result is undefined. We
                                 * just return an uninitialized object.
                                 */
                                return Ok(self.wrap_object(Object::Uninitialized)?);
                            }
                        }
                        BlockKind::Scope { old_scope } => {
                            context.current_block = context
                                .block_stack
                                .pop()
                                .ok_or_else(|| AmlError::InternalError("AML block stack underflow".to_string()))?;
                            context.leave_scope(old_scope)?;
                            // Go round the loop again to get the next opcode for the new block
                            continue;
                        }
                        BlockKind::Package => {
                            /*
                             * We've reached the end of the package. The in-flight op may have
                             * already been completed in the case of the package specifying all of
                             * its elements, or reach the end of the block here if it does not.
                             *
                             * In the latter case, fill in the rest of the package with
                             * *distinct* uninitialized objects, and go round again to complete the
                             * in-flight op.
                             */
                            if context.block_stack.is_empty() {
                                return Err(AmlError::InternalError("package block stack underflow".to_string()));
                            }

                            if let Some(package_op) = context.in_flight.last_mut()
                                && package_op.op == Opcode::Package
                            {
                                let num_elements_left = package_op
                                    .expected_arguments
                                    .checked_sub(package_op.arguments.len())
                                    .ok_or(AmlError::InvalidPackageLength)?;
                                for _ in 0..num_elements_left {
                                    package_op
                                        .arguments
                                        .push(Argument::Object(self.wrap_object(Object::Uninitialized)?));
                                }
                            }

                            // XXX: don't remove the package's block. Refer to completion of
                            // package ops for rationale here.
                            continue;
                        }
                        BlockKind::VarPackage => {
                            if context.block_stack.is_empty() {
                                return Err(AmlError::InternalError(
                                    "VarPackage block stack underflow".to_string(),
                                ));
                            }

                            if let Some(package_op) = context.in_flight.last_mut()
                                && package_op.op == Opcode::VarPackage
                            {
                                let num_elements_left = {
                                    let Some(Argument::Object(total_elements)) = package_op.arguments.first()
                                    else {
                                        return Err(AmlError::InvalidPackageLength);
                                    };
                                    let total_elements =
                                        total_elements.clone().unwrap_transparent_reference()?.as_integer()?
                                            as usize;
                                    let parsed_elements = package_op
                                        .arguments
                                        .len()
                                        .checked_sub(1)
                                        .ok_or(AmlError::InvalidPackageLength)?;

                                    // Update the expected number of arguments to terminate the in-flight op
                                    package_op.expected_arguments = package_op.arguments.len();
                                    total_elements
                                        .checked_sub(parsed_elements)
                                        .ok_or(AmlError::InvalidPackageLength)?
                                };

                                for _ in 0..num_elements_left {
                                    package_op
                                        .arguments
                                        .push(Argument::Object(self.wrap_object(Object::Uninitialized)?));
                                }
                            }

                            // As above, leave the package's block.
                            continue;
                        }
                        BlockKind::IfThenBranch => {
                            context.current_block = context.pop_block()?;

                            /*
                             * Check for an else-branch, and skip over it. We need to handle the
                             * case here where there isn't a next byte - that just means the `If`
                             * is the last op in a block.
                             */
                            const DEF_ELSE_OP: u8 = 0xa1;
                            match context.peek() {
                                Ok(DEF_ELSE_OP) => {
                                    context.next()?;
                                    let start_pc = context.current_block.pc;
                                    let else_length = context.pkglength()?;
                                    let encoded_length = context
                                        .current_block
                                        .pc
                                        .checked_sub(start_pc)
                                        .ok_or(AmlError::InvalidPackageLength)?;
                                    let remaining_length = else_length
                                        .checked_sub(encoded_length)
                                        .ok_or(AmlError::InvalidPackageLength)?;
                                    let next_pc = context
                                        .current_block
                                        .pc
                                        .checked_add(remaining_length)
                                        .filter(|next_pc| *next_pc <= context.current_block.stream().len())
                                        .ok_or(AmlError::InvalidPackageLength)?;
                                    context.current_block.pc = next_pc;
                                }
                                Ok(_) => (),
                                Err(AmlError::RunOutOfStream) => (),
                                Err(other) => Err(other)?,
                            };

                            continue;
                        }
                        BlockKind::While { .. } => {
                            self.continue_loop(&mut context)?;
                            continue;
                        }
                    }
                }
                Err(other_err) => return Err(other_err),
            };
            match opcode {
                Opcode::Zero => {
                    /*
                     * This represents a `Zero` operand that should create an `Integer` operand in
                     * most places, but could also encode a `NullName` if we are expecting a
                     * `Target`. We handle the latter in logic for stores to targets.
                     */
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(0))?));
                }
                Opcode::One => {
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(1))?));
                }
                Opcode::MatchOp(opcode) => {
                    if context.expecting_match_opcode() {
                        context.contribute_arg(Argument::MatchOp(opcode));
                    } else {
                        Err(AmlError::IllegalOpcode(opcode as u16))?
                    }
                }
                Opcode::Ones => {
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(u64::MAX))?));
                }
                Opcode::Alias => {
                    let source = context.namestring()?;
                    let alias = context.namestring()?;
                    let source_path = context.resolve_name(&source, &context.current_scope)?;
                    if *source_path == AmlName::from_str("\\_GL_")? {
                        self.global_lock_dependency.store(true, Ordering::Release);
                    }

                    let mut namespace = self.namespace.lock();
                    let object = namespace.get_absolute(&source_path)?;
                    let alias = context.resolve_name(&alias, &context.current_scope)?;
                    let reservation = self.reserve_namespace_object(&alias, 0)?;
                    let (alias, _alias_heap_reservation) = alias.into_parts();
                    namespace.create_alias_with_reservation(alias, object, reservation)?;
                }
                Opcode::Name => {
                    let name = context.namestring()?;
                    context.start_with_arguments(
                        Opcode::Name,
                        2,
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                        1,
                        move |arguments| {
                            arguments.push(Argument::Namestring(name));
                            Ok(())
                        },
                    )?;
                }
                Opcode::BytePrefix => {
                    let value = context.next()?;
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(value as u64))?));
                }
                Opcode::WordPrefix => {
                    let value = context.next_u16()?;
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(value as u64))?));
                }
                Opcode::DWordPrefix => {
                    let value = context.next_u32()?;
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(value as u64))?));
                }
                Opcode::StringPrefix => {
                    let str_start = context.current_block.pc;
                    while context.next()? != b'\0' {}
                    let string_bytes = &context.current_block.stream()[str_start..(context.current_block.pc - 1)];
                    let heap_bytes = string_bytes
                        .len()
                        .checked_add(24)
                        .and_then(|bytes| u64::try_from(bytes).ok())
                        .ok_or_else(|| AmlError::BudgetExceeded)?;
                    let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                    let text = str::from_utf8(string_bytes).map_err(|_| AmlError::InvalidOperationOnObject {
                        op: Operation::ToString,
                        typ: ObjectType::Buffer,
                    })?;
                    let mut string = String::new();
                    string.try_reserve_exact(text.len()).map_err(|_| AmlError::BudgetExceeded)?;
                    string.push_str(text);
                    context.contribute_arg(Argument::Object(
                        self.wrap_object_with_heap_reservation(Object::String(string), reservation)?,
                    ));
                }
                Opcode::QWordPrefix => {
                    let value = context.next_u64()?;
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(value))?));
                }
                Opcode::Scope => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;

                    let remaining_length = context.remaining_package_length(start_pc, pkg_length)?;

                    let new_scope = context.resolve_name(&name, &context.current_scope)?;
                    let reservation = self.reserve_namespace_object(&new_scope, 0)?;
                    let scope_path_reservation = new_scope.reserve_clone_scratch()?;
                    self.namespace.lock().add_level_with_reservation(
                        new_scope.try_clone()?,
                        NamespaceLevelKind::Scope,
                        reservation,
                    )?;
                    drop(scope_path_reservation);

                    let old_scope = context.enter_scope(new_scope)?;
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length)?;
                }
                Opcode::Buffer => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    context.start_with_arguments(
                        Opcode::Buffer,
                        3,
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                        2,
                        |arguments| {
                            arguments.push(Argument::TrackedPc(start_pc));
                            arguments.push(Argument::PkgLength(pkg_length));
                            Ok(())
                        },
                    )?;
                }
                Opcode::Package => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let num_elements = context.next()?;

                    let remaining_length = context.remaining_package_length(start_pc, pkg_length)?;

                    /*
                     * We now need to interpret an arbitrary number of package elements, bounded by
                     * the remaining pkglength. This may be less than `num_elements` - the
                     * remaining elements of the package are uninitialized. We utilise a
                     * combination of a block to manage the pkglength, plus an in-flight op to
                     * store interpreted arguments.
                     */
                    context.start(OpInFlight::new_dynamic(
                        Opcode::Package,
                        num_elements as usize,
                        &[ResolveBehaviour::AsPackageElements],
                    ));
                    context.start_new_block(BlockKind::Package, remaining_length)?;
                }
                Opcode::VarPackage => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let remaining_length = context.remaining_package_length(start_pc, pkg_length)?;

                    /*
                     * For variable packages, we're first going to parse a `TermArg` that encodes,
                     * dynamically, how many elements the package will have. We then accept as many
                     * elements as remain in the block, and we'll sort out how many are supposed to
                     * be in the package later.
                     */
                    context.start(OpInFlight::new_dynamic(
                        Opcode::VarPackage,
                        usize::MAX,
                        &[ResolveBehaviour::TermArg, ResolveBehaviour::AsPackageElements],
                    ));
                    context.start_new_block(BlockKind::VarPackage, remaining_length)?;
                }
                Opcode::Method => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;
                    let flags = MethodFlags(context.next()?);

                    let code_len = context.remaining_package_length(start_pc, pkg_length)?;
                    let name = context.resolve_name(&name, &context.current_scope)?;
                    let reservation = self.reserve_namespace_object(&name, code_len as u64)?;
                    let code_bytes = &context.current_block.stream()
                        [context.current_block.pc..(context.current_block.pc + code_len)];
                    let code = try_vec_from_slice(code_bytes)?;
                    let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
                    if method_global_lock_facts(&code, meter.as_deref())?.acquires_global_lock {
                        self.global_lock_dependency.store(true, Ordering::Release);
                    }
                    context.current_block.pc += code_len;

                    self.insert_metered_name(name, Object::Method { code, flags }.wrap(), reservation)?;
                }
                Opcode::External => {
                    let _name = context.namestring()?;
                    let _object_type = context.next()?;
                    let _arg_count = context.next()?;
                }
                Opcode::Mutex => {
                    let name = context.namestring()?;
                    let sync_level = context.next()?;

                    let name = context.resolve_name(&name, &context.current_scope)?;
                    let reservation = self.reserve_namespace_object(&name, 0)?;
                    let mutex = self.handler.create_mutex();
                    self.insert_metered_name(name, Object::Mutex { mutex, sync_level }.wrap(), reservation)?;
                }
                Opcode::Event => {
                    let name = context.namestring()?;

                    let name = context.resolve_name(&name, &context.current_scope)?;
                    let reservation = self.reserve_namespace_object(&name, 0)?;
                    self.insert_metered_name(
                        name,
                        Object::Event(allocator::arc_new(AtomicU64::new(0))).wrap(),
                        reservation,
                    )?;
                }
                Opcode::LoadTable => {
                    context.start(OpInFlight::new(Opcode::LoadTable, &[ResolveBehaviour::TermArg; 6]));
                }
                Opcode::Load => {
                    let name = context.namestring()?;
                    context.start_with_arguments(
                        Opcode::Load,
                        1,
                        &[ResolveBehaviour::Target],
                        1,
                        move |arguments| {
                            arguments.push(Argument::Namestring(name));
                            Ok(())
                        },
                    )?;
                }
                Opcode::Stall => context.start(OpInFlight::new(Opcode::Stall, &[ResolveBehaviour::TermArg])),
                Opcode::Sleep => context.start(OpInFlight::new(Opcode::Sleep, &[ResolveBehaviour::TermArg])),
                Opcode::Acquire => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Release => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Signal => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Wait => context
                    .start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName, ResolveBehaviour::TermArg])),
                Opcode::Reset => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Notify => context
                    .start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName, ResolveBehaviour::TermArg])),
                Opcode::FromBCD | Opcode::ToBCD => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target]))
                }
                Opcode::Revision => {
                    context.contribute_arg(Argument::Object(
                        self.wrap_object(Object::Integer(INTERPRETER_REVISION))?,
                    ));
                }
                Opcode::Debug => context.contribute_arg(Argument::Object(self.wrap_object(Object::Debug)?)),
                Opcode::Fatal => {
                    let typ = context.next()?;
                    let code = context.next_u32()?;
                    context.start_with_arguments(
                        Opcode::Fatal,
                        3,
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                        2,
                        |arguments| {
                            arguments.push(Argument::ByteData(typ));
                            arguments.push(Argument::DWordData(code));
                            Ok(())
                        },
                    )?;
                }
                Opcode::Timer => {
                    // Time has to be monotonically-increasing, in 100ns units
                    let time = self.handler.nanos_since_boot() / 100;
                    context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(time))?));
                }
                Opcode::OpRegion => {
                    let name = context.namestring()?;
                    let region_space = context.next()?;
                    context.start_with_arguments(
                        Opcode::OpRegion,
                        4,
                        &[
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::TermArg,
                            ResolveBehaviour::TermArg,
                        ],
                        2,
                        move |arguments| {
                            arguments.push(Argument::Namestring(name));
                            arguments.push(Argument::ByteData(region_space));
                            Ok(())
                        },
                    )?;
                }
                Opcode::DataRegion => {
                    let name = context.namestring()?;
                    context.start_with_arguments(
                        Opcode::DataRegion,
                        4,
                        &[
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::TermArg,
                            ResolveBehaviour::TermArg,
                            ResolveBehaviour::TermArg,
                        ],
                        1,
                        move |arguments| {
                            arguments.push(Argument::Namestring(name));
                            Ok(())
                        },
                    )?;
                }
                Opcode::Field => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let region_name = context.namestring()?;
                    let field_flags = context.next()?;

                    let (_, region) = context.search(&self.namespace.lock(), &region_name)?;
                    let kind = FieldUnitKind::Normal { region };
                    self.parse_field_list(&mut context, kind, start_pc, pkg_length, field_flags)?;
                }
                Opcode::BankField => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let region_name = context.namestring()?;
                    let bank_name = context.namestring()?;

                    context.start_with_arguments(
                        Opcode::BankField,
                        5,
                        &[
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::Placeholder,
                            ResolveBehaviour::TermArg,
                        ],
                        4,
                        move |arguments| {
                            arguments.push(Argument::TrackedPc(start_pc));
                            arguments.push(Argument::PkgLength(pkg_length));
                            arguments.push(Argument::Namestring(region_name));
                            arguments.push(Argument::Namestring(bank_name));
                            Ok(())
                        },
                    )?;
                }
                Opcode::IndexField => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let index_name = context.namestring()?;
                    let data_name = context.namestring()?;
                    let field_flags = context.next()?;

                    let (index, data) = {
                        let namespace = self.namespace.lock();
                        let (_, index) = context.search(&namespace, &index_name)?;
                        let (_, data) = context.search(&namespace, &data_name)?;
                        (index, data)
                    };

                    if let Object::FieldUnit(ref data_fu) = *data {
                        if data_fu.flags.access_type_bytes()? < FieldFlags(field_flags).access_type_bytes()? {
                            // On ACPICA this causes reads/writes to be truncated to the width of
                            // the data register.
                            //
                            // The issue comes from this part of the spec:
                            // "The value written to the IndexName register is defined to be a byte
                            // offset that is aligned on an AccessType boundary."
                            //
                            // Consider the case where the index field is WordAcc but the data
                            // register is ByteAcc. Only even numbers could be written to the index
                            // register - multiples of width of word (2 bytes). But to access the
                            // high byte of the word for the field, we'd need to write an odd number
                            // to the index register. Which is not compatible with the spec.
                            //
                            // The ASL writer shouldn't have allowed this. And it seems that most
                            // uses of IndexField are ByteAcc all around. So whatever strange
                            // behaviour we allow is probably OK.
                            //
                            // But warn the user, just in case.
                            warn!("Data field access width is smaller than normal field width at {:?}", start_pc);
                        }
                    } else {
                        warn!("Wrong data field type in IndexField: {:?}", data);
                    };

                    let kind = FieldUnitKind::Index { index, data };
                    self.parse_field_list(&mut context, kind, start_pc, pkg_length, field_flags)?;
                }
                Opcode::Device | Opcode::ThermalZone => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;

                    let remaining_length = context.remaining_package_length(start_pc, pkg_length)?;

                    let new_scope = context.resolve_name(&name, &context.current_scope)?;
                    let level_reservation = self.reserve_namespace_object(&new_scope, 0)?;
                    let (kind, object) = match opcode {
                        Opcode::Device => (NamespaceLevelKind::Device, Object::Device),
                        Opcode::ThermalZone => (NamespaceLevelKind::ThermalZone, Object::ThermalZone),
                        _ => unreachable!(),
                    };
                    let mut namespace = self.namespace.lock();
                    let level_path_reservation = new_scope.reserve_clone_scratch()?;
                    namespace.add_level_with_reservation(new_scope.try_clone()?, kind, level_reservation)?;
                    drop(level_path_reservation);
                    let object_reservation = self.reserve_namespace_object(&new_scope, 0)?;
                    let object_path_reservation = new_scope.reserve_clone_scratch()?;
                    namespace
                        .insert_if_mutable_with_reservation(
                            new_scope.try_clone()?,
                            object.wrap(),
                            object_reservation,
                        )
                        .map(|_| ())?;
                    drop(object_path_reservation);
                    drop(namespace);

                    let old_scope = context.enter_scope(new_scope)?;
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length)?;
                }
                Opcode::Processor => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;
                    let proc_id = context.next()?;
                    let pblk_address = context.next_u32()?;
                    let pblk_length = context.next()?;

                    let remaining_length = context.remaining_package_length(start_pc, pkg_length)?;

                    let new_scope = context.resolve_name(&name, &context.current_scope)?;
                    let object = Object::Processor { proc_id, pblk_address, pblk_length };
                    let level_reservation = self.reserve_namespace_object(&new_scope, 0)?;
                    let mut namespace = self.namespace.lock();
                    let level_path_reservation = new_scope.reserve_clone_scratch()?;
                    namespace.add_level_with_reservation(
                        new_scope.try_clone()?,
                        NamespaceLevelKind::Processor,
                        level_reservation,
                    )?;
                    drop(level_path_reservation);
                    let object_reservation = self.reserve_namespace_object(&new_scope, 0)?;
                    let object_path_reservation = new_scope.reserve_clone_scratch()?;
                    namespace
                        .insert_if_mutable_with_reservation(
                            new_scope.try_clone()?,
                            object.wrap(),
                            object_reservation,
                        )
                        .map(|_| ())?;
                    drop(object_path_reservation);
                    drop(namespace);

                    let old_scope = context.enter_scope(new_scope)?;
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length)?;
                }
                Opcode::PowerRes => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let name = context.namestring()?;
                    let system_level = context.next()?;
                    let resource_order = context.next_u16()?;

                    let remaining_length = context.remaining_package_length(start_pc, pkg_length)?;

                    let new_scope = context.resolve_name(&name, &context.current_scope)?;
                    let object = Object::PowerResource { system_level, resource_order };
                    let level_reservation = self.reserve_namespace_object(&new_scope, 0)?;
                    let mut namespace = self.namespace.lock();
                    let level_path_reservation = new_scope.reserve_clone_scratch()?;
                    namespace.add_level_with_reservation(
                        new_scope.try_clone()?,
                        NamespaceLevelKind::PowerResource,
                        level_reservation,
                    )?;
                    drop(level_path_reservation);
                    let object_reservation = self.reserve_namespace_object(&new_scope, 0)?;
                    let object_path_reservation = new_scope.reserve_clone_scratch()?;
                    namespace
                        .insert_if_mutable_with_reservation(
                            new_scope.try_clone()?,
                            object.wrap(),
                            object_reservation,
                        )
                        .map(|_| ())?;
                    drop(object_path_reservation);
                    drop(namespace);

                    let old_scope = context.enter_scope(new_scope)?;
                    context.start_new_block(BlockKind::Scope { old_scope }, remaining_length)?;
                }
                Opcode::Local(local) => {
                    let local = context.locals[local as usize].clone();
                    context.contribute_arg(Argument::Object(
                        self.wrap_object(Object::Reference { kind: ReferenceKind::Local, inner: local })?,
                    ));
                }
                Opcode::Arg(arg) => {
                    let arg = context.args[arg as usize].clone();
                    context.contribute_arg(Argument::Object(
                        self.wrap_object(Object::Reference { kind: ReferenceKind::Arg, inner: arg })?,
                    ));
                }
                Opcode::Store => context.start(OpInFlight::new(
                    Opcode::Store,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::SuperName],
                )),
                Opcode::CopyObject => context.start(OpInFlight::new(
                    Opcode::CopyObject,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::SimpleName],
                )),
                Opcode::RefOf => context.start(OpInFlight::new(Opcode::RefOf, &[ResolveBehaviour::SuperName])),
                Opcode::CondRefOf => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::SuperNameIfExists, ResolveBehaviour::Target],
                )),

                Opcode::DualNamePrefix
                | Opcode::MultiNamePrefix
                | Opcode::Digit(_)
                | Opcode::NameChar(_)
                | Opcode::RootChar
                | Opcode::ParentPrefixChar => {
                    context.current_block.pc -= 1;
                    let name = context.namestring()?;

                    let behaviour = context
                        .in_flight
                        .last()
                        .map(|op| op.resolve_behaviour())
                        .transpose()?
                        .unwrap_or(ResolveBehaviour::TermArg);
                    match behaviour {
                        // XXX: `NullName` is handled separately given its ambiguity with `Zero`
                        ResolveBehaviour::SimpleName | ResolveBehaviour::SuperName | ResolveBehaviour::Target => {
                            let object = context.search_with_flags(&self.namespace.lock(), &name);
                            match object {
                                Ok((_resolved_name, object, flags)) => {
                                    context.contribute_arg(Argument::Object(self.wrap_object(
                                        Object::Reference {
                                            kind: if flags.is_immutable() {
                                                ReferenceKind::ImmutableNamed
                                            } else {
                                                ReferenceKind::Named
                                            },
                                            inner: object,
                                        },
                                    )?));
                                }
                                Err(err) => Err(err)?,
                            }
                        }
                        ResolveBehaviour::SuperNameIfExists => {
                            let object = context.search_if_present(&self.namespace.lock(), &name);
                            match object {
                                Ok(Some((_resolved_name, object))) => {
                                    context.contribute_arg(Argument::Object(object));
                                }
                                Ok(None) => {
                                    let unresolved_name = context.render_name_object(&name)?;
                                    let reference = Object::Reference {
                                        kind: ReferenceKind::Unresolved,
                                        inner: unresolved_name,
                                    };
                                    context.contribute_arg(Argument::Object(self.wrap_object(reference)?));
                                }
                                Err(error) => Err(error)?,
                            }
                        }
                        ResolveBehaviour::TermArg => {
                            let object = context.search(&self.namespace.lock(), &name);
                            match object {
                                Ok((resolved_name, object)) => {
                                    if let Object::Method { flags, .. } | Object::NativeMethod { flags, .. } =
                                        *object
                                    {
                                        context.start_with_arguments(
                                            Opcode::InternalMethodCall,
                                            2 + flags.arg_count(),
                                            &[
                                                ResolveBehaviour::Placeholder,
                                                ResolveBehaviour::Placeholder,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                                ResolveBehaviour::TermArg,
                                            ],
                                            2,
                                            move |arguments| {
                                                arguments.push(Argument::Object(object));
                                                arguments.push(Argument::Namestring(resolved_name));
                                                Ok(())
                                            },
                                        )?
                                    } else if let Object::FieldUnit(ref field) = *object {
                                        let value = self.do_field_read(field)?;
                                        context.contribute_arg(Argument::Object(value));
                                    } else if let Object::BufferField { .. } = *object {
                                        let value = self.read_buffer_field(&object)?;
                                        context.contribute_arg(Argument::Object(value));
                                    } else {
                                        context.contribute_arg(Argument::Object(object));
                                    }
                                }
                                Err(err) => Err(err)?,
                            }
                        }
                        ResolveBehaviour::AsPackageElements => {
                            /*
                             * NameStrings in packages can refer to objects that are defined later,
                             * including ones in another table. Keep the name and the scope it
                             * appeared in, so resolution doesn't depend on table load order, or on
                             * the scope the package is eventually used from. Store it directly;
                             * indexing the package creates a reference to the element when needed.
                             */
                            let persistent_path_heap = name_path_heap_bytes(&name, &context.current_scope)?;
                            let heap_reservation = self.reserve_object_heap(persistent_path_heap, 0, 0)?;
                            let scope = match context.current_scope.try_clone() {
                                Ok(scope) => scope,
                                Err(error) => return Err(error),
                            };
                            let (name, _name_heap_reservation) = name.into_parts();
                            context.contribute_arg(Argument::Object(self.wrap_object_with_heap_reservation(
                                Object::NamePath { name, scope },
                                heap_reservation,
                            )?));
                        }
                        ResolveBehaviour::Placeholder => {
                            return Err(AmlError::InternalError(
                                "placeholder resolve behaviour reached name resolution".to_string(),
                            ));
                        }
                        ResolveBehaviour::MatchOpcode => Err(AmlError::InvalidArgType)?,
                    }
                }

                Opcode::Add
                | Opcode::Subtract
                | Opcode::Multiply
                | Opcode::ShiftLeft
                | Opcode::ShiftRight
                | Opcode::Mod
                | Opcode::Nand
                | Opcode::And
                | Opcode::Or
                | Opcode::Nor
                | Opcode::Xor
                | Opcode::Concat => {
                    context.start(OpInFlight::new(
                        opcode,
                        &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                    ));
                }

                Opcode::Divide => context.start(OpInFlight::new(
                    Opcode::Divide,
                    &[
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::Target,
                        ResolveBehaviour::Target,
                    ],
                )),
                Opcode::Increment | Opcode::Decrement => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName]))
                }
                Opcode::Not => context
                    .start(OpInFlight::new(Opcode::Not, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target])),
                Opcode::FindSetLeftBit | Opcode::FindSetRightBit => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target]))
                }
                // Resolve as a SuperName so BufferField and FieldUnit objects are not read eagerly
                // as TermArgs; DerefOf handles those reads itself when executing the opcode.
                Opcode::DerefOf => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::ConcatRes => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                )),
                Opcode::SizeOf => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Index => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                )),
                Opcode::Match => context.start(OpInFlight::new(
                    opcode,
                    &[
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::MatchOpcode,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::MatchOpcode,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::TermArg,
                    ],
                )),

                Opcode::CreateBitField
                | Opcode::CreateByteField
                | Opcode::CreateWordField
                | Opcode::CreateDWordField
                | Opcode::CreateQWordField => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg; 2]))
                }
                Opcode::CreateField => {
                    context.start(OpInFlight::new(Opcode::CreateField, &[ResolveBehaviour::TermArg; 3]))
                }

                Opcode::LNot => {
                    context.start(OpInFlight::new(Opcode::LNot, &[ResolveBehaviour::TermArg]));
                }

                Opcode::LAnd
                | Opcode::LOr
                | Opcode::LNotEqual
                | Opcode::LLessEqual
                | Opcode::LGreaterEqual
                | Opcode::LEqual
                | Opcode::LGreater
                | Opcode::LLess => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg; 2]));
                }

                Opcode::ToBuffer | Opcode::ToDecimalString | Opcode::ToHexString | Opcode::ToInteger => {
                    context.start(OpInFlight::new(opcode, &[ResolveBehaviour::TermArg, ResolveBehaviour::Target]))
                }
                Opcode::ToString => context.start(OpInFlight::new(
                    opcode,
                    &[ResolveBehaviour::TermArg, ResolveBehaviour::TermArg, ResolveBehaviour::Target],
                )),

                Opcode::ObjectType => context.start(OpInFlight::new(opcode, &[ResolveBehaviour::SuperName])),
                Opcode::Mid => context.start(OpInFlight::new(
                    Opcode::Mid,
                    &[
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::TermArg,
                        ResolveBehaviour::Target,
                    ],
                )),
                Opcode::If => {
                    let start_pc = context.current_block.pc;
                    let then_length = context.pkglength()?;
                    context.start_with_arguments(
                        Opcode::If,
                        3,
                        &[ResolveBehaviour::Placeholder, ResolveBehaviour::Placeholder, ResolveBehaviour::TermArg],
                        2,
                        |arguments| {
                            arguments.push(Argument::TrackedPc(start_pc));
                            arguments.push(Argument::PkgLength(then_length));
                            Ok(())
                        },
                    )?;
                }
                Opcode::Else => return Err(AmlError::ElseFoundWithoutCorrespondingIf),
                Opcode::While => {
                    let start_pc = context.current_block.pc;
                    let pkg_length = context.pkglength()?;
                    let remaining_length = context.remaining_package_length(start_pc, pkg_length)?;
                    context.start_new_block(
                        BlockKind::While {
                            start_pc: context.current_block.pc,
                            start_nanos: self.handler.nanos_since_boot(),
                        },
                        remaining_length,
                    )?;
                    context.start(OpInFlight::new(Opcode::While, &[ResolveBehaviour::TermArg]));
                }
                Opcode::Continue => {
                    self.continue_loop(&mut context)?;
                }
                Opcode::Break => {
                    if let BlockKind::While { .. } = &context.current_block.kind {
                        context.current_block = context.pop_block()?;
                    } else {
                        loop {
                            let Some(block) = context.block_stack.pop() else {
                                Err(AmlError::BreakOutsideOfWhile)?
                            };
                            if let BlockKind::While { .. } = block.kind {
                                context.current_block = context.pop_block()?;
                                break;
                            }
                        }
                    }
                }
                Opcode::Return => context.start(OpInFlight::new(Opcode::Return, &[ResolveBehaviour::TermArg])),
                Opcode::Noop => {}
                Opcode::Breakpoint => {
                    self.handler.breakpoint();
                }

                Opcode::InternalMethodCall => {
                    return Err(AmlError::InternalError(
                        "internal method call opcode appeared in AML byte stream".to_string(),
                    ));
                }
            }
        }
    }

    fn parse_field_list(
        &self,
        context: &mut MethodContext,
        kind: FieldUnitKind,
        start_pc: usize,
        pkg_length: usize,
        mut flags: u8,
    ) -> Result<(), AmlError> {
        const RESERVED_FIELD: u8 = 0x00;
        const ACCESS_FIELD: u8 = 0x01;
        const CONNECT_FIELD: u8 = 0x02;
        const EXTENDED_ACCESS_FIELD: u8 = 0x03;

        if FieldFlags(flags).lock_rule() {
            self.global_lock_dependency.store(true, Ordering::Release);
        }
        let mut field_offset = 0;

        let package_end = start_pc
            .checked_add(pkg_length)
            .filter(|end| *end <= context.current_block.stream().len())
            .ok_or(AmlError::InvalidPackageLength)?;
        if context.current_block.pc > package_end {
            return Err(AmlError::InvalidPackageLength);
        }
        while context.current_block.pc < package_end {
            match context.next()? {
                RESERVED_FIELD => {
                    let length = context.pkglength()?;
                    field_offset += length;
                }
                ACCESS_FIELD => {
                    /*
                     * These aren't actually fields themselves, but are created by `AccessAs` AML
                     * elements. They change the access type and attributes for remaining fields in
                     * the list.
                     */
                    let access_type = context.next()?;
                    let _access_attrib = context.next()?;
                    flags.set_bits(0..4, access_type);
                }
                EXTENDED_ACCESS_FIELD => {
                    let access_type = context.next()?;
                    let _extended_access_attrib = context.next()?;
                    let _access_length = context.next()?;
                    flags.set_bits(0..4, access_type);
                    warn!("Ignoring extended attributes and length in ExtendedAccessField");
                }
                CONNECT_FIELD => {
                    return Err(AmlError::LibUnimplemented);
                }
                _ => {
                    context.current_block.pc -= 1;
                    // TODO: this should only ever be a nameseg really...
                    let field_name = context.namestring()?;
                    let field_length = context.pkglength()?;
                    let field_name = context.resolve_name(&field_name, &context.current_scope)?;
                    let reservation = self.reserve_namespace_object(&field_name, 0)?;

                    let field = Object::FieldUnit(FieldUnit {
                        kind: kind.clone(),
                        bit_index: field_offset,
                        bit_length: field_length,
                        flags: FieldFlags(flags),
                    });
                    self.insert_metered_name(field_name, field.wrap(), reservation)?;

                    field_offset += field_length;
                }
            }
            if context.current_block.pc > package_end {
                return Err(AmlError::InvalidPackageLength);
            }
        }

        Ok(())
    }

    fn do_binary_maths(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op[0..3] => [Argument::Object(left), Argument::Object(right), Argument::Object(target)]);
        let target2 = if op.op == Opcode::Divide { Some(&op.arguments[3]) } else { None };

        let left = left.clone().unwrap_transparent_reference()?.to_integer(self.integer_size)?;
        let right = right.clone().unwrap_transparent_reference()?.to_integer(self.integer_size)?;
        if right == 0 && matches!(op.op, Opcode::Divide | Opcode::Mod) {
            return Err(AmlError::DivisionByZero);
        }

        let result = match op.op {
            Opcode::Add => left.wrapping_add(right),
            Opcode::Subtract => left.wrapping_sub(right),
            Opcode::Multiply => left.wrapping_mul(right),
            Opcode::Divide => {
                if let Some(Argument::Object(remainder)) = target2 {
                    self.do_store(
                        remainder.clone(),
                        self.wrap_object(Object::Integer(left.wrapping_rem(right)))?,
                    )?;
                }
                left.wrapping_div_euclid(right)
            }
            Opcode::ShiftLeft => left.wrapping_shl(right as u32),
            Opcode::ShiftRight => left.wrapping_shr(right as u32),
            Opcode::Mod => left.wrapping_rem(right),
            Opcode::Nand => !(left & right),
            Opcode::And => left & right,
            Opcode::Or => left | right,
            Opcode::Nor => !(left | right),
            Opcode::Xor => left ^ right,
            _ => {
                return Err(AmlError::InternalError("invalid binary math operation".to_string()));
            }
        };

        let result = self.wrap_object(Object::Integer(result))?;
        let result = self.do_store(target.clone(), result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_unary_maths(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(operand)]);
        let operand = operand.clone().unwrap_transparent_reference()?.as_integer()?;

        let result = match op.op {
            Opcode::FindSetLeftBit => {
                if operand == 0 {
                    0
                } else {
                    /*
                     * This is a particularly important place to respect the integer width as set
                     * by the DSDT revision.
                     */
                    match self.integer_size {
                        IntegerSize::FourBytes => ((operand as u32).leading_zeros() + 1) as u64,
                        IntegerSize::EightBytes => (operand.leading_zeros() + 1) as u64,
                    }
                }
            }
            Opcode::FindSetRightBit => {
                if operand == 0 {
                    0
                } else {
                    (operand.trailing_zeros() + 1) as u64
                }
            }
            Opcode::Not => {
                if operand == 0 {
                    u64::MAX
                } else {
                    0
                }
            }
            _ => {
                return Err(AmlError::InternalError("invalid unary math operation".to_string()));
            }
        };

        context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(result))?));
        context.retire_op(op);
        Ok(())
    }

    fn do_logical_op(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        if op.op == Opcode::LNot {
            extract_args!(op => [Argument::Object(operand)]);
            let operand = operand.clone().unwrap_transparent_reference()?.as_integer()?;
            let result = if operand == 0 { u64::MAX } else { 0 };

            context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(result))?));
            context.retire_op(op);
            return Ok(());
        }

        extract_args!(op => [Argument::Object(left), Argument::Object(right)]);
        let left = left.clone().unwrap_transparent_reference()?;
        let right = right.clone().unwrap_transparent_reference()?;

        let mut int_size = self.integer_size;
        let mut _temporary_buffer_reservation = None;
        let converted_right = match *left {
            Object::Buffer(_) if right.typ() != ObjectType::Buffer => {
                let buffer_bytes = right.buffer_conversion_size(self.integer_size)?;
                _temporary_buffer_reservation = Some(self.reserve_temporary_buffer(buffer_bytes)?);
                Some(Object::Buffer(right.to_buffer(self.integer_size)?))
            }
            _ => None,
        };

        // Validate matching operands without cloning scalar or string values.
        let right: &Object = match *left {
            Object::Integer(_) => {
                right.as_integer()?;
                &right
            }
            Object::String(_) => {
                right.as_string()?;
                &right
            }
            Object::Buffer(_) => {
                // When doing && or ||, uACPI and NT only compare the first 4 bytes of a buffer.
                int_size = IntegerSize::FourBytes;
                if right.typ() == ObjectType::Buffer {
                    &right
                } else {
                    converted_right.as_ref().expect("converted buffer is initialized")
                }
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::LogicalOp, typ: left.typ() })?,
        };

        let ordering = left.aml_cmp(right);
        let result = match op.op {
            Opcode::LAnd => (left.to_integer(int_size)? > 0) && (right.to_integer(int_size)? > 0),
            Opcode::LOr => (left.to_integer(int_size)? > 0) || (right.to_integer(int_size)? > 0),
            Opcode::LNotEqual => ordering?.is_ne(),
            Opcode::LLessEqual => ordering?.is_le(),
            Opcode::LGreaterEqual => ordering?.is_ge(),
            Opcode::LEqual => ordering?.is_eq(),
            Opcode::LGreater => ordering?.is_gt(),
            Opcode::LLess => ordering?.is_lt(),
            _ => {
                return Err(AmlError::InternalError("invalid logical operation".to_string()));
            }
        };
        let result = if result { Object::Integer(u64::MAX) } else { Object::Integer(0) };

        context.contribute_arg(Argument::Object(self.wrap_object(result)?));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_buffer(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(operand), Argument::Object(target)]);
        let operand = operand.clone().unwrap_transparent_reference()?;

        let (result, heap_reservation) = match *operand {
            Object::Buffer(ref bytes) => {
                let buffer_bytes = u64::try_from(bytes.len()).map_err(|_| AmlError::BudgetExceeded)?;
                let heap_bytes = buffer_bytes
                    .checked_add(if bytes.is_empty() { 0 } else { 24 })
                    .ok_or(AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                (Object::Buffer(try_vec_from_slice(bytes)?), reservation)
            }
            Object::Integer(value) => {
                let buffer_bytes = if self.integer_size == IntegerSize::EightBytes { 8_u64 } else { 4_u64 };
                let heap_bytes = buffer_bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                let bytes = if self.integer_size == IntegerSize::EightBytes {
                    try_vec_from_slice(&value.to_le_bytes())?
                } else {
                    try_vec_from_slice(&(value as u32).to_le_bytes())?
                };
                (Object::Buffer(bytes), reservation)
            }
            Object::String(ref value) => {
                // XXX: an empty string is converted to an empty buffer, *without* the null-terminator
                if value.is_empty() {
                    let reservation = self.reserve_object_heap(0, 0, 0)?;
                    (Object::Buffer(crate::aml_vec![]), reservation)
                } else {
                    let buffer_bytes = u64::try_from(value.len())
                        .map_err(|_| AmlError::BudgetExceeded)?
                        .checked_add(1)
                        .ok_or(AmlError::BudgetExceeded)?;
                    let heap_bytes = buffer_bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?;
                    let reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                    let mut bytes = try_vec_with_capacity(value.len() + 1)?;
                    bytes.extend_from_slice(value.as_bytes());
                    bytes.push(b'\0');
                    (Object::Buffer(bytes), reservation)
                }
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ToBuffer, typ: operand.typ() })?,
        };
        let result = self.wrap_object_with_heap_reservation(result, heap_reservation)?;

        let result = self.do_store(target.clone(), result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_integer(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(operand), Argument::Object(target)]);
        let operand = operand.clone().unwrap_transparent_reference()?;

        let result = self.wrap_object(Object::Integer(operand.to_integer(self.integer_size)?))?;
        let result = self.do_store(target.clone(), result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_string(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(source), Argument::Object(length), Argument::Object(target)]);
        let source = source.clone().unwrap_transparent_reference()?;
        let source = source.as_buffer()?;
        let length = length.clone().unwrap_transparent_reference()?.as_integer()? as usize;

        let (result, heap_reservation) = if source.is_empty() {
            let reservation = self.reserve_object_heap(0, 0, 0)?;
            (Object::String(String::new()), reservation)
        } else {
            let mut buffer = source.split_inclusive(|b| *b == b'\0').next().unwrap();
            if length < usize::MAX {
                buffer = &buffer[0..usize::min(length, buffer.len())];
            }
            let string = str::from_utf8(buffer).map_err(|_| AmlError::InvalidOperationOnObject {
                op: Operation::ToString,
                typ: ObjectType::Buffer,
            })?;
            let heap_bytes = string
                .len()
                .checked_add(if string.is_empty() { 0 } else { 24 })
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| AmlError::BudgetExceeded)?;
            let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
            (Object::String(try_string_from_str(string)?), reservation)
        };
        let result = self.wrap_object_with_heap_reservation(result, heap_reservation)?;

        let result = self.do_store(target.clone(), result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    /// Perform a `ToDecimalString` or `ToHexString` operation
    fn do_to_dec_hex_string(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(operand), Argument::Object(target)]);
        let operand = operand.clone().unwrap_transparent_reference()?;

        let (result, heap_reservation) = match *operand {
            Object::String(ref value) => {
                let heap_bytes = string_heap_bytes(value.len()).ok_or_else(|| AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                (Object::String(try_string_from_str(value)?), reservation)
            }
            Object::Integer(value) => {
                let hexadecimal = match op.op {
                    Opcode::ToDecimalString => false,
                    Opcode::ToHexString => true,
                    _ => return Err(AmlError::InternalError("invalid integer string conversion".to_string())),
                };
                let capacity = integer_string_capacity(value, hexadecimal);
                let heap_bytes = string_heap_bytes(capacity).ok_or_else(|| AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                (Object::String(try_format_integer_string(value, hexadecimal)?), reservation)
            }
            Object::Buffer(ref bytes) => {
                let hexadecimal = match op.op {
                    Opcode::ToDecimalString => false,
                    Opcode::ToHexString => true,
                    _ => return Err(AmlError::InternalError("invalid buffer string conversion".to_string())),
                };
                let capacity =
                    buffer_string_capacity(bytes.len(), hexadecimal).ok_or_else(|| AmlError::BudgetExceeded)?;
                let heap_bytes = string_heap_bytes(capacity).ok_or_else(|| AmlError::BudgetExceeded)?;
                let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                (Object::String(try_format_buffer_string(bytes, hexadecimal)?), reservation)
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ToDecOrHexString, typ: operand.typ() })?,
        };
        let result = self.wrap_object_with_heap_reservation(result, heap_reservation)?;

        let result = self.do_store(target.clone(), result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_mid(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(source), Argument::Object(index), Argument::Object(length), Argument::Object(target)]);
        let index = index.clone().unwrap_transparent_reference()?.as_integer()? as usize;
        let length = length.clone().unwrap_transparent_reference()?.as_integer()? as usize;

        let (result, heap_reservation) = match **source {
            Object::String(ref string) => {
                if index >= string.len() {
                    let reservation = self.reserve_object_heap(0, 0, 0)?;
                    (Object::String(String::new()), reservation)
                } else {
                    let upper = index.saturating_add(length).min(string.len());
                    let chars = string.get(index..upper).ok_or_else(|| AmlError::InvalidOperationOnObject {
                        op: Operation::Mid,
                        typ: ObjectType::String,
                    })?;
                    let heap_bytes = chars
                        .len()
                        .checked_add(if chars.is_empty() { 0 } else { 24 })
                        .and_then(|bytes| u64::try_from(bytes).ok())
                        .ok_or_else(|| AmlError::BudgetExceeded)?;
                    let reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                    (Object::String(try_string_from_str(chars)?), reservation)
                }
            }
            Object::Buffer(ref buffer) => {
                if index >= buffer.len() {
                    let reservation = self.reserve_object_heap(0, 0, 0)?;
                    (Object::Buffer(crate::aml_vec![]), reservation)
                } else {
                    let upper = index.saturating_add(length).min(buffer.len());
                    let bytes = &buffer[index..upper];
                    let buffer_bytes = u64::try_from(bytes.len()).map_err(|_| AmlError::BudgetExceeded)?;
                    let heap_bytes = buffer_bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?;
                    let reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                    (Object::Buffer(try_vec_from_slice(bytes)?), reservation)
                }
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::Mid, typ: source.typ() })?,
        };
        let result = self.wrap_object_with_heap_reservation(result, heap_reservation)?;

        self.do_store(target.clone(), result.clone())?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_concat(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(source1), Argument::Object(source2), Argument::Object(target)]);
        let source1 = source1.clone().unwrap_transparent_reference()?;
        let source2 = source2.clone().unwrap_transparent_reference()?;

        let mut heap_reservation = None;
        let result = match source1.typ() {
            ObjectType::Integer | ObjectType::Buffer => {
                let buffer = try_concat_buffer_values(&source1, &source2, self.integer_size, |buffer_bytes| {
                    let heap_bytes = buffer_bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?;
                    heap_reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
                    Ok(())
                })?;
                self.wrap_object_with_heap_reservation(Object::Buffer(buffer), heap_reservation.take())?
            }
            _ => {
                let string = try_concat_string_values(&source1, &source2, |heap_bytes| {
                    heap_reservation = self.reserve_object_heap(heap_bytes, 0, 0)?;
                    Ok(())
                })?;
                self.wrap_object_with_heap_reservation(Object::String(string), heap_reservation.take())?
            }
        };

        let result = self.do_store(target.clone(), result)?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    fn do_from_bcd(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(value)]);
        let mut value = value.clone().unwrap_transparent_reference()?.as_integer()?;

        let mut result = 0;
        let mut i = 1;
        while value > 0 {
            result += (value & 0x0f) * i;
            i *= 10;
            value >>= 4;
        }

        context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(result))?));
        context.retire_op(op);
        Ok(())
    }

    fn do_to_bcd(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(value)]);
        let mut value = value.clone().unwrap_transparent_reference()?.as_integer()?;

        let mut result = 0;
        let mut i = 0;
        while value > 0 {
            result |= (value % 10) << (4 * i);
            value /= 10;
            i += 1;
        }

        context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(result))?));
        context.retire_op(op);
        Ok(())
    }

    fn do_size_of(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(object)]);
        let object = self.resolve_name_path(object.clone())?;

        let result = match *object {
            Object::Buffer(ref buffer) => buffer.len(),
            Object::String(ref str) => str.len(),
            Object::Package(ref package) => package.len(),
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::SizeOf, typ: object.typ() })?,
        };

        context.contribute_arg(Argument::Object(self.wrap_object(Object::Integer(result as u64))?));
        context.retire_op(op);
        Ok(())
    }

    fn do_index(&self, context: &mut MethodContext, op: OpInFlight) -> Result<(), AmlError> {
        extract_args!(op => [Argument::Object(object), Argument::Object(index_value), Argument::Object(target)]);
        let object = object.clone().unwrap_transparent_reference()?;
        let index_value = index_value.clone().unwrap_transparent_reference()?.as_integer()?;

        let result = match *object {
            Object::Buffer(ref buffer) => {
                if index_value as usize >= buffer.len() {
                    Err(AmlError::IndexOutOfBounds)?
                }

                self.wrap_object(Object::Reference {
                    kind: ReferenceKind::RefOf,
                    inner: self.wrap_object(Object::BufferField {
                        buffer: object.clone(),
                        offset: index_value as usize * 8,
                        length: 8,
                    })?,
                })?
            }
            Object::String(ref string) => {
                if index_value as usize >= string.len() {
                    Err(AmlError::IndexOutOfBounds)?
                }

                self.wrap_object(Object::Reference {
                    kind: ReferenceKind::RefOf,
                    inner: self.wrap_object(Object::BufferField {
                        buffer: object.clone(),
                        offset: index_value as usize * 8,
                        length: 8,
                    })?,
                })?
            }
            Object::Package(ref package) => {
                let Some(element) = package.get(index_value as usize) else { Err(AmlError::IndexOutOfBounds)? };
                self.wrap_object(Object::Reference { kind: ReferenceKind::RefOf, inner: element.clone() })?
            }
            _ => Err(AmlError::IndexOutOfBounds)?,
        };

        self.do_store(target.clone(), result.clone())?;
        context.contribute_arg(Argument::Object(result));
        context.retire_op(op);
        Ok(())
    }

    /// Resolve an object to the value it refers to, looking up `NamePath`s (i.e. names that were
    /// used as package elements) in the namespace. Objects that aren't references are returned
    /// unchanged.
    fn resolve_name_path(&self, object: WrappedObject) -> Result<WrappedObject, AmlError> {
        let mut object = object.unwrap_reference()?;

        /*
         * A name can resolve to another unresolved name, and firmware is free to make that
         * circular. Bound the number of indirections we follow rather than recursing.
         */
        for _ in 0..MAX_NAME_PATH_INDIRECTIONS {
            let (name, scope) = match &*object {
                Object::NamePath { name, scope } => (name, scope),
                _ => return Ok(object),
            };
            let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
            let name = MeteredNameString::clone_name_metered(name, meter.clone())?;
            let scope = MeteredNameString::clone_name_metered(scope, meter)?;
            let (_, resolved) = self.search_name(&name, &scope)?;
            object = resolved.unwrap_reference()?;
        }

        Err(AmlError::NameResolutionLoop)
    }

    fn object_type(&self, object: WrappedObject) -> Result<u64, AmlError> {
        let object = self.resolve_name_path(object)?;

        // TODO: this should technically support scopes as well - this is less easy
        // (they should return `0`)
        Ok(match object.typ() {
            ObjectType::Uninitialized => 0,
            ObjectType::Integer => 1,
            ObjectType::String => 2,
            ObjectType::Buffer => 3,
            ObjectType::Package => 4,
            ObjectType::FieldUnit => 5,
            ObjectType::Device => 6,
            ObjectType::Event => 7,
            ObjectType::Method => 8,
            ObjectType::Mutex => 9,
            ObjectType::OpRegion => 10,
            ObjectType::PowerResource => 11,
            ObjectType::Processor => 12,
            ObjectType::ThermalZone => 13,
            ObjectType::BufferField => 14,
            // XXX: 15 is reserved
            ObjectType::Debug => 16,
            ObjectType::RawDataBuffer => 17,
            ObjectType::Reference => unreachable!(),
        })
    }

    fn do_deref_of(&self, object: WrappedObject, current_scope: &AmlName) -> Result<WrappedObject, AmlError> {
        let object = self.resolve_name_path(object)?;
        match &*object {
            Object::BufferField { .. } => self.read_buffer_field(&object),
            Object::FieldUnit(field) => self.do_field_read(field),
            Object::String(path) => {
                let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
                let path = MeteredNameString::parse_name_metered(path, meter)?;
                let (_, object) = self.search_name(&path, current_scope)?;
                Ok(object.clone())
            }
            Object::Reference { .. } => unreachable!(),
            _ => Ok(object),
        }
    }

    /// Perform a store of `object` into `target`, matching the expected behaviour of `DefStore`,
    /// which depends on the target:
    ///    - Locals and Index references are overwritten, unless the Local contains a reference
    ///      (e.g. obtained via `RefOf`), in which case a store is performed to the referenced
    ///      object with implicit casting.
    ///    - Args are overwritten, unless they contain a reference, in which case the referenced
    ///      object is *usually* overwritten. References from Arg to Local without an intermediate
    ///      `RefOf` cause the Arg to be overwritten instead of the Local (see
    ///      [issue #313](https://github.com/rust-osdev/acpi/issues/313)).
    ///    - There is an exception to this, needed to match the Windows NT interpreter: if an Arg
    ///      referencing a Local is found whilst unwrapping a real reference (e.g. a Local holding
    ///      `RefOf(Arg0)`, where `Arg0` was itself passed a `Local` directly) and the final result
    ///      of unwrapping is a string, the string is modified instead of the Arg. This does not
    ///      apply if the Arg is the direct target of the store and only applies to Strings. (This
    ///      sounds odd, but is the Windows way)
    ///    - Named objects are stored into, with implicit casting
    ///
    /// This is complex so may be explained better by `tests/store.asl` and
    /// [`Object::unwrap_ref_for_store`]!
    fn do_store(&self, target: WrappedObject, object: WrappedObject) -> Result<WrappedObject, AmlError> {
        let object = object.unwrap_transparent_reference()?;
        match &*target {
            Object::Reference { .. } => {
                let (target_object, implicit_cast_reqd) = target.clone().unwrap_ref_for_store()?;

                if !implicit_cast_reqd {
                    let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
                    if object.would_create_reference_cycle(&target_object, meter.as_ref())? {
                        return Err(AmlError::ReferenceCycle);
                    }
                    let copy = self.try_clone_object(&object)?;
                    let token = self.object_token.lock();
                    unsafe {
                        target_object.replace_metered(copy, &token);
                    }
                } else {
                    match &*target_object {
                        Object::Integer(_) | Object::String(_) | Object::Buffer(_) => {
                            let replacement = self.try_implicit_cast_copy(target_object.typ(), &object)?;
                            let token = self.object_token.lock();
                            unsafe {
                                target_object.replace_metered(replacement, &token);
                            }
                        }
                        Object::BufferField { .. } => {
                            let token = self.object_token.lock();
                            let target_object = unsafe { target_object.gain_mut(&token) };
                            match unsafe { object.gain_mut(&token) } {
                                Object::Integer(value) => {
                                    target_object.write_buffer_field(&value.to_le_bytes(), &token)?
                                }
                                Object::Buffer(value) => {
                                    target_object.write_buffer_field(value.as_slice(), &token)?
                                }
                                _ => {
                                    return Err(AmlError::ObjectNotOfExpectedType {
                                        expected: ObjectType::Integer,
                                        got: object.typ(),
                                    });
                                }
                            }
                        }

                        /*
                         * TODO: stores to fields with BufferAcc can return a different value to the
                         * object stored. This is used for complex field types with a
                         * write-then-read pattern. We should perform a read in those cases and
                         * return that instead.
                         */
                        Object::FieldUnit(field_unit) => self.do_field_write(field_unit, object.clone())?,

                        _ => {
                            return Err(AmlError::InvalidOperationOnObject {
                                op: Operation::Store,
                                typ: target_object.typ(),
                            });
                        }
                    }
                }
            }
            Object::Debug => self.handler.handle_debug(&object),
            Object::Integer(0) => {} // Store to NullName
            _ => return Err(AmlError::InvalidOperationOnObject { op: Operation::Store, typ: target.typ() }),
        }

        Ok(object)
    }

    /// Copy `object` into `target`, matching the expected behaviour of `DefCopyObject`, which
    /// depends on the object referenced:
    ///    - Locals are overwritten
    ///    - Args are overwritten, unless they are references, in which case the referenced object is overwritten
    ///    - Objects referenced by name are overwritten
    ///    - Index references cause the object at the index to be overwritten
    ///    - Other reference operations are not allowed
    fn do_copy_object(&self, target: WrappedObject, object: WrappedObject) -> Result<(), AmlError> {
        let Object::Reference { kind, ref inner } = *target else {
            return Err(AmlError::InternalError("Target of CopyObject must be a reference".to_string()));
        };
        let object = object.clone().unwrap_transparent_reference()?;

        let dst = match kind {
            ReferenceKind::Named | ReferenceKind::Local => inner.clone().unwrap_transparent_reference()?,
            ReferenceKind::ImmutableNamed => return Err(AmlError::StoreToInvalidReferenceType),
            ReferenceKind::Arg => {
                if let Object::Reference { kind: _, inner: ref inner_inner } = **inner {
                    inner_inner.clone()
                } else {
                    inner.clone().unwrap_transparent_reference()?
                }
            }
            ReferenceKind::Index => return Err(AmlError::LibUnimplemented),
            ReferenceKind::RefOf | ReferenceKind::Unresolved => return Err(AmlError::StoreToInvalidReferenceType),
        };

        let meter = self.execution_meter.lock().as_ref().map(Arc::clone);
        if object.would_create_reference_cycle(&dst, meter.as_ref())? {
            return Err(AmlError::ReferenceCycle);
        }
        let copy = self.try_clone_object(&object)?;
        let token = self.object_token.lock();
        unsafe {
            dst.replace_metered(copy, &token);
        }

        Ok(())
    }

    /// Do a read from a field by performing one or more well-formed accesses to the underlying
    /// operation regions, and then shifting and masking the resulting value as appropriate. Will
    /// return either an `Integer` or `Buffer` as appropriate, guided by the size of the field
    /// and expected integer size (as per the DSDT revision).
    fn field_unit(object: &WrappedObject) -> Result<&FieldUnit, AmlError> {
        let Object::FieldUnit(field) = &**object else {
            return Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::FieldUnit, got: object.typ() });
        };
        Ok(field)
    }

    fn normal_field_unit(object: &WrappedObject) -> Result<&FieldUnit, AmlError> {
        let field = Self::field_unit(object)?;
        if !matches!(&field.kind, FieldUnitKind::Normal { .. }) {
            return Err(AmlError::InternalError("expected a normal AML FieldUnit".to_string()));
        }
        Ok(field)
    }

    fn operation_region(object: &WrappedObject) -> Result<&OpRegion, AmlError> {
        let Object::OpRegion(region) = &**object else {
            return Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::OpRegion, got: object.typ() });
        };
        Ok(region)
    }

    fn with_global_lock<T>(
        &self,
        required: bool,
        operation: impl FnOnce() -> Result<T, AmlError>,
    ) -> Result<T, AmlError> {
        if !required {
            return operation();
        }
        self.acquire_global_lock(u16::MAX)?;
        let operation_result = operation();
        let release_result = self.release_global_lock();
        match (operation_result, release_result) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(_), Err(_)) => Err(AmlError::InternalError(
                "locked field operation and Global Lock release both failed".to_string(),
            )),
        }
    }

    fn do_field_read(&self, field: &FieldUnit) -> Result<WrappedObject, AmlError> {
        self.with_global_lock(field.flags.lock_rule(), || self.do_field_read_unlocked(field))
    }

    fn do_field_read_unlocked(&self, field: &FieldUnit) -> Result<WrappedObject, AmlError> {
        let needs_buffer = field.bit_length > (self.integer_size as usize * 8);
        let access_width_bits = field.flags.access_type_bytes()? * 8;

        trace!("AML field read. Field = {:?}", field);

        enum Output {
            Integer([u8; 8]),
            Buffer(Vec<u8>),
        }
        let mut heap_reservation = None;
        let mut output = if needs_buffer {
            let bytes = field.bit_length.next_multiple_of(8);
            let buffer_bytes = u64::try_from(bytes).map_err(|_| AmlError::BudgetExceeded)?;
            let heap_bytes = buffer_bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?;
            heap_reservation = self.reserve_object_heap(heap_bytes, 0, buffer_bytes)?;
            let mut buffer = try_vec_with_capacity(bytes)?;
            buffer.resize(bytes, 0);
            Output::Buffer(buffer)
        } else {
            Output::Integer([0; 8])
        };
        let output_bytes = match &mut output {
            Output::Buffer(bytes) => bytes.as_mut_slice(),
            Output::Integer(value) => value,
        };

        let (read_region, index_field_idx) = match field.kind {
            FieldUnitKind::Normal { ref region } => (Self::operation_region(region)?, 0),
            FieldUnitKind::Bank { ref region, ref bank, bank_value } => {
                let bank = Self::normal_field_unit(bank)?;
                self.do_field_write(bank, self.wrap_object(Object::Integer(bank_value))?)?;
                (Self::operation_region(region)?, 0)
            }
            FieldUnitKind::Index { ref index, ref data } => {
                let _index = Self::normal_field_unit(index)?;
                let data = Self::normal_field_unit(data)?;
                let FieldUnitKind::Normal { region } = &data.kind else {
                    return Err(AmlError::InternalError("IndexField data is not a normal field".to_string()));
                };
                let reg_idx = field.bit_index / 8;
                (Self::operation_region(region)?, reg_idx)
            }
        };

        /*
         * TODO: it might be worth having a fast path here for reads that don't do weird
         * unaligned accesses, which I'm guessing might be relatively common on real
         * hardware? Eg. single native read + mask
         */

        /*
         * Break the field read into native reads that respect the region's access width.
         * Copy each potentially-unaligned part into the destination's bit range.
         */
        let native_accesses_needed =
            (field.bit_length + (field.bit_index % access_width_bits)).div_ceil(access_width_bits);
        let mut read_so_far = 0;
        for i in 0..native_accesses_needed {
            // Advance the read pointer. For Index fields, this also means updating the Index
            // register.
            let aligned_offset = match field.kind {
                FieldUnitKind::Normal { .. } | FieldUnitKind::Bank { .. } => {
                    object::align_down(field.bit_index + i * access_width_bits, access_width_bits)
                }
                FieldUnitKind::Index { ref index, ref data } => {
                    // Update index register
                    let index = Self::normal_field_unit(index)?;
                    let data = Self::normal_field_unit(data)?;
                    self.do_field_write(
                        index,
                        self.wrap_object(Object::Integer((index_field_idx + i * (access_width_bits / 8)) as u64))?,
                    )?;

                    // The offset is always that of the data register, as we always read from the
                    // base of the data register.
                    data.bit_index
                }
            };

            let raw = self.do_native_region_read(read_region, aligned_offset / 8, access_width_bits / 8)?;
            let src_index = if i == 0 { field.bit_index % access_width_bits } else { 0 };
            let remaining_length = field.bit_length - read_so_far;
            let length = if i == 0 {
                usize::min(remaining_length, access_width_bits - (field.bit_index % access_width_bits))
            } else {
                usize::min(remaining_length, access_width_bits)
            };

            object::copy_bits(&raw.to_le_bytes(), src_index, output_bytes, read_so_far, length);
            read_so_far += length;
        }

        match output {
            Output::Buffer(bytes) => {
                self.wrap_object_with_heap_reservation(Object::Buffer(bytes), heap_reservation)
            }
            Output::Integer(value) => self.wrap_object(Object::Integer(u64::from_le_bytes(value))),
        }
    }

    fn do_field_write(&self, field: &FieldUnit, value: WrappedObject) -> Result<(), AmlError> {
        self.with_global_lock(field.flags.lock_rule(), || self.do_field_write_unlocked(field, value))
    }

    fn do_field_write_unlocked(&self, field: &FieldUnit, value: WrappedObject) -> Result<(), AmlError> {
        trace!("AML field write. Field = {:?}. Value = {}", field, value);

        let value_bytes = match &*value {
            Object::Integer(value) => &value.to_le_bytes() as &[u8],
            Object::Buffer(bytes) => bytes,
            _ => Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::Integer, got: value.typ() })?,
        };
        let access_width_bits = field.flags.access_type_bytes()? * 8;

        // In this tuple:
        // - write_region is the region that the data will be written to.
        // - index_field_idx is the initial index to write into the Index register of an index
        //   field. For all other field types it is unused and set to zero.
        let (write_region, index_field_idx) = match field.kind {
            FieldUnitKind::Normal { ref region } => (Self::operation_region(region)?, 0),
            FieldUnitKind::Bank { ref region, ref bank, bank_value } => {
                let bank = Self::normal_field_unit(bank)?;
                self.do_field_write(bank, self.wrap_object(Object::Integer(bank_value))?)?;
                (Self::operation_region(region)?, 0)
            }
            FieldUnitKind::Index { ref index, ref data } => {
                let _index = Self::normal_field_unit(index)?;
                let data = Self::normal_field_unit(data)?;
                let FieldUnitKind::Normal { region: data_region } = &data.kind else {
                    return Err(AmlError::InternalError("IndexField data is not a normal field".to_string()));
                };
                let reg_idx = field.bit_index / 8;
                (Self::operation_region(data_region)?, reg_idx)
            }
        };

        // TODO: if the region wants locking, do that

        // TODO: maybe also a fast path for writes

        let native_accesses_needed =
            (field.bit_length + (field.bit_index % access_width_bits)).div_ceil(access_width_bits);
        let mut written_so_far = 0;

        for i in 0..native_accesses_needed {
            // Advance the write pointer... For normal and bank fields this is straightforward. For
            // Index fields, this involves updating the index register.
            let aligned_offset = match field.kind {
                FieldUnitKind::Normal { .. } | FieldUnitKind::Bank { .. } => {
                    object::align_down(field.bit_index + i * access_width_bits, access_width_bits)
                }
                FieldUnitKind::Index { ref index, ref data } => {
                    // Update index register
                    let index = Self::normal_field_unit(index)?;
                    let data = Self::normal_field_unit(data)?;
                    self.do_field_write(
                        index,
                        self.wrap_object(Object::Integer((index_field_idx + i * (access_width_bits / 8)) as u64))?,
                    )?;

                    // The offset is always that of the data register, as we always read from the
                    // base of the data register.
                    data.bit_index
                }
            };
            let dst_index = if i == 0 { field.bit_index % access_width_bits } else { 0 };

            /*
             * If we're not going to write a whole native access, respect the field's
             * update rule. If we're meant to preserve the surrounding bits, we need to do
             * a read first.
             */
            let mut bytes = if dst_index > 0 || (field.bit_length - written_so_far) < access_width_bits {
                match field.flags.update_rule()? {
                    FieldUpdateRule::Preserve => self
                        .do_native_region_read(write_region, aligned_offset / 8, access_width_bits / 8)?
                        .to_le_bytes(),
                    FieldUpdateRule::WriteAsOnes => [0xff; 8],
                    FieldUpdateRule::WriteAsZeros => [0; 8],
                }
            } else {
                [0; 8]
            };

            let remaining_length = field.bit_length - written_so_far;
            let length = if i == 0 {
                usize::min(remaining_length, access_width_bits - (field.bit_index % access_width_bits))
            } else {
                usize::min(remaining_length, access_width_bits)
            };

            object::copy_bits(value_bytes, written_so_far, &mut bytes, dst_index, length);
            self.do_native_region_write(
                write_region,
                aligned_offset / 8,
                access_width_bits / 8,
                u64::from_le_bytes(bytes),
            )?;
            written_so_far += length;
        }

        Ok(())
    }

    /// Performs an actual read from an operation region. `offset` and `length` must respect the
    /// access requirements of the field being read, and are supplied in **bytes**. This may call
    /// AML methods if required, and may invoke user-supplied handlers.
    fn do_native_region_read(&self, region: &OpRegion, offset: usize, length: usize) -> Result<u64, AmlError> {
        trace!("Native field read. Region = {:?}, offset = {:#x}, length={:#x}", region, offset, length);

        if matches!(region.space, RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig) {
            if !self.native_region_is_connected(region) {
                return Err(AmlError::NoHandlerForRegionAccess(region.space));
            }
            let handler = self
                .native_region_handler
                .lock()
                .as_ref()
                .map(Arc::clone)
                .ok_or(AmlError::NoHandlerForRegionAccess(region.space))?;
            let token = handler.authorize(region, NativeRegionAccessMode::Read)?;
            return handler.read(&token, offset, length);
        }

        match region.space {
            RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig => unreachable!(),
            RegionSpace::EmbeddedControl
            | RegionSpace::SmBus
            | RegionSpace::SystemCmos
            | RegionSpace::PciBarTarget
            | RegionSpace::Ipmi
            | RegionSpace::GeneralPurposeIo
            | RegionSpace::GenericSerialBus
            | RegionSpace::Pcc
            | RegionSpace::Oem(_) => {
                if let Some(_handler) = self.region_handlers.lock().get(&region.space) {
                    warn!("Custom region handlers aren't actually supported yet.");
                    Err(AmlError::LibUnimplemented)
                } else {
                    Err(AmlError::NoHandlerForRegionAccess(region.space))
                }
            }
        }
    }

    /// Performs an actual write to an operation region. `offset` and `length` must respect the
    /// access requirements of the field being read, and are supplied in **bytes**. This may call
    /// AML methods if required, and may invoke user-supplied handlers.
    fn do_native_region_write(
        &self,
        region: &OpRegion,
        offset: usize,
        length: usize,
        value: u64,
    ) -> Result<(), AmlError> {
        trace!(
            "Native field write. Region = {:?}, offset = {:#x}, length={:#x}, value={:#x}",
            region, offset, length, value
        );

        if matches!(region.space, RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig) {
            if !self.native_region_is_connected(region) {
                return Err(AmlError::NoHandlerForRegionAccess(region.space));
            }
            let handler = self
                .native_region_handler
                .lock()
                .as_ref()
                .map(Arc::clone)
                .ok_or(AmlError::NoHandlerForRegionAccess(region.space))?;
            let token = handler.authorize(region, NativeRegionAccessMode::ReadWrite)?;
            return handler.write(&token, offset, length, value);
        }

        match region.space {
            RegionSpace::SystemMemory | RegionSpace::SystemIO | RegionSpace::PciConfig => unreachable!(),
            RegionSpace::EmbeddedControl
            | RegionSpace::SmBus
            | RegionSpace::SystemCmos
            | RegionSpace::PciBarTarget
            | RegionSpace::Ipmi
            | RegionSpace::GeneralPurposeIo
            | RegionSpace::GenericSerialBus
            | RegionSpace::Pcc
            | RegionSpace::Oem(_) => {
                if let Some(_handler) = self.region_handlers.lock().get(&region.space) {
                    warn!("Custom region handlers aren't actually supported yet.");
                    Err(AmlError::LibUnimplemented)
                } else {
                    Err(AmlError::NoHandlerForRegionAccess(region.space))
                }
            }
        }
    }

    /// Return to the beginning of a While loop - either because `Continue` was executed or
    /// because we reached the end of the While block.
    fn continue_loop(&self, context: &mut MethodContext) -> Result<(), AmlError> {
        self.charge_loop_iteration()?;
        let start: u64;

        if let BlockKind::While { start_pc, start_nanos } = &context.current_block.kind {
            context.current_block.pc = *start_pc;
            start = *start_nanos;
        } else {
            loop {
                let Some(block) = context.block_stack.pop() else { Err(AmlError::ContinueOutsideOfWhile)? };
                context.current_block = block;
                if let BlockKind::While { start_pc, start_nanos } = context.current_block.kind {
                    context.current_block.pc = start_pc;
                    start = start_nanos;
                    break;
                }
            }
        }

        if self.handler.nanos_since_boot() - start > LOOP_TIMEOUT_NS {
            return Err(AmlError::LoopTimeout);
        }

        context.start(OpInFlight::new(Opcode::While, &[ResolveBehaviour::TermArg]));
        Ok(())
    }
}

/// A `MethodContext` represents a piece of running AML code - either a real method, or the
/// top-level of an AML table.
const MAX_AML_CONTEXT_STACK_DEPTH: usize = 256;
const MAX_AML_NAME_COMPONENTS: usize = namespace::MAX_AML_NAME_COMPONENTS;

struct NameStringHeapReservation {
    meter: Option<Arc<dyn ExecutionMeter>>,
    bytes: u64,
    persistent_bytes: u64,
    generation: Option<u64>,
}

impl NameStringHeapReservation {
    fn new(meter: Option<Arc<dyn ExecutionMeter>>) -> Self {
        Self { meter, bytes: 0, persistent_bytes: 0, generation: None }
    }

    fn reserve_bytes(&mut self, bytes: u64) -> Result<(), AmlError> {
        let total = self.bytes.checked_add(bytes).ok_or(AmlError::BudgetExceeded)?;
        if let Some(meter) = &self.meter {
            let generation = meter.reserve_working_heap_scoped(bytes)?;
            if self.generation.is_some_and(|previous| previous != generation) {
                meter.release_working_heap_scoped(bytes, generation);
                return Err(AmlError::BudgetExceeded);
            }
            self.generation = Some(generation);
            self.bytes = total;
        }
        Ok(())
    }

    fn commit_to_persistent(&mut self) -> Result<(), AmlError> {
        self.commit_bytes_to_persistent(self.bytes)
    }

    fn commit_bytes_to_persistent(&mut self, bytes: u64) -> Result<(), AmlError> {
        if bytes > self.bytes {
            return Err(AmlError::BudgetExceeded);
        }
        if bytes == 0 {
            return Ok(());
        }
        if let Some(meter) = &self.meter {
            let persistent_bytes = self.persistent_bytes.checked_add(bytes).ok_or(AmlError::BudgetExceeded)?;
            meter.commit_working_heap_scoped(bytes, self.generation.unwrap_or(0))?;
            self.bytes -= bytes;
            self.persistent_bytes = persistent_bytes;
            if self.bytes == 0 {
                self.generation = None;
            }
        }
        Ok(())
    }

    fn reserve_vec_capacity<T>(&mut self, values: &mut Vec<T>, max_capacity: usize) -> Result<(), AmlError> {
        if let Some(meter) = &self.meter {
            meter.loop_iteration()?;
        }
        if values.len() < values.capacity() {
            return Ok(());
        }
        if values.len() >= max_capacity {
            return Err(AmlError::BudgetExceeded);
        }

        let doubled_capacity = values.capacity().checked_mul(2).unwrap_or(max_capacity);
        let new_capacity = usize::min(max_capacity, usize::max(4, doubled_capacity));
        let additional = new_capacity
            .checked_sub(values.len())
            .filter(|additional| *additional != 0)
            .ok_or(AmlError::BudgetExceeded)?;
        let new_bytes = new_capacity
            .checked_mul(mem::size_of::<T>())
            .and_then(|bytes| bytes.checked_add(24))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(AmlError::BudgetExceeded)?;
        let additional_bytes = new_bytes.checked_sub(self.bytes).ok_or(AmlError::BudgetExceeded)?;
        let generation = if let Some(meter) = &self.meter {
            let generation = meter.reserve_working_heap_scoped(additional_bytes)?;
            if self.generation.is_some_and(|previous| previous != generation) {
                meter.release_working_heap_scoped(additional_bytes, generation);
                return Err(AmlError::BudgetExceeded);
            }
            Some(generation)
        } else {
            None
        };
        if values.try_reserve_exact(additional).is_err() {
            if let (Some(meter), Some(generation)) = (&self.meter, generation) {
                meter.release_working_heap_scoped(additional_bytes, generation);
            }
            return Err(AmlError::BudgetExceeded);
        }
        if self.meter.is_some() {
            self.generation = generation;
            self.bytes = new_bytes;
        }
        Ok(())
    }

    fn push<T>(&mut self, values: &mut Vec<T>, max_capacity: usize, value: T) -> Result<(), AmlError> {
        self.reserve_vec_capacity(values, max_capacity)?;
        values.push(value);
        Ok(())
    }
}

impl core::fmt::Debug for NameStringHeapReservation {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("NameStringHeapReservation").field("bytes", &self.bytes).finish()
    }
}

impl Drop for NameStringHeapReservation {
    fn drop(&mut self) {
        if let Some(meter) = &self.meter {
            meter.release_working_heap_scoped(self.bytes, self.generation.unwrap_or(0));
            let _ = meter.release_persistent_allocation(self.persistent_bytes, 0, 0);
        }
    }
}

impl ObjectHeapReservation for NameStringHeapReservation {
    fn commit_to_persistent(&mut self) -> Result<(), AmlError> {
        NameStringHeapReservation::commit_to_persistent(self)
    }
}

fn reserve_argument_scratch(
    meter: Option<Arc<dyn ExecutionMeter>>,
    argument_count: usize,
) -> Result<NameStringHeapReservation, AmlError> {
    let bytes = argument_count
        .checked_mul(mem::size_of::<WrappedObject>())
        .and_then(|bytes| bytes.checked_add(24))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(AmlError::BudgetExceeded)?;
    let mut reservation = NameStringHeapReservation::new(meter);
    reservation.reserve_bytes(bytes)?;
    Ok(reservation)
}

#[doc(hidden)]
pub struct MeteredNameString {
    name: AmlName,
    _heap_reservation: NameStringHeapReservation,
}

impl MeteredNameString {
    pub fn as_name(&self) -> &AmlName {
        &self.name
    }

    fn parse_name_metered(
        input: &str,
        meter: Option<Arc<dyn ExecutionMeter>>,
    ) -> Result<MeteredNameString, AmlError> {
        let max_encoded_bytes = namespace::MAX_AML_NAME_COMPONENTS
            .checked_mul(5)
            .and_then(|bytes| bytes.checked_add(1))
            .ok_or(AmlError::BudgetExceeded)?;
        if input.len() > max_encoded_bytes {
            return Err(AmlError::BudgetExceeded);
        }
        let Some(meter) = meter else {
            return Ok(Self::from_owned_name(AmlName::from_str(input)?));
        };

        for _ in 0..input.len() {
            meter.loop_iteration()?;
        }
        let component_capacity = usize::min(input.len(), namespace::MAX_AML_NAME_COMPONENTS);
        let allocation_bytes = mem::size_of::<Vec<namespace::NameComponent>>()
            .checked_add(
                component_capacity
                    .checked_mul(mem::size_of::<namespace::NameComponent>())
                    .ok_or(AmlError::BudgetExceeded)?,
            )
            .and_then(|bytes| bytes.checked_add(24))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(AmlError::BudgetExceeded)?;
        let mut heap_reservation = NameStringHeapReservation::new(Some(meter));
        heap_reservation.reserve_bytes(allocation_bytes)?;
        let name = AmlName::from_str(input)?;
        Ok(Self::with_heap_reservation(name, heap_reservation))
    }

    fn from_owned_name(name: AmlName) -> Self {
        Self { name, _heap_reservation: NameStringHeapReservation::new(None) }
    }

    fn account_owned_name(
        name: AmlName,
        meter: Option<Arc<dyn ExecutionMeter>>,
    ) -> Result<MeteredNameString, AmlError> {
        let mut heap_reservation = NameStringHeapReservation::new(meter);
        heap_reservation.reserve_bytes(name.heap_allocation_upper_bound().ok_or(AmlError::BudgetExceeded)?)?;
        Ok(Self::with_heap_reservation(name, heap_reservation))
    }

    fn clone_name_metered(
        name: &AmlName,
        meter: Option<Arc<dyn ExecutionMeter>>,
    ) -> Result<MeteredNameString, AmlError> {
        let mut heap_reservation = NameStringHeapReservation::new(meter);
        heap_reservation.reserve_bytes(name.clone_heap_upper_bound().ok_or(AmlError::BudgetExceeded)?)?;
        let name = name.try_clone()?;
        Ok(Self::with_heap_reservation(name, heap_reservation))
    }

    fn with_heap_reservation(name: AmlName, heap_reservation: NameStringHeapReservation) -> Self {
        Self { name, _heap_reservation: heap_reservation }
    }

    fn into_parts(self) -> (AmlName, NameStringHeapReservation) {
        (self.name, self._heap_reservation)
    }

    fn reserve_clone_scratch(&self) -> Result<NameStringHeapReservation, AmlError> {
        let bytes = self.name.clone_heap_upper_bound().ok_or(AmlError::BudgetExceeded)?;
        let mut reservation = NameStringHeapReservation::new(self._heap_reservation.meter.clone());
        reservation.reserve_bytes(bytes)?;
        Ok(reservation)
    }

    fn try_clone_metered(&self) -> Result<MeteredNameString, AmlError> {
        Self::clone_name_metered(&self.name, self._heap_reservation.meter.clone())
    }
}

impl core::fmt::Debug for MeteredNameString {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.name.fmt(formatter)
    }
}

impl core::fmt::Display for MeteredNameString {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.name, formatter)
    }
}

impl core::ops::Deref for MeteredNameString {
    type Target = AmlName;

    fn deref(&self) -> &Self::Target {
        &self.name
    }
}

fn name_path_heap_bytes(name: &AmlName, scope: &AmlName) -> Result<u64, AmlError> {
    name.heap_allocation_upper_bound()
        .and_then(|name_bytes| scope.clone_heap_upper_bound()?.checked_add(name_bytes))
        .ok_or_else(|| AmlError::BudgetExceeded)
}

fn push_namestring_component(
    components: &mut Vec<NameComponent>,
    component: NameComponent,
    reservation: &mut NameStringHeapReservation,
) -> Result<(), AmlError> {
    reservation.push(components, MAX_AML_NAME_COMPONENTS, component)
}

/// Construct a vector after the caller has reserved its estimated heap cost.
fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, AmlError> {
    let mut values = allocator::vec_new();
    values.try_reserve_exact(capacity).map_err(|_| AmlError::BudgetExceeded)?;
    Ok(values)
}

fn try_vec_from_slice<T: Clone>(source: &[T]) -> Result<Vec<T>, AmlError> {
    let mut values = try_vec_with_capacity(source.len())?;
    values.extend_from_slice(source);
    Ok(values)
}

fn try_string_with_capacity(capacity: usize) -> Result<String, AmlError> {
    let mut value = String::new();
    value.try_reserve_exact(capacity).map_err(|_| AmlError::BudgetExceeded)?;
    Ok(value)
}

fn try_string_from_str(source: &str) -> Result<String, AmlError> {
    let mut value = try_string_with_capacity(source.len())?;
    value.push_str(source);
    Ok(value)
}

fn string_heap_bytes(capacity: usize) -> Option<u64> {
    capacity.checked_add(if capacity == 0 { 0 } else { 24 }).and_then(|bytes| u64::try_from(bytes).ok())
}

fn integer_string_capacity(value: u64, hexadecimal: bool) -> usize {
    if hexadecimal {
        let digits = if value == 0 { 1 } else { (u64::BITS - value.leading_zeros()).div_ceil(4) as usize };
        2 + digits
    } else {
        let mut digits = 1;
        let mut remaining = value;
        while remaining >= 10 {
            remaining /= 10;
            digits += 1;
        }
        digits
    }
}

fn buffer_string_capacity(byte_count: usize, hexadecimal: bool) -> Option<usize> {
    if byte_count == 0 {
        return Some(0);
    }
    byte_count.checked_mul(if hexadecimal { 5 } else { 4 })?.checked_sub(1)
}

fn try_format_integer_string(value: u64, hexadecimal: bool) -> Result<String, AmlError> {
    use core::fmt::Write as _;

    let mut output = try_string_with_capacity(integer_string_capacity(value, hexadecimal))?;
    let result = if hexadecimal { write!(&mut output, "{value:#X}") } else { write!(&mut output, "{value}") };
    result.map_err(|_| AmlError::BudgetExceeded)?;
    Ok(output)
}

fn try_format_buffer_string(bytes: &[u8], hexadecimal: bool) -> Result<String, AmlError> {
    use core::fmt::Write as _;

    let capacity = buffer_string_capacity(bytes.len(), hexadecimal).ok_or_else(|| AmlError::BudgetExceeded)?;
    let mut output = try_string_with_capacity(capacity)?;
    for (index, byte) in bytes.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        let result = if hexadecimal { write!(&mut output, "{byte:#04X}") } else { write!(&mut output, "{byte}") };
        result.map_err(|_| AmlError::BudgetExceeded)?;
    }
    Ok(output)
}

fn concat_buffer_capacity(first: &Object, second: &Object, integer_size: IntegerSize) -> Result<usize, AmlError> {
    match first {
        Object::Integer(_) => Ok(2 * integer_size as usize),
        Object::Buffer(bytes) => bytes
            .len()
            .checked_add(object_buffer_capacity(second, integer_size)?)
            .ok_or_else(|| AmlError::BudgetExceeded),
        _ => Err(AmlError::InternalError("invalid first object for buffer Concat".to_string())),
    }
}

fn object_buffer_capacity(object: &Object, integer_size: IntegerSize) -> Result<usize, AmlError> {
    match object {
        Object::Buffer(bytes) => Ok(bytes.len()),
        Object::Integer(_) => Ok(integer_size as usize),
        Object::String(value) => Ok(value.len()),
        _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ConvertToBuffer, typ: object.typ() }),
    }
}

fn try_concat_buffer_values(
    first: &Object,
    second: &Object,
    integer_size: IntegerSize,
    reserve: impl FnOnce(u64) -> Result<(), AmlError>,
) -> Result<Vec<u8>, AmlError> {
    let second_integer =
        if matches!(first, Object::Integer(_)) { Some(second.to_integer(integer_size)?) } else { None };
    let capacity = concat_buffer_capacity(first, second, integer_size)?;
    reserve(u64::try_from(capacity).map_err(|_| AmlError::BudgetExceeded)?)?;
    let mut output = try_vec_with_capacity(capacity)?;
    match first {
        Object::Integer(value) => {
            append_integer_buffer(&mut output, *value, integer_size);
            let second_integer = second_integer
                .ok_or_else(|| AmlError::InternalError("Concat lost integer operand".to_string()))?;
            append_integer_buffer(&mut output, second_integer, integer_size);
        }
        Object::Buffer(bytes) => {
            output.extend_from_slice(bytes);
            append_object_buffer(&mut output, second, integer_size)?;
        }
        _ => return Err(AmlError::InternalError("invalid first object for buffer Concat".to_string())),
    }
    Ok(output)
}

fn append_integer_buffer(output: &mut Vec<u8>, value: u64, integer_size: IntegerSize) {
    match integer_size {
        IntegerSize::FourBytes => output.extend_from_slice(&(value as u32).to_le_bytes()),
        IntegerSize::EightBytes => output.extend_from_slice(&value.to_le_bytes()),
    }
}

fn append_object_buffer(output: &mut Vec<u8>, object: &Object, integer_size: IntegerSize) -> Result<(), AmlError> {
    match object {
        Object::Buffer(bytes) => output.extend_from_slice(bytes),
        Object::Integer(value) => append_integer_buffer(output, *value, integer_size),
        Object::String(value) => output.extend_from_slice(value.as_bytes()),
        _ => return Err(AmlError::InvalidOperationOnObject { op: Operation::ConvertToBuffer, typ: object.typ() }),
    }
    Ok(())
}

fn concat_string_capacity(first: &Object, second: &Object) -> Result<usize, AmlError> {
    object_string_capacity(first)?
        .checked_add(object_string_capacity(second)?)
        .ok_or_else(|| AmlError::BudgetExceeded)
}

fn object_string_capacity(object: &Object) -> Result<usize, AmlError> {
    let mut object = object;
    while let Object::Reference { inner, .. } = object {
        object = inner;
    }
    match object {
        Object::Uninitialized => Ok("[Uninitialized Object]".len()),
        Object::Buffer(bytes) => bytes.len().checked_mul(3).ok_or_else(|| AmlError::BudgetExceeded),
        Object::BufferField { .. } => Ok("[Buffer Field]".len()),
        Object::Device => Ok("[Device]".len()),
        Object::Event(_) => Ok("[Event]".len()),
        Object::FieldUnit(_) => Ok("[Field]".len()),
        Object::Integer(value) => Ok(integer_string_capacity(*value, false)),
        Object::Method { .. } | Object::NativeMethod { .. } => Ok("[Control Method]".len()),
        Object::Mutex { .. } => Ok("[Mutex]".len()),
        Object::NamePath { name, .. } => name.encoded_len().ok_or_else(|| AmlError::BudgetExceeded),
        Object::OpRegion(_) => Ok("[Operation Region]".len()),
        Object::Package(_) => Ok("[Package]".len()),
        Object::PowerResource { .. } => Ok("[Power Resource]".len()),
        Object::Processor { .. } => Ok("[Processor]".len()),
        Object::RawDataBuffer => Ok("[Raw Data Buffer]".len()),
        Object::String(value) => Ok(value.len()),
        Object::ThermalZone => Ok("[Thermal Zone]".len()),
        Object::Debug => Ok("[Debug Object]".len()),
        Object::Reference { .. } => unreachable!(),
    }
}

fn try_concat_string_values(
    first: &Object,
    second: &Object,
    reserve: impl FnOnce(u64) -> Result<(), AmlError>,
) -> Result<String, AmlError> {
    let capacity = concat_string_capacity(first, second)?;
    let heap_bytes = string_heap_bytes(capacity).ok_or_else(|| AmlError::BudgetExceeded)?;
    reserve(heap_bytes)?;
    let mut output = try_string_with_capacity(capacity)?;
    append_object_string(&mut output, first)?;
    append_object_string(&mut output, second)?;
    Ok(output)
}

fn append_object_string(output: &mut String, object: &Object) -> Result<(), AmlError> {
    use core::fmt::Write as _;

    let mut object = object;
    while let Object::Reference { inner, .. } = object {
        object = inner;
    }
    match object {
        Object::Uninitialized => output.push_str("[Uninitialized Object]"),
        Object::Buffer(bytes) => append_utf8_lossy(output, bytes)?,
        Object::BufferField { .. } => output.push_str("[Buffer Field]"),
        Object::Device => output.push_str("[Device]"),
        Object::Event(_) => output.push_str("[Event]"),
        Object::FieldUnit(_) => output.push_str("[Field]"),
        Object::Integer(value) => write!(output, "{value}").map_err(|_| AmlError::BudgetExceeded)?,
        Object::Method { .. } | Object::NativeMethod { .. } => output.push_str("[Control Method]"),
        Object::Mutex { .. } => output.push_str("[Mutex]"),
        Object::NamePath { name, .. } => append_name_string(output, name),
        Object::OpRegion(_) => output.push_str("[Operation Region]"),
        Object::Package(_) => output.push_str("[Package]"),
        Object::PowerResource { .. } => output.push_str("[Power Resource]"),
        Object::Processor { .. } => output.push_str("[Processor]"),
        Object::RawDataBuffer => output.push_str("[Raw Data Buffer]"),
        Object::String(value) => output.push_str(value),
        Object::ThermalZone => output.push_str("[Thermal Zone]"),
        Object::Debug => output.push_str("[Debug Object]"),
        Object::Reference { .. } => {
            return Err(AmlError::InternalError("Concat reference did not resolve".to_string()));
        }
    }
    Ok(())
}

fn append_name_string(output: &mut String, name: &AmlName) {
    name.append_rendered_to(output);
}

fn append_utf8_lossy(output: &mut String, bytes: &[u8]) -> Result<(), AmlError> {
    let mut remaining = bytes;
    while !remaining.is_empty() {
        match core::str::from_utf8(remaining) {
            Ok(valid) => {
                output.push_str(valid);
                break;
            }
            Err(error) => {
                let valid_length = error.valid_up_to();
                let valid_prefix = core::str::from_utf8(&remaining[..valid_length]).map_err(|_| {
                    AmlError::InternalError("UTF-8 validator returned an invalid prefix".to_string())
                })?;
                output.push_str(valid_prefix);
                let invalid_length = error.error_len().unwrap_or(remaining.len() - valid_length);
                output.push('\u{fffd}');
                remaining = &remaining[valid_length + invalid_length..];
            }
        }
    }
    Ok(())
}

fn try_lossy_string_prefix(bytes: &[u8]) -> Result<String, AmlError> {
    if bytes.is_empty() {
        return Ok(String::new());
    }
    let capacity = bytes.len().checked_mul(3).ok_or_else(|| AmlError::BudgetExceeded)?;
    let mut output = String::new();
    output.try_reserve_exact(capacity).map_err(|_| AmlError::BudgetExceeded)?;

    let mut remaining = bytes;
    while !remaining.is_empty() {
        match core::str::from_utf8(remaining) {
            Ok(valid) => {
                output.push_str(valid.split('\0').next().unwrap_or(""));
                break;
            }
            Err(error) => {
                let valid_length = error.valid_up_to();
                let valid_prefix = core::str::from_utf8(&remaining[..valid_length]).map_err(|_| {
                    AmlError::InternalError("UTF-8 validator returned an invalid prefix".to_string())
                })?;
                if let Some(nul_index) = valid_prefix.find('\0') {
                    output.push_str(&valid_prefix[..nul_index]);
                    break;
                }
                output.push_str(valid_prefix);
                let invalid_length = error.error_len().unwrap_or(remaining.len() - valid_length);
                output.push('\u{fffd}');
                remaining = &remaining[valid_length + invalid_length..];
            }
        }
    }
    Ok(output)
}

fn aml_name_clone_heap_bytes(name: &AmlName) -> Result<u64, AmlError> {
    name.encoded_len()
        .and_then(|bytes| bytes.checked_mul(mem::size_of::<namespace::NameComponent>()))
        .and_then(|bytes| bytes.checked_add(24))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| AmlError::BudgetExceeded)
}

struct MethodContext {
    current_block: Block,
    block_stack: Vec<Block>,
    in_flight: Vec<OpInFlight>,
    args: [WrappedObject; 8],
    locals: [WrappedObject; 8],
    current_scope: AmlName,

    _method_scope_heap_reservations: [Option<NameStringHeapReservation>; 2],
    scope_heap_reservations: Vec<NameStringHeapReservation>,
    _scope_stack_heap_reservation: NameStringHeapReservation,
    _block_stack_heap_reservation: NameStringHeapReservation,
    _in_flight_heap_reservation: NameStringHeapReservation,
    _method: Option<WrappedObject>,
    execution_meter: Option<Arc<dyn ExecutionMeter>>,
    allocation_error: Option<AmlError>,
}

struct Block {
    stream: Vec<u8>,
    pc: usize,
    kind: BlockKind,
    _heap_reservation: NameStringHeapReservation,
}

impl Block {
    fn stream(&self) -> &[u8] {
        &self.stream
    }
}

#[derive(PartialEq, Debug)]
pub enum BlockKind {
    Table,
    Method {
        method_scope: AmlName,
    },
    Scope {
        old_scope: AmlName,
    },
    Package,
    VarPackage,
    /// Used for executing the then-branch of an `DefIfElse`. After finishing, it will check for
    /// and skip over an else-branch, if present.
    IfThenBranch,
    While {
        start_pc: usize,
        start_nanos: u64,
    },
}

/// A `ResolveBehaviour` describes how a name at the top-level should be resolved as part of an
/// operation.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ResolveBehaviour {
    /// Attempt to resolve the name to an object that has already been defined. There are generally
    /// no forward definitions in AML, so this is the usual resolution behaviour for most operands.
    TermArg,
    /// Resolve a name to reference an object. This is used when an operation needs to operate on
    /// the object itself, rather than evaluate it to a value (for example, given a `FieldUnit`, `TermArg` would read a
    /// value from the field, while this would resolve to the `FieldUnit` itself. Can be a name,
    /// argument, or local.
    SimpleName,
    /// Like a `SimpleName`, but can also be the `Debug` object or an operation that produces a
    /// reference.
    SuperName,
    /// Behaves the same as `SuperName` if the object exists, but resolves successfully to an
    /// unresolved reference if the object does not exist. Used by `DefCondRefOf`.
    SuperNameIfExists,
    /// `SuperName`, but can also be `NullName`
    Target,
    /// Surrogate argument, used by `DefPackage` and `DefVarPackage`. Only one of these is emitted,
    /// but represents parsing of potentially many package elements. Names in packages refer to the
    /// named object (see `Package` in the ASL reference, §19.6.101 of ACPI 6.5). We keep the name
    /// and the scope it appeared in, and only resolve it when the reference is used, as the object
    /// may not exist yet. ACPICA resolves these at package creation, but that makes resolution
    /// depend on the order tables are loaded in.
    AsPackageElements,
    /// Used with [`OpInFlight::new_with`] to represent arguments that have already been resolved
    /// when an operation enters flight.
    Placeholder,
    /// The match statement takes two single byte arguments - the "match opcode".
    MatchOpcode,
}

#[derive(Debug)]
struct OpInFlight {
    op: Opcode,
    expected_arguments: usize,
    arguments: Vec<Argument>,
    _arguments_heap_reservation: NameStringHeapReservation,
    resolve_behaviour: &'static [ResolveBehaviour],
}

#[derive(Debug)]
enum Argument {
    Object(WrappedObject),
    Namestring(MeteredNameString),
    ByteData(u8),
    DWordData(u32),
    TrackedPc(usize),
    PkgLength(usize),
    MatchOp(MatchOp),
}

impl OpInFlight {
    /// Creates a new `OpInFlight`. The number of expected arguments is inferred from the number of
    /// `ResolveBehaviour`s passed.
    pub fn new(op: Opcode, resolve_behaviour: &'static [ResolveBehaviour]) -> OpInFlight {
        OpInFlight {
            op,
            expected_arguments: resolve_behaviour.len(),
            arguments: allocator::vec_new(),
            _arguments_heap_reservation: NameStringHeapReservation::new(None),
            resolve_behaviour,
        }
    }

    /// Creates a new `OpInFlight` with the given number of expected arguments. This should be used
    /// when the correct number of expected arguments differs from the number of
    /// `ResolveBehaviour`s passed.
    pub fn new_dynamic(
        op: Opcode,
        expected_arguments: usize,
        resolve_behaviour: &'static [ResolveBehaviour],
    ) -> OpInFlight {
        OpInFlight {
            op,
            expected_arguments,
            arguments: allocator::vec_new(),
            _arguments_heap_reservation: NameStringHeapReservation::new(None),
            resolve_behaviour,
        }
    }

    /// Get the desired `ResolveBehaviour` for the argument currently being interpreted
    fn resolve_behaviour(&self) -> Result<ResolveBehaviour, AmlError> {
        if let Some(behaviour) = self.resolve_behaviour.get(self.arguments.len()) {
            Ok(*behaviour)
        } else if self.op == Opcode::Package || (self.op == Opcode::VarPackage && !self.arguments.is_empty()) {
            Ok(ResolveBehaviour::AsPackageElements)
        } else {
            Err(AmlError::InternalError(format!("missing resolve behaviour for operation {:?}", self.op)))
        }
    }
}

fn try_wrapped_object_array<const N: usize>(
    meter: Option<Arc<dyn ExecutionMeter>>,
    mut build: impl FnMut(usize) -> Result<WrappedObject, AmlError>,
) -> Result<[WrappedObject; N], AmlError> {
    let slots_bytes = N
        .checked_mul(mem::size_of::<WrappedObject>())
        .and_then(|bytes| bytes.checked_add(24))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(AmlError::BudgetExceeded)?;
    let mut slots_reservation = NameStringHeapReservation::new(meter);
    slots_reservation.reserve_bytes(slots_bytes)?;

    let mut values = allocator::vec_new();
    values.try_reserve_exact(N).map_err(|_| AmlError::BudgetExceeded)?;
    for index in 0..N {
        values.push(build(index)?);
    }
    values.try_into().map_err(|_| AmlError::BudgetExceeded)
}

impl MethodContext {
    fn resolve_name(&self, name: &AmlName, scope: &AmlName) -> Result<MeteredNameString, AmlError> {
        match &self.execution_meter {
            Some(meter) => name.resolve_metered(scope, Arc::clone(meter)),
            None => Ok(MeteredNameString::from_owned_name(name.resolve(scope)?)),
        }
    }

    fn search_with_flags(
        &self,
        namespace: &Namespace,
        name: &AmlName,
    ) -> Result<(MeteredNameString, WrappedObject, namespace::ObjectFlags), AmlError> {
        match &self.execution_meter {
            Some(meter) => namespace.search_with_flags_metered(name, &self.current_scope, Arc::clone(meter)),
            None => {
                let (resolved, object, flags) = namespace.search_with_flags(name, &self.current_scope)?;
                Ok((MeteredNameString::from_owned_name(resolved), object, flags))
            }
        }
    }

    fn search(
        &self,
        namespace: &Namespace,
        name: &AmlName,
    ) -> Result<(MeteredNameString, WrappedObject), AmlError> {
        let (resolved, object, _) = self.search_with_flags(namespace, name)?;
        Ok((resolved, object))
    }

    fn search_if_present(
        &self,
        namespace: &Namespace,
        name: &AmlName,
    ) -> Result<Option<(MeteredNameString, WrappedObject)>, AmlError> {
        match &self.execution_meter {
            Some(meter) => namespace
                .search_with_flags_metered_if_present(name, &self.current_scope, Arc::clone(meter))
                .map(|result| result.map(|(resolved, object, _)| (resolved, object))),
            None => match namespace.search(name, &self.current_scope) {
                Ok((resolved, object)) => Ok(Some((MeteredNameString::from_owned_name(resolved), object))),
                Err(AmlError::ObjectDoesNotExist(_)) => Ok(None),
                Err(error) => Err(error),
            },
        }
    }

    fn enter_scope(&mut self, scope: MeteredNameString) -> Result<AmlName, AmlError> {
        let (scope, heap_reservation) = scope.into_parts();
        self._scope_stack_heap_reservation.push(
            &mut self.scope_heap_reservations,
            namespace::MAX_AML_NAME_COMPONENTS,
            heap_reservation,
        )?;
        Ok(mem::replace(&mut self.current_scope, scope))
    }

    fn leave_scope(&mut self, old_scope: AmlName) -> Result<(), AmlError> {
        let _scope_heap_reservation = self
            .scope_heap_reservations
            .pop()
            .ok_or_else(|| AmlError::InternalError("AML scope reservation stack underflow".to_string()))?;
        self.current_scope = old_scope;
        Ok(())
    }

    fn render_name_object(&mut self, name: &AmlName) -> Result<WrappedObject, AmlError> {
        let length = name.encoded_len().ok_or(AmlError::BudgetExceeded)?;
        let heap_bytes = length
            .checked_add(if length == 0 { 0 } else { 24 })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(AmlError::BudgetExceeded)?;
        if let Some(meter) = &self.execution_meter {
            for _ in 0..length {
                meter.loop_iteration()?;
            }
        }
        let value_reservation =
            reserve_boxed_metered_object_allocation(self.execution_meter.clone(), heap_bytes, 0, 0)?;
        let text = name.try_to_string_bounded(length)?.ok_or(AmlError::BudgetExceeded)?;
        let cell_reservation = reserve_boxed_metered_object_allocation(
            self.execution_meter.clone(),
            wrapped_object_cell_heap_bytes()?,
            0,
            0,
        )?;
        Ok(Object::String(text).wrap_metered(cell_reservation, value_reservation))
    }

    unsafe fn new_from_table(
        stream: &[u8],
        execution_meter: Option<Arc<dyn ExecutionMeter>>,
    ) -> Result<MethodContext, AmlError> {
        let allocation_bytes = stream
            .len()
            .checked_add(24)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| AmlError::BudgetExceeded)?;
        let mut stream_heap_reservation = NameStringHeapReservation::new(execution_meter.clone());
        stream_heap_reservation.reserve_bytes(allocation_bytes)?;
        let mut stream_copy = allocator::vec_new();
        stream_copy.try_reserve_exact(stream.len()).map_err(|_| AmlError::BudgetExceeded)?;
        stream_copy.extend_from_slice(stream);
        let block = Block {
            stream: stream_copy,
            pc: 0,
            kind: BlockKind::Table,
            _heap_reservation: stream_heap_reservation,
        };
        let args = try_wrapped_object_array(execution_meter.clone(), |_| {
            wrap_metered_object(Object::Uninitialized, execution_meter.clone())
        })?;
        let locals = try_wrapped_object_array(execution_meter.clone(), |_| {
            wrap_metered_object(Object::Uninitialized, execution_meter.clone())
        })?;
        Ok(MethodContext {
            current_block: block,
            block_stack: allocator::vec_new(),
            in_flight: allocator::vec_new(),
            args,
            locals,
            current_scope: AmlName::root(),
            _method_scope_heap_reservations: [None, None],
            scope_heap_reservations: allocator::vec_new(),
            _scope_stack_heap_reservation: NameStringHeapReservation::new(execution_meter.clone()),
            _block_stack_heap_reservation: NameStringHeapReservation::new(execution_meter.clone()),
            _in_flight_heap_reservation: NameStringHeapReservation::new(execution_meter.clone()),
            _method: None,
            execution_meter,
            allocation_error: None,
        })
    }

    fn new_from_method(
        method: WrappedObject,
        args: Vec<WrappedObject>,
        scope: MeteredNameString,
        execution_meter: Option<Arc<dyn ExecutionMeter>>,
    ) -> Result<MethodContext, AmlError> {
        if let Object::Method { code, flags } = &*method {
            if args.len() != flags.arg_count() {
                return Err(AmlError::MethodArgCountIncorrect);
            }
            let method_scope = scope.try_clone_metered()?;
            let (method_scope, method_scope_heap_reservation) = method_scope.into_parts();
            let (current_scope, current_scope_heap_reservation) = scope.into_parts();
            let allocation_bytes = code
                .len()
                .checked_add(24)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| AmlError::BudgetExceeded)?;
            let mut stream_heap_reservation = NameStringHeapReservation::new(execution_meter.clone());
            stream_heap_reservation.reserve_bytes(allocation_bytes)?;
            let mut stream = allocator::vec_new();
            stream.try_reserve_exact(code.len()).map_err(|_| AmlError::BudgetExceeded)?;
            stream.extend_from_slice(code);
            let block = Block {
                stream,
                pc: 0,
                kind: BlockKind::Method { method_scope },
                _heap_reservation: stream_heap_reservation,
            };
            let args = try_wrapped_object_array(execution_meter.clone(), |i| {
                if let Some(arg) = args.get(i) {
                    Ok(arg.clone())
                } else {
                    wrap_metered_object(Object::Uninitialized, execution_meter.clone())
                }
            })?;
            let locals = try_wrapped_object_array(execution_meter.clone(), |_| {
                wrap_metered_object(Object::Uninitialized, execution_meter.clone())
            })?;
            let context = MethodContext {
                current_block: block,
                block_stack: allocator::vec_new(),
                in_flight: allocator::vec_new(),
                args,
                locals,
                current_scope,
                _method_scope_heap_reservations: [
                    Some(method_scope_heap_reservation),
                    Some(current_scope_heap_reservation),
                ],
                scope_heap_reservations: allocator::vec_new(),
                _scope_stack_heap_reservation: NameStringHeapReservation::new(execution_meter.clone()),
                _block_stack_heap_reservation: NameStringHeapReservation::new(execution_meter.clone()),
                _in_flight_heap_reservation: NameStringHeapReservation::new(execution_meter.clone()),
                _method: Some(method.clone()),
                execution_meter,
                allocation_error: None,
            };
            Ok(context)
        } else {
            Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::Method, got: method.typ() })
        }
    }

    fn contribute_arg(&mut self, arg: Argument) {
        if self.allocation_error.is_some() {
            return;
        }
        let allocation_error = match self.in_flight.last_mut() {
            Some(in_flight) if in_flight.arguments.len() < in_flight.expected_arguments => {
                match in_flight._arguments_heap_reservation.push(
                    &mut in_flight.arguments,
                    in_flight.expected_arguments,
                    arg,
                ) {
                    Ok(()) => None,
                    Err(error) => Some(error),
                }
            }
            _ => None,
        };
        if allocation_error.is_some() {
            self.allocation_error = allocation_error;
        }
    }

    fn take_allocation_error(&mut self) -> Option<AmlError> {
        self.allocation_error.take()
    }

    fn method_scope(&self) -> Option<&AmlName> {
        if let BlockKind::Method { method_scope } = &self.current_block.kind {
            return Some(method_scope);
        }
        self.block_stack.iter().rev().find_map(|block| {
            if let BlockKind::Method { method_scope } = &block.kind { Some(method_scope) } else { None }
        })
    }

    /// Start a new `InFlightOp`.
    fn start(&mut self, mut op: OpInFlight) {
        trace!(
            "START OP: {:?}, args: {:?}, with {} more needed ({:?})",
            op.op,
            op.arguments,
            op.expected_arguments - op.arguments.len(),
            op.resolve_behaviour
        );
        if self.allocation_error.is_some() {
            return;
        }
        op._arguments_heap_reservation.meter = self.execution_meter.clone();
        let initial_argument_bytes = if op.arguments.capacity() == 0 {
            Some(0)
        } else {
            op.arguments
                .capacity()
                .checked_mul(mem::size_of::<Argument>())
                .and_then(|bytes| bytes.checked_add(24))
                .and_then(|bytes| u64::try_from(bytes).ok())
        };
        let Some(initial_argument_bytes) = initial_argument_bytes else {
            self.allocation_error = Some(AmlError::BudgetExceeded);
            return;
        };
        let additional_argument_bytes =
            initial_argument_bytes.saturating_sub(op._arguments_heap_reservation.bytes);
        if let Err(error) = op._arguments_heap_reservation.reserve_bytes(additional_argument_bytes) {
            self.allocation_error = Some(error);
            return;
        }
        if let Err(error) =
            self._in_flight_heap_reservation.push(&mut self.in_flight, MAX_AML_CONTEXT_STACK_DEPTH, op)
        {
            self.allocation_error = Some(error);
        }
    }

    fn start_with_arguments<F>(
        &mut self,
        op: Opcode,
        expected_arguments: usize,
        resolve_behaviour: &'static [ResolveBehaviour],
        argument_capacity: usize,
        fill_arguments: F,
    ) -> Result<(), AmlError>
    where
        F: FnOnce(&mut Vec<Argument>) -> Result<(), AmlError>,
    {
        if argument_capacity > expected_arguments {
            return Err(AmlError::InternalError("AML argument capacity exceeds its operation".to_string()));
        }
        let allocation_bytes = argument_capacity
            .checked_mul(mem::size_of::<Argument>())
            .and_then(|bytes| bytes.checked_add(if argument_capacity == 0 { 0 } else { 24 }))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(AmlError::BudgetExceeded)?;
        let mut heap_reservation = NameStringHeapReservation::new(self.execution_meter.clone());
        heap_reservation.reserve_bytes(allocation_bytes)?;

        let mut arguments = allocator::vec_new();
        arguments.try_reserve_exact(argument_capacity).map_err(|_| AmlError::BudgetExceeded)?;
        fill_arguments(&mut arguments)?;
        if arguments.len() > argument_capacity {
            return Err(AmlError::BudgetExceeded);
        }

        self.start(OpInFlight {
            op,
            expected_arguments,
            arguments,
            _arguments_heap_reservation: heap_reservation,
            resolve_behaviour,
        });
        Ok(())
    }

    fn retire_op(&mut self, op: OpInFlight) {
        trace!("RETIRE OP: {:?}, args: {:?}", op.op, op.arguments);
    }

    fn remaining_package_length(&self, start: usize, package_length: usize) -> Result<usize, AmlError> {
        let package_end = start
            .checked_add(package_length)
            .filter(|end| *end <= self.current_block.stream().len())
            .ok_or(AmlError::InvalidPackageLength)?;
        package_end.checked_sub(self.current_block.pc).ok_or(AmlError::InvalidPackageLength)
    }

    fn pop_block(&mut self) -> Result<Block, AmlError> {
        self.block_stack.pop().ok_or_else(|| AmlError::InternalError("AML block stack underflow".to_string()))
    }

    fn start_new_block(&mut self, kind: BlockKind, length: usize) -> Result<(), AmlError> {
        if self.block_stack.len() >= MAX_AML_CONTEXT_STACK_DEPTH {
            return Err(AmlError::BudgetExceeded);
        }
        let end = self
            .current_block
            .pc
            .checked_add(length)
            .filter(|end| *end <= self.current_block.stream().len())
            .ok_or(AmlError::InvalidPackageLength)?;
        self._block_stack_heap_reservation
            .reserve_vec_capacity(&mut self.block_stack, MAX_AML_CONTEXT_STACK_DEPTH)?;
        let meter = self.execution_meter.clone();
        let allocation_bytes = end
            .checked_add(24)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| AmlError::BudgetExceeded)?;
        let mut stream_heap_reservation = NameStringHeapReservation::new(meter);
        stream_heap_reservation.reserve_bytes(allocation_bytes)?;
        let mut stream = allocator::vec_new();
        stream.try_reserve_exact(end).map_err(|_| AmlError::BudgetExceeded)?;
        stream.extend_from_slice(&self.current_block.stream()[..end]);
        let block = Block { stream, pc: self.current_block.pc, kind, _heap_reservation: stream_heap_reservation };
        self.current_block.pc = end;
        self.block_stack.push(mem::replace(&mut self.current_block, block));
        Ok(())
    }

    fn opcode(&mut self) -> Result<Opcode, AmlError> {
        let opcode: u16 = match self.next()? {
            0x5b => {
                let ext = self.next()?;
                (0x5b << 8) as u16 | ext as u16
            }
            other => other as u16,
        };

        // Opcodes Zero and One overlap with the Match Opcodes that are only used in the match
        // expression. It's reasonable to assume that if Zero or One appear at the time we expect a
        // Match Opcode, then they *are* Match Opcodes instead of regular arguments.
        Ok(match opcode {
            0x00 => {
                if self.expecting_match_opcode() {
                    Opcode::MatchOp(MatchOp::MTR)
                } else {
                    Opcode::Zero
                }
            }
            0x01 => {
                if self.expecting_match_opcode() {
                    Opcode::MatchOp(MatchOp::MEQ)
                } else {
                    Opcode::One
                }
            }
            0x02 => Opcode::MatchOp(MatchOp::MLE),
            0x03 => Opcode::MatchOp(MatchOp::MLT),
            0x04 => Opcode::MatchOp(MatchOp::MGE),
            0x05 => Opcode::MatchOp(MatchOp::MGT),
            0x06 => Opcode::Alias,
            0x08 => Opcode::Name,
            0x0a => Opcode::BytePrefix,
            0x0b => Opcode::WordPrefix,
            0x0c => Opcode::DWordPrefix,
            0x0d => Opcode::StringPrefix,
            0x0e => Opcode::QWordPrefix,
            0x10 => Opcode::Scope,
            0x11 => Opcode::Buffer,
            0x12 => Opcode::Package,
            0x13 => Opcode::VarPackage,
            0x14 => Opcode::Method,
            0x15 => Opcode::External,
            0x2e => Opcode::DualNamePrefix,
            0x2f => Opcode::MultiNamePrefix,
            0x30..=0x39 => Opcode::Digit(opcode as u8),    // b'0'..=b'9'
            0x41..=0x5a => Opcode::NameChar(opcode as u8), // b'A'..=b'Z'
            0x5b01 => Opcode::Mutex,
            0x5b02 => Opcode::Event,
            0x5b12 => Opcode::CondRefOf,
            0x5b13 => Opcode::CreateField,
            0x5b1f => Opcode::LoadTable,
            0x5b20 => Opcode::Load,
            0x5b21 => Opcode::Stall,
            0x5b22 => Opcode::Sleep,
            0x5b23 => Opcode::Acquire,
            0x5b24 => Opcode::Signal,
            0x5b25 => Opcode::Wait,
            0x5b26 => Opcode::Reset,
            0x5b27 => Opcode::Release,
            0x5b28 => Opcode::FromBCD,
            0x5b29 => Opcode::ToBCD,
            0x5b30 => Opcode::Revision,
            0x5b31 => Opcode::Debug,
            0x5b32 => Opcode::Fatal,
            0x5b33 => Opcode::Timer,
            0x5b80 => Opcode::OpRegion,
            0x5b81 => Opcode::Field,
            0x5b82 => Opcode::Device,
            0x5b83 => Opcode::Processor,
            0x5b84 => Opcode::PowerRes,
            0x5b85 => Opcode::ThermalZone,
            0x5b86 => Opcode::IndexField,
            0x5b87 => Opcode::BankField,
            0x5b88 => Opcode::DataRegion,
            0x5c => Opcode::RootChar,
            0x5e => Opcode::ParentPrefixChar,
            0x5f => Opcode::NameChar(b'_'),
            0x60..=0x67 => Opcode::Local(opcode as u8 - 0x60),
            0x68..=0x6e => Opcode::Arg(opcode as u8 - 0x68),
            0x70 => Opcode::Store,
            0x71 => Opcode::RefOf,
            0x72 => Opcode::Add,
            0x73 => Opcode::Concat,
            0x74 => Opcode::Subtract,
            0x75 => Opcode::Increment,
            0x76 => Opcode::Decrement,
            0x77 => Opcode::Multiply,
            0x78 => Opcode::Divide,
            0x79 => Opcode::ShiftLeft,
            0x7a => Opcode::ShiftRight,
            0x7b => Opcode::And,
            0x7c => Opcode::Nand,
            0x7d => Opcode::Or,
            0x7e => Opcode::Nor,
            0x7f => Opcode::Xor,
            0x80 => Opcode::Not,
            0x81 => Opcode::FindSetLeftBit,
            0x82 => Opcode::FindSetRightBit,
            0x83 => Opcode::DerefOf,
            0x84 => Opcode::ConcatRes,
            0x85 => Opcode::Mod,
            0x86 => Opcode::Notify,
            0x87 => Opcode::SizeOf,
            0x88 => Opcode::Index,
            0x89 => Opcode::Match,
            0x8a => Opcode::CreateDWordField,
            0x8b => Opcode::CreateWordField,
            0x8c => Opcode::CreateByteField,
            0x8d => Opcode::CreateBitField,
            0x8e => Opcode::ObjectType,
            0x8f => Opcode::CreateQWordField,
            0x90 => Opcode::LAnd,
            0x91 => Opcode::LOr,
            /*
             * `0x92` is a bit strange. It can be an opcode in its own right (`LNotOp`), but when
             * followed by `0x93..=0x95`, it instead serves as a negating prefix to encode
             * `LNotEqualOp`, `LLessEqualOp`, and `LGreaterEqualOp`.
             */
            0x92 => match self.peek() {
                Ok(0x93) => {
                    self.current_block.pc += 1;
                    Opcode::LNotEqual
                }
                Ok(0x94) => {
                    self.current_block.pc += 1;
                    Opcode::LLessEqual
                }
                Ok(0x95) => {
                    self.current_block.pc += 1;
                    Opcode::LGreaterEqual
                }
                _ => Opcode::LNot,
            },
            0x93 => Opcode::LEqual,
            0x94 => Opcode::LGreater,
            0x95 => Opcode::LLess,
            0x96 => Opcode::ToBuffer,
            0x97 => Opcode::ToDecimalString,
            0x98 => Opcode::ToHexString,
            0x99 => Opcode::ToInteger,
            0x9c => Opcode::ToString,
            0x9d => Opcode::CopyObject,
            0x9e => Opcode::Mid,
            0x9f => Opcode::Continue,
            0xa0 => Opcode::If,
            0xa1 => Opcode::Else,
            0xa2 => Opcode::While,
            0xa3 => Opcode::Noop,
            0xa4 => Opcode::Return,
            0xa5 => Opcode::Break,
            0xcc => Opcode::Breakpoint,
            0xff => Opcode::Ones,

            _ => Err(AmlError::IllegalOpcode(opcode))?,
        })
    }

    fn pkglength(&mut self) -> Result<usize, AmlError> {
        let lead_byte = self.next()?;
        let byte_count = lead_byte.get_bits(6..8);
        if byte_count >= 4 {
            return Err(AmlError::InvalidPackageLength);
        }

        if byte_count == 0 {
            Ok(lead_byte.get_bits(0..6) as usize)
        } else {
            let mut length = lead_byte.get_bits(0..4) as usize;
            for i in 0..byte_count {
                length |= (self.next()? as usize) << (4 + i * 8);
            }
            Ok(length)
        }
    }

    fn namestring(&mut self) -> Result<MeteredNameString, AmlError> {
        use namespace::{NameComponent, NameSeg};

        /*
         * The NameString grammar is actually a little finicky and annoying.
         *
         * NameString := <RootChar NamePath> | <PrefixPath NamePath>
         * PrefixPath := Nothing | <'^' PrefixPath>
         * NamePath := NameSeg | DualNamePath | MultiNamePath | NullName
         * DualNamePath := DualNamePrefix NameSeg NameSeg
         * MultiNamePath := MultiNamePrefix SegCount NameSeg(SegCount)
         */
        const NULL_NAME: u8 = 0x00;
        const DUAL_NAME_PREFIX: u8 = 0x2e;
        const MULTI_NAME_PREFIX: u8 = 0x2f;

        let mut components = allocator::vec_new();
        let mut heap_reservation = NameStringHeapReservation::new(self.execution_meter.clone());

        match self.peek()? {
            b'\\' => {
                self.next()?;
                push_namestring_component(&mut components, NameComponent::Root, &mut heap_reservation)?;
            }
            b'^' => {
                self.next()?;
                push_namestring_component(&mut components, NameComponent::Prefix, &mut heap_reservation)?;
                while self.peek()? == b'^' {
                    self.next()?;
                    push_namestring_component(&mut components, NameComponent::Prefix, &mut heap_reservation)?;
                }
            }
            _ => (),
        }

        let next = self.next()?;
        match next {
            NULL_NAME => {}
            DUAL_NAME_PREFIX => {
                for _ in 0..2 {
                    let name_seg = [self.next()?, self.next()?, self.next()?, self.next()?];
                    push_namestring_component(
                        &mut components,
                        NameComponent::Segment(NameSeg::from_bytes(name_seg)?),
                        &mut heap_reservation,
                    )?;
                }
            }
            MULTI_NAME_PREFIX => {
                let count = self.next()?;
                for _ in 0..count {
                    let name_seg = [self.next()?, self.next()?, self.next()?, self.next()?];
                    push_namestring_component(
                        &mut components,
                        NameComponent::Segment(NameSeg::from_bytes(name_seg)?),
                        &mut heap_reservation,
                    )?;
                }
            }
            first_char => {
                if !namespace::is_lead_name_char(first_char) {
                    self.current_block.pc -= 1;
                }
                let name_seg = [first_char, self.next()?, self.next()?, self.next()?];
                push_namestring_component(
                    &mut components,
                    NameComponent::Segment(NameSeg::from_bytes(name_seg)?),
                    &mut heap_reservation,
                )?;
            }
        }

        Ok(MeteredNameString { name: AmlName::from_components(components), _heap_reservation: heap_reservation })
    }

    fn next(&mut self) -> Result<u8, AmlError> {
        if self.current_block.pc >= self.current_block.stream.len() {
            return Err(AmlError::RunOutOfStream);
        }

        let byte = self.current_block.stream()[self.current_block.pc];
        self.current_block.pc += 1;

        Ok(byte)
    }

    fn next_u16(&mut self) -> Result<u16, AmlError> {
        Ok(u16::from_le_bytes([self.next()?, self.next()?]))
    }

    fn next_u32(&mut self) -> Result<u32, AmlError> {
        Ok(u32::from_le_bytes([self.next()?, self.next()?, self.next()?, self.next()?]))
    }

    fn next_u64(&mut self) -> Result<u64, AmlError> {
        Ok(u64::from_le_bytes([
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
            self.next()?,
        ]))
    }

    fn peek(&self) -> Result<u8, AmlError> {
        if self.current_block.pc >= self.current_block.stream.len() {
            return Err(AmlError::RunOutOfStream);
        }

        Ok(self.current_block.stream()[self.current_block.pc])
    }

    fn expecting_match_opcode(&self) -> bool {
        self.in_flight.last().is_some_and(|op| matches!(op.resolve_behaviour(), Ok(ResolveBehaviour::MatchOpcode)))
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Opcode {
    Zero,
    One,
    MatchOp(MatchOp),
    Alias,
    Name,
    BytePrefix,
    WordPrefix,
    DWordPrefix,
    StringPrefix,
    QWordPrefix,
    Scope,
    Buffer,
    Package,
    VarPackage,
    Method,
    External,
    DualNamePrefix,
    MultiNamePrefix,
    Digit(u8),
    NameChar(u8),
    Mutex,
    Event,
    CondRefOf,
    CreateField,
    LoadTable,
    Load,
    Stall,
    Sleep,
    Acquire,
    Signal,
    Wait,
    Reset,
    Release,
    FromBCD,
    ToBCD,
    Revision,
    Debug,
    Fatal,
    Timer,
    OpRegion,
    Field,
    Device,
    Processor,
    PowerRes,
    ThermalZone,
    IndexField,
    BankField,
    DataRegion,
    RootChar,
    ParentPrefixChar,
    Local(u8),
    Arg(u8),
    Store,
    RefOf,
    Add,
    Concat,
    Subtract,
    Increment,
    Decrement,
    Multiply,
    Divide,
    ShiftLeft,
    ShiftRight,
    And,
    Nand,
    Or,
    Nor,
    Xor,
    Not,
    FindSetLeftBit,
    FindSetRightBit,
    DerefOf,
    ConcatRes,
    Mod,
    Notify,
    SizeOf,
    Index,
    Match,
    CreateDWordField,
    CreateWordField,
    CreateByteField,
    CreateBitField,
    ObjectType,
    CreateQWordField,
    LAnd,
    LOr,
    LNot,
    LNotEqual,
    LLessEqual,
    LGreaterEqual,
    LEqual,
    LGreater,
    LLess,
    ToBuffer,
    ToDecimalString,
    ToHexString,
    ToInteger,
    ToString,
    CopyObject,
    Mid,
    Continue,
    If,
    Else,
    While,
    Noop,
    Return,
    Break,
    Breakpoint,
    Ones,

    /*
     * Internal opcodes are not produced from the bytecode, but are used to track special in-flight
     * ops etc.
     */
    InternalMethodCall,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Operation {
    Mid,
    SizeOf,
    Acquire,
    Release,
    ConvertToBuffer,

    ToBuffer,
    ToInteger,
    ToString,
    ToDecOrHexString,

    ReadBufferField,
    WriteBufferField,
    LogicalOp,
    DecodePrt,
    ParseResource,

    Store,

    ResetEvent,
    SignalEvent,
    WaitEvent,
}

#[derive(Clone, PartialEq, Debug)]
#[non_exhaustive]
pub enum AmlError {
    RunOutOfStream,
    InvalidPackageLength,
    IllegalOpcode(u16),
    InvalidFieldFlags,

    InvalidName(Option<AmlName>),
    /// A metered lookup failed to find an object. The path is omitted so an
    /// error can safely outlive the lookup's bounded path scratch.
    ObjectDoesNotExistWithoutPath,
    /// A metered name resolution was given a non-absolute scope. The path is
    /// omitted so reporting the error does not allocate outside its budget.
    NameNotAbsoluteWithoutPath,
    /// A metered name resolution normalized to an invalid path. The path is
    /// omitted so reporting the error does not retain unmetered heap storage.
    InvalidNormalizedNameWithoutPath,

    InvalidNameSeg([u8; 4]),
    InvalidNormalizedName(AmlName),
    /// An absolute name was required, but a relative one was supplied.
    NameNotAbsolute(AmlName),
    /// A name in a package resolved to another unresolved name too many times, probably because
    /// the names form a cycle.
    NameResolutionLoop,
    /// Storing an AML reference would make an object strongly reference itself.
    ReferenceCycle,
    RootHasNoParent,
    EmptyNamesAreInvalid,
    LevelDoesNotExist(AmlName),
    NameCollision(AmlName),
    ObjectDoesNotExist(AmlName),

    NoCurrentOp,
    ElseFoundWithoutCorrespondingIf,
    ContinueOutsideOfWhile,
    BreakOutsideOfWhile,

    MethodArgCountIncorrect,
    DivisionByZero,

    InvalidOperationOnObject {
        op: Operation,
        typ: ObjectType,
    },
    IndexOutOfBounds,
    ObjectNotOfExpectedType {
        expected: ObjectType,
        got: ObjectType,
    },
    InvalidImplicitCast {
        from: ObjectType,
        to: ObjectType,
    },
    StoreToInvalidReferenceType,

    InvalidResourceDescriptor,
    UnexpectedResourceType,

    NoHandlerForRegionAccess(RegionSpace),
    MutexAcquireTimeout,
    /// The host could not confirm whether the firmware Global Lock changed
    /// ownership. Further AML execution must stop until the executor is torn down.
    GlobalLockOutcomeUnknown,
    /// The host could not confirm whether a native OperationRegion operation
    /// completed. Further AML execution must stop until the executor is torn down.
    NativeRegionOutcomeUnknown,

    PrtInvalidAddress,
    PrtInvalidPin,
    PrtInvalidGsi,
    PrtInvalidSource,
    PrtNoEntry,

    /// An OEM-defined fatal error has occured. The specification states a host should log this
    /// fatal error and then shutdown in a timely fashion.
    FatalErrorEncountered,

    /// This is emitted to signal that the library does not support the requested behaviour. This
    /// should eventually never be emitted.
    LibUnimplemented,

    /// A bounded AML allocation or execution resource was exhausted. This is
    /// kept as a payload-free variant so reporting allocation failure never
    /// needs another heap allocation.
    BudgetExceeded,

    /// An AML work-unit deadline expired. This is payload-free so reporting
    /// a timeout remains reliable when the work-unit heap quota is exhausted.
    DeadlineExceeded,

    /// Native access fuel was exhausted. This remains distinct from the
    /// work-unit budget so firmware degradation policy can reject only the
    /// hardware-accessing method without allocating an error message.
    NativeAccessBudgetExceeded,

    /// The library has given a response the host does not understand, or the host is otherwise
    /// unable to continue operating the library correctly. The specific reason is given in the
    /// contained diagnostic string, allocated outside the bounded AML object heap so an error can
    /// still be reported after the AML quota is exhausted.
    ///
    /// This variant is set by the host, not by the library, and can be used when it is convenient
    /// not to construct a more complex error type around [`AmlError`].
    HostError(alloc::string::String),

    /// An internal interpreter error has occured, and the interpreter has been left in an unknown
    /// state. More information may be given in the contained diagnostic string, allocated outside
    /// the bounded AML object heap.
    InternalError(alloc::string::String),

    /// The maximum stack depths of method calls [`MAX_STACK_DEPTH`] has been reached, so the method
    /// call has been aborted.
    MethodStackExceeded,

    /// The maximum length of time a loop ([`LOOP_TIMEOUT_NS`]) can execute for has been exceeded.
    LoopTimeout,

    /// The provided argument to the current operation is invalid.
    ///
    /// For example, trying to pass an object as the Operation parameter in a Match expression is
    /// invalid.
    InvalidArgType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntegerSize {
    FourBytes = 4,
    EightBytes = 8,
}

impl IntegerSize {
    pub fn from_revision(revision: u8) -> IntegerSize {
        if revision >= 2 { IntegerSize::EightBytes } else { IntegerSize::FourBytes }
    }
}

fn validate_compatibility_method_entry(
    path: &AmlName,
    pic_evaluation_state: u8,
    routing_methods_enabled: bool,
) -> Result<(), AmlError> {
    if path.is_rooted_single_segment(*b"_PIC") && pic_evaluation_state != 1 {
        return Err(AmlError::HostError("NOT_SUPPORTED_PIC_REENTRY".to_string()));
    }
    if !routing_methods_enabled && path.ends_with_segment(*b"_PRT") {
        return Err(AmlError::HostError("NOT_SUPPORTED_PRT_BEFORE_PIC".to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod compatibility_method_entry_tests {
    use super::{AmlError, AmlName, validate_compatibility_method_entry};
    use core::str::FromStr;

    #[test]
    fn top_level_and_nested_method_paths_share_the_pic_prt_gate() {
        let pic = AmlName::from_str("\\_PIC").unwrap();
        let prt = AmlName::from_str("\\_SB.PCI0._PRT").unwrap();

        assert!(matches!(
            validate_compatibility_method_entry(&pic, 0, false),
            Err(AmlError::HostError(message)) if message == "NOT_SUPPORTED_PIC_REENTRY"
        ));
        assert!(validate_compatibility_method_entry(&pic, 1, false).is_ok());
        assert!(matches!(
            validate_compatibility_method_entry(&prt, 0, false),
            Err(AmlError::HostError(message)) if message == "NOT_SUPPORTED_PRT_BEFORE_PIC"
        ));
        assert!(validate_compatibility_method_entry(&prt, 2, true).is_ok());
    }
}

#[cfg(test)]
mod budgeted_allocation_tests {
    use super::{
        AmlError,
        AmlName,
        Argument,
        BlockKind,
        ExecutionMeter,
        IntegerSize,
        MeteredNameString,
        MethodContext,
        NameSeg,
        Object,
        Opcode,
        ReferenceKind,
        ResolveBehaviour,
        buffer_string_capacity,
        concat_buffer_capacity,
        concat_string_capacity,
        integer_string_capacity,
        name_path_heap_bytes,
        namespace::NameComponent,
        string_heap_bytes,
        try_concat_buffer_values,
        try_concat_string_values,
        try_format_buffer_string,
        try_format_integer_string,
        try_lossy_string_prefix,
        try_string_from_str,
        try_string_with_capacity,
        try_vec_from_slice,
        try_vec_with_capacity,
    };
    use alloc::sync::Arc;
    use core::{
        str::FromStr,
        sync::atomic::{AtomicBool, AtomicU64, Ordering},
    };

    struct NamestringBudgetMeter {
        loop_iterations: AtomicU64,
        max_loop_iterations: u64,
        heap_reservations: AtomicU64,
        heap_bytes: AtomicU64,
        reserved_heap_bytes: AtomicU64,
        committed_heap_bytes: AtomicU64,
        released_heap_bytes: AtomicU64,
        allowed_heap_reservations: AtomicU64,
        generation: AtomicU64,
        released_generation: AtomicU64,
    }

    impl ExecutionMeter for NamestringBudgetMeter {
        fn charge_opcode(&self) -> Result<(), AmlError> {
            Ok(())
        }

        fn enter_method(&self) -> Result<(), AmlError> {
            Ok(())
        }

        fn leave_method(&self) {}

        fn loop_iteration(&self) -> Result<(), AmlError> {
            let iteration = self.loop_iterations.fetch_add(1, Ordering::Relaxed);
            if iteration >= self.max_loop_iterations {
                return Err(AmlError::BudgetExceeded);
            }
            Ok(())
        }

        fn reserve_working_heap(&self, bytes: u64) -> Result<(), AmlError> {
            let reservation = self.heap_reservations.fetch_add(1, Ordering::Relaxed);
            if reservation >= self.allowed_heap_reservations.load(Ordering::Relaxed) {
                return Err(AmlError::BudgetExceeded);
            }
            self.heap_bytes.fetch_add(bytes, Ordering::Relaxed);
            self.reserved_heap_bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(())
        }

        fn reserve_working_heap_scoped(&self, bytes: u64) -> Result<u64, AmlError> {
            self.reserve_working_heap(bytes)?;
            Ok(self.generation.load(Ordering::Acquire))
        }

        fn commit_working_heap(&self, bytes: u64) -> Result<(), AmlError> {
            self.heap_bytes.fetch_sub(bytes, Ordering::Relaxed);
            self.committed_heap_bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(())
        }

        fn release_working_heap(&self, bytes: u64) {
            self.heap_bytes.fetch_sub(bytes, Ordering::Relaxed);
            self.released_heap_bytes.fetch_add(bytes, Ordering::Relaxed);
        }

        fn release_working_heap_scoped(&self, bytes: u64, generation: u64) {
            self.released_generation.store(generation, Ordering::Release);
            self.release_working_heap(bytes);
        }
    }

    fn namestring_budget_meter(
        max_loop_iterations: u64,
        allowed_heap_reservations: u64,
    ) -> Arc<NamestringBudgetMeter> {
        allocator::arc_new(NamestringBudgetMeter {
            loop_iterations: AtomicU64::new(0),
            max_loop_iterations,
            heap_reservations: AtomicU64::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            committed_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            allowed_heap_reservations: AtomicU64::new(allowed_heap_reservations),
            generation: AtomicU64::new(1),
            released_generation: AtomicU64::new(0),
        })
    }

    fn namestring_context(stream: &[u8], meter: Arc<NamestringBudgetMeter>) -> MethodContext {
        const TABLE_CONTEXT_RESERVATIONS: u64 = 1 + 2 * (8 + 1);
        let mut allowed = meter.allowed_heap_reservations.load(Ordering::Relaxed);
        loop {
            match meter.allowed_heap_reservations.compare_exchange_weak(
                allowed,
                allowed.saturating_add(TABLE_CONTEXT_RESERVATIONS),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => allowed = observed,
            }
        }
        let meter: Arc<dyn ExecutionMeter> = meter;
        // SAFETY: MethodContext copies the supplied stream before returning.
        unsafe { MethodContext::new_from_table(stream, Some(meter)).unwrap() }
    }

    struct TemporaryAllocationMeter {
        buffer_bytes: AtomicU64,
        working_heap_bytes: AtomicU64,
        reject_buffer: bool,
        reject_heap: bool,
        generation: AtomicU64,
        released_generation: AtomicU64,
    }

    impl super::ExecutionMeter for TemporaryAllocationMeter {
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
            if self.reject_buffer {
                return Err(AmlError::HostError("test buffer budget exhausted".to_string()));
            }
            self.buffer_bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(())
        }

        fn rollback_buffer_reservation(&self, bytes: u64) {
            self.buffer_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }

        fn reserve_working_heap(&self, bytes: u64) -> Result<(), AmlError> {
            if self.reject_heap {
                return Err(AmlError::HostError("test heap budget exhausted".to_string()));
            }
            self.working_heap_bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(())
        }

        fn reserve_working_heap_scoped(&self, bytes: u64) -> Result<u64, AmlError> {
            self.reserve_working_heap(bytes)?;
            Ok(self.generation.load(Ordering::Acquire))
        }

        fn release_working_heap(&self, bytes: u64) {
            self.working_heap_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }

        fn release_working_heap_scoped(&self, bytes: u64, generation: u64) {
            self.released_generation.store(generation, Ordering::Release);
            self.release_working_heap(bytes);
        }

        fn reserve_working_allocation_scoped(
            &self,
            heap_bytes: u64,
            _package_bytes: u64,
            buffer_bytes: u64,
        ) -> Result<u64, AmlError> {
            if self.reject_buffer {
                return Err(AmlError::HostError("test buffer budget exhausted".to_string()));
            }
            self.buffer_bytes.fetch_add(buffer_bytes, Ordering::Relaxed);
            if let Err(error) = self.reserve_working_heap_scoped(heap_bytes) {
                self.buffer_bytes.fetch_sub(buffer_bytes, Ordering::Relaxed);
                return Err(error);
            }
            Ok(self.generation.load(Ordering::Acquire))
        }

        fn release_working_allocation_scoped(
            &self,
            heap_bytes: u64,
            _package_bytes: u64,
            buffer_bytes: u64,
            generation: u64,
        ) {
            self.released_generation.store(generation, Ordering::Release);
            self.buffer_bytes.fetch_sub(buffer_bytes, Ordering::Relaxed);
            self.release_working_heap(heap_bytes);
        }
    }

    fn temporary_allocation_meter(reject_buffer: bool, reject_heap: bool) -> Arc<TemporaryAllocationMeter> {
        allocator::arc_new(TemporaryAllocationMeter {
            buffer_bytes: AtomicU64::new(0),
            working_heap_bytes: AtomicU64::new(0),
            reject_buffer,
            reject_heap,
            generation: AtomicU64::new(1),
            released_generation: AtomicU64::new(0),
        })
    }

    #[test]
    fn temporary_buffer_reservation_rolls_back_on_rejection_and_drop() {
        let rejected_buffer_meter = temporary_allocation_meter(true, false);
        assert!(super::reserve_temporary_buffer(Some(rejected_buffer_meter.clone()), 12).is_err());
        assert_eq!(rejected_buffer_meter.buffer_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(rejected_buffer_meter.working_heap_bytes.load(Ordering::Relaxed), 0);

        let rejected_heap_meter = temporary_allocation_meter(false, true);
        assert!(super::reserve_temporary_buffer(Some(rejected_heap_meter.clone()), 12).is_err());
        assert_eq!(rejected_heap_meter.buffer_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(rejected_heap_meter.working_heap_bytes.load(Ordering::Relaxed), 0);

        let meter = temporary_allocation_meter(false, false);
        let reservation = super::reserve_temporary_buffer(Some(meter.clone()), 12).unwrap();
        assert_eq!(meter.buffer_bytes.load(Ordering::Relaxed), 12);
        assert_eq!(meter.working_heap_bytes.load(Ordering::Relaxed), 36);
        drop(reservation);
        assert_eq!(meter.buffer_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(meter.working_heap_bytes.load(Ordering::Relaxed), 0);

        let stale_meter = temporary_allocation_meter(false, false);
        let stale_reservation = super::reserve_temporary_buffer(Some(stale_meter.clone()), 12).unwrap();
        stale_meter.generation.store(2, Ordering::Release);
        drop(stale_reservation);
        assert_eq!(stale_meter.released_generation.load(Ordering::Acquire), 1);
        assert_eq!(stale_meter.buffer_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(stale_meter.working_heap_bytes.load(Ordering::Relaxed), 0);
    }

    struct ObjectHeapAccountingMeter {
        working_heap: AtomicU64,
        persistent_heap: AtomicU64,
        working_package: AtomicU64,
        persistent_package: AtomicU64,
        working_buffer: AtomicU64,
        persistent_buffer: AtomicU64,
    }

    impl ExecutionMeter for ObjectHeapAccountingMeter {
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

        fn reserve_working_allocation_scoped(
            &self,
            heap_bytes: u64,
            package_bytes: u64,
            buffer_bytes: u64,
        ) -> Result<u64, AmlError> {
            self.working_heap.fetch_add(heap_bytes, Ordering::Relaxed);
            self.working_package.fetch_add(package_bytes, Ordering::Relaxed);
            self.working_buffer.fetch_add(buffer_bytes, Ordering::Relaxed);
            Ok(1)
        }

        fn release_working_allocation_scoped(
            &self,
            heap_bytes: u64,
            package_bytes: u64,
            buffer_bytes: u64,
            _generation: u64,
        ) {
            self.working_heap.fetch_sub(heap_bytes, Ordering::Relaxed);
            self.working_package.fetch_sub(package_bytes, Ordering::Relaxed);
            self.working_buffer.fetch_sub(buffer_bytes, Ordering::Relaxed);
        }

        fn commit_working_allocation_scoped(
            &self,
            heap_bytes: u64,
            package_bytes: u64,
            buffer_bytes: u64,
            _generation: u64,
        ) -> Result<(), AmlError> {
            self.working_heap.fetch_sub(heap_bytes, Ordering::Relaxed);
            self.working_package.fetch_sub(package_bytes, Ordering::Relaxed);
            self.working_buffer.fetch_sub(buffer_bytes, Ordering::Relaxed);
            self.persistent_heap.fetch_add(heap_bytes, Ordering::Relaxed);
            self.persistent_package.fetch_add(package_bytes, Ordering::Relaxed);
            self.persistent_buffer.fetch_add(buffer_bytes, Ordering::Relaxed);
            Ok(())
        }

        fn release_persistent_allocation(
            &self,
            heap_bytes: u64,
            package_bytes: u64,
            buffer_bytes: u64,
        ) -> Result<(), AmlError> {
            self.persistent_heap.fetch_sub(heap_bytes, Ordering::Relaxed);
            self.persistent_package.fetch_sub(package_bytes, Ordering::Relaxed);
            self.persistent_buffer.fetch_sub(buffer_bytes, Ordering::Relaxed);
            Ok(())
        }
    }

    fn object_heap_accounting_meter() -> Arc<ObjectHeapAccountingMeter> {
        allocator::arc_new(ObjectHeapAccountingMeter {
            working_heap: AtomicU64::new(0),
            persistent_heap: AtomicU64::new(0),
            working_package: AtomicU64::new(0),
            persistent_package: AtomicU64::new(0),
            working_buffer: AtomicU64::new(0),
            persistent_buffer: AtomicU64::new(0),
        })
    }

    #[test]
    fn dynamic_object_reservation_follows_clone_replace_commit_and_drop() {
        let meter = object_heap_accounting_meter();
        let trait_meter: Arc<dyn ExecutionMeter> = meter.clone();
        let old_reservation =
            super::reserve_boxed_metered_object_allocation(Some(trait_meter.clone()), 10, 0, 0).unwrap().unwrap();
        let cell_heap_bytes = super::wrapped_object_cell_heap_bytes().unwrap();
        let reservation_box_bytes = super::boxed_metered_reservation_heap_bytes().unwrap();
        let old_cell_reservation =
            super::reserve_boxed_metered_object_allocation(Some(trait_meter.clone()), cell_heap_bytes, 0, 0)
                .unwrap()
                .unwrap();
        let old_heap = cell_heap_bytes + 10 + 2 * reservation_box_bytes;
        let target = Object::String(alloc::string::String::from("old"))
            .wrap_metered(Some(old_cell_reservation), Some(old_reservation));
        let target_clone = target.clone();
        assert_eq!(meter.working_heap.load(Ordering::Relaxed), old_heap);
        drop(target_clone);

        let replacement_reservation =
            super::reserve_boxed_metered_object_allocation(Some(trait_meter.clone()), 20, 0, 0).unwrap().unwrap();
        let replacement_heap = cell_heap_bytes + 20 + 2 * reservation_box_bytes;
        let token = unsafe { super::ObjectToken::create_interpreter_token() };
        unsafe {
            target.replace_metered(
                super::MeteredObject {
                    object: Object::String(alloc::string::String::from("new")),
                    heap_reservation: Some(replacement_reservation),
                },
                &token,
            );
        }
        assert_eq!(meter.working_heap.load(Ordering::Relaxed), replacement_heap);

        target.commit_heap_reservation(&token).unwrap();
        assert_eq!(meter.working_heap.load(Ordering::Relaxed), 0);
        assert_eq!(meter.persistent_heap.load(Ordering::Relaxed), replacement_heap);
        drop(target);
        assert_eq!(meter.persistent_heap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn persistent_package_commits_nested_object_reservations() {
        let meter = object_heap_accounting_meter();
        let trait_meter: Arc<dyn ExecutionMeter> = meter.clone();
        let child_reservation =
            super::reserve_boxed_metered_object_allocation(Some(trait_meter.clone()), 4, 0, 8).unwrap().unwrap();
        let cell_heap_bytes = super::wrapped_object_cell_heap_bytes().unwrap();
        let reservation_box_bytes = super::boxed_metered_reservation_heap_bytes().unwrap();
        let child_cell_reservation =
            super::reserve_boxed_metered_object_allocation(Some(trait_meter.clone()), cell_heap_bytes, 0, 0)
                .unwrap()
                .unwrap();
        let child_heap = cell_heap_bytes + 4 + 2 * reservation_box_bytes;
        let child =
            Object::Buffer(crate::aml_vec![0; 8]).wrap_metered(Some(child_cell_reservation), Some(child_reservation));
        let package_bytes = core::mem::size_of::<super::WrappedObject>() as u64;
        let package_heap = cell_heap_bytes + 12 + 2 * reservation_box_bytes;
        let package_reservation =
            super::reserve_boxed_metered_object_allocation(Some(trait_meter.clone()), 12, package_bytes, 0)
                .unwrap()
                .unwrap();
        let package_cell_reservation =
            super::reserve_boxed_metered_object_allocation(Some(trait_meter), cell_heap_bytes, 0, 0)
                .unwrap()
                .unwrap();
        let package = Object::Package(crate::aml_vec![child])
            .wrap_metered(Some(package_cell_reservation), Some(package_reservation));
        let token = unsafe { super::ObjectToken::create_interpreter_token() };

        package.commit_heap_reservation(&token).unwrap();
        assert_eq!(meter.working_heap.load(Ordering::Relaxed), 0);
        assert_eq!(meter.persistent_heap.load(Ordering::Relaxed), child_heap + package_heap);
        assert_eq!(meter.persistent_package.load(Ordering::Relaxed), package_bytes);
        assert_eq!(meter.persistent_buffer.load(Ordering::Relaxed), 8);

        drop(package);
        assert_eq!(meter.persistent_heap.load(Ordering::Relaxed), 0);
        assert_eq!(meter.persistent_package.load(Ordering::Relaxed), 0);
        assert_eq!(meter.persistent_buffer.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn metered_native_method_output_owns_its_object_cell_lease() {
        let meter = object_heap_accounting_meter();
        let execution_meter: Arc<dyn ExecutionMeter> = meter.clone();
        let native_method =
            Object::native_method_metered(0, |_, meter| super::wrap_metered_object(Object::Integer(7), meter));
        let Object::NativeMethod { f, .. } = native_method else {
            panic!("native method constructor returned a non-method object");
        };

        let result = f(&[], Some(execution_meter)).unwrap();
        assert_eq!(
            meter.working_heap.load(Ordering::Relaxed),
            super::wrapped_object_cell_heap_bytes().unwrap()
                + super::boxed_metered_reservation_heap_bytes().unwrap()
        );
        drop(result);
        assert_eq!(meter.working_heap.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn namestring_parent_prefixes_consume_loop_budget() {
        let meter = namestring_budget_meter(2, u64::MAX);
        let mut context = namestring_context(b"^^^ABCD", meter);

        assert_eq!(context.namestring().map(|_| ()), Err(AmlError::BudgetExceeded));
    }

    #[test]
    fn namestring_multiname_growth_is_bounded() {
        const MAX_NAME_COMPONENTS: usize = 257;
        let mut stream = crate::aml_vec![b'^'; 3];
        stream.extend_from_slice(&[0x2f, u8::MAX]);
        for _ in 0..u8::MAX {
            stream.extend_from_slice(b"AAAA");
        }
        let meter = namestring_budget_meter(1024, u64::MAX);
        let mut context = namestring_context(&stream, meter);

        assert_eq!(context.namestring().map(|_| ()), Err(AmlError::BudgetExceeded));
        assert!(MAX_NAME_COMPONENTS < 3 + usize::from(u8::MAX));
    }

    #[test]
    fn namestring_reserves_heap_before_allocating_components() {
        let meter = namestring_budget_meter(u64::MAX, 0);
        let mut context = namestring_context(b"ABCD", meter);

        assert_eq!(context.namestring().map(|_| ()), Err(AmlError::BudgetExceeded));
    }

    #[test]
    fn in_flight_arguments_are_reserved_before_the_vector_is_built() {
        let meter = namestring_budget_meter(u64::MAX, 0);
        let mut context = namestring_context(b"", meter);
        let argument_vector_built = AtomicBool::new(false);

        let result =
            context.start_with_arguments(Opcode::Add, 1, &[ResolveBehaviour::Placeholder], 1, |arguments| {
                argument_vector_built.store(true, Ordering::Relaxed);
                arguments.push(Argument::Object(Object::Integer(1).wrap()));
                Ok(())
            });

        assert_eq!(result, Err(AmlError::BudgetExceeded));
        assert!(!argument_vector_built.load(Ordering::Relaxed));
    }

    #[test]
    fn namestring_heap_reservation_is_released_with_the_name() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let mut context = namestring_context(b"ABCD", meter.clone());
        let before = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        let name = context.namestring().unwrap();
        let after = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        let released_before = meter.released_heap_bytes.load(Ordering::Relaxed);

        assert!(after > before);
        drop(name);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed) - released_before, after - before);
    }

    #[test]
    fn namestring_heap_reservation_keeps_its_work_unit_generation() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let mut reservation = super::NameStringHeapReservation::new(Some(meter.clone()));
        reservation.reserve_bytes(16).unwrap();
        meter.generation.store(2, Ordering::Release);

        drop(reservation);

        assert_eq!(meter.released_generation.load(Ordering::Acquire), 1);
    }

    #[test]
    fn argument_vector_scratch_releases_after_call_frame_construction() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let before = meter.heap_bytes.load(Ordering::Relaxed);
        let reservation = super::reserve_argument_scratch(Some(meter.clone()), 3).unwrap();
        let during = meter.heap_bytes.load(Ordering::Relaxed);
        assert!(during > before);

        drop(reservation);

        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), before);
    }

    #[test]
    fn method_context_releases_owned_stream_heap_on_drop() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let reserved_before = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        let released_before = meter.released_heap_bytes.load(Ordering::Relaxed);
        let context = namestring_context(b"ABCD", meter.clone());
        let reserved_after = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        let released_after = meter.released_heap_bytes.load(Ordering::Relaxed);
        assert!(reserved_after > reserved_before);
        assert!(released_after > released_before);

        drop(context);

        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed) - released_before,
            meter.reserved_heap_bytes.load(Ordering::Relaxed) - reserved_before
        );
    }

    #[test]
    fn method_context_releases_block_stack_heap_on_drop() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        // SAFETY: The test owns the source bytes for the full constructor call.
        let mut context = unsafe {
            let meter: Arc<dyn ExecutionMeter> = meter.clone();
            MethodContext::new_from_table(b"ABCD", Some(meter)).unwrap()
        };
        context.start_new_block(BlockKind::IfThenBranch, 2).unwrap();

        drop(context);

        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            meter.reserved_heap_bytes.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn method_context_keeps_scope_heap_charged_until_scope_exit() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let mut context = namestring_context(b"", meter.clone());
        let root = AmlName::root();
        let scope = AmlName::from_str("\\_SB_").unwrap().resolve_metered(&root, meter.clone()).unwrap();
        let before =
            meter.reserved_heap_bytes.load(Ordering::Relaxed) - meter.released_heap_bytes.load(Ordering::Relaxed);

        let old_scope = context.enter_scope(scope).unwrap();
        let entered =
            meter.reserved_heap_bytes.load(Ordering::Relaxed) - meter.released_heap_bytes.load(Ordering::Relaxed);
        assert!(entered > before);

        context.leave_scope(old_scope).unwrap();
        let after_exit =
            meter.reserved_heap_bytes.load(Ordering::Relaxed) - meter.released_heap_bytes.load(Ordering::Relaxed);
        assert!(after_exit < entered);

        drop(context);
        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            meter.reserved_heap_bytes.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn execution_context_resolves_names_with_budgeted_scratch() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let context = namestring_context(b"", meter.clone());
        let name = AmlName::from_str("^DEV0").unwrap();
        let scope = AmlName::from_str("\\_SB_.PCI0").unwrap();
        let before_heap = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        let before_iterations = meter.loop_iterations.load(Ordering::Relaxed);

        let resolved = context.resolve_name(&name, &scope).unwrap();
        assert_eq!(resolved.to_string(), "\\_SB_.DEV0");
        let reserved_heap = meter.reserved_heap_bytes.load(Ordering::Relaxed) - before_heap;
        assert!(reserved_heap > 0);
        assert_eq!(meter.loop_iterations.load(Ordering::Relaxed) - before_iterations, 5);

        let released_before = meter.released_heap_bytes.load(Ordering::Relaxed);
        drop(resolved);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed) - released_before, reserved_heap);
    }

    #[test]
    fn execution_context_rejects_name_scratch_before_resolution() {
        let meter = namestring_budget_meter(u64::MAX, 0);
        let context = namestring_context(b"", meter.clone());
        let name = AmlName::from_str("DEV0").unwrap();
        let scope = AmlName::root();

        assert_eq!(context.resolve_name(&name, &scope).err(), Some(AmlError::BudgetExceeded));
        assert_eq!(meter.heap_reservations.load(Ordering::Relaxed), 1 + 2 * (8 + 1) + 1);
    }

    #[test]
    fn runtime_name_string_parsing_reserves_bounded_scratch() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let name = MeteredNameString::parse_name_metered("\\_SB_.PCI0", Some(meter.clone())).unwrap();
        assert_eq!(name.to_string(), "\\_SB_.PCI0");
        let reserved = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        assert!(reserved > 0);
        assert_eq!(meter.loop_iterations.load(Ordering::Relaxed), 10);

        drop(name);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed), reserved);
    }

    #[test]
    fn runtime_name_string_parsing_rejects_overlong_text_before_scanning() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let overlong = "A".repeat(super::namespace::MAX_AML_NAME_COMPONENTS * 5 + 2);

        assert_eq!(
            MeteredNameString::parse_name_metered(&overlong, Some(meter.clone())).err(),
            Some(AmlError::BudgetExceeded)
        );
        assert_eq!(meter.loop_iterations.load(Ordering::Relaxed), 0);
        assert_eq!(meter.heap_reservations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rendered_unresolved_name_is_charged_before_allocation_and_released_with_object() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let mut context = namestring_context(b"", meter.clone());
        let name = AmlName::from_str("\\_SB_.DEV0").unwrap();
        let reserved_before = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        let released_before = meter.released_heap_bytes.load(Ordering::Relaxed);
        let heap_before = meter.heap_bytes.load(Ordering::Relaxed);

        let rendered = context.render_name_object(&name).unwrap();
        assert_eq!(rendered.as_string().unwrap(), "\\_SB_.DEV0");
        let reserved_bytes = meter.reserved_heap_bytes.load(Ordering::Relaxed) - reserved_before;
        assert!(reserved_bytes > 0);
        drop(rendered);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed) - released_before, reserved_bytes);
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), heap_before);

        drop(context);
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn committed_name_path_heap_moves_from_working_to_persistent_accounting() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let name = AmlName::from_str("\\_SB_.PCI0").unwrap();
        let name = MeteredNameString::clone_name_metered(&name, Some(meter.clone())).unwrap();
        let (_, mut reservation) = name.into_parts();
        let reserved_bytes = meter.heap_bytes.load(Ordering::Relaxed);
        assert!(reserved_bytes > 0);

        reservation.commit_to_persistent().unwrap();
        drop(reservation);

        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(meter.committed_heap_bytes.load(Ordering::Relaxed), reserved_bytes);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn partial_heap_commit_releases_only_the_remaining_scratch() {
        let meter = namestring_budget_meter(u64::MAX, u64::MAX);
        let name = AmlName::from_str("\\_SB_.PCI0").unwrap();
        let name = MeteredNameString::clone_name_metered(&name, Some(meter.clone())).unwrap();
        let (_, mut reservation) = name.into_parts();
        let reserved_bytes = meter.heap_bytes.load(Ordering::Relaxed);
        let persistent_bytes = reserved_bytes / 2;

        reservation.commit_bytes_to_persistent(persistent_bytes).unwrap();
        drop(reservation);

        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(meter.committed_heap_bytes.load(Ordering::Relaxed), persistent_bytes);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed), reserved_bytes - persistent_bytes);
    }

    #[test]
    fn package_namepath_heap_bound_includes_both_owned_paths() {
        let name = AmlName::from_str("ABCD").unwrap();
        let scope = AmlName::from_str("\\_SB_").unwrap();
        let expected = name.heap_allocation_upper_bound().unwrap() + scope.clone_heap_upper_bound().unwrap();

        assert_eq!(name_path_heap_bytes(&name, &scope), Ok(expected));
    }

    #[test]
    fn impossible_vector_capacity_returns_budget_error_without_panicking() {
        let result = try_vec_with_capacity::<u8>(usize::MAX);
        let error = result.expect_err("an impossible capacity must be rejected");

        assert_eq!(error, AmlError::BudgetExceeded);
    }

    #[test]
    fn slice_copy_helper_preserves_bytes() {
        let source = [0x5a_u8, 0xa5, 0x00];

        assert_eq!(try_vec_from_slice(&source).unwrap(), source);
    }

    #[test]
    fn lossy_string_copy_preserves_replacement_and_nul_termination() {
        assert_eq!(try_lossy_string_prefix(&[b'A', 0xff, 0, b'B']).unwrap(), "A\u{fffd}");
    }

    #[test]
    fn string_copy_helper_preserves_utf8_without_infallible_growth() {
        assert_eq!(try_string_from_str("ACPI-λ").unwrap(), "ACPI-λ");
    }

    #[test]
    fn impossible_string_capacity_returns_budget_error_without_panicking() {
        let result = try_string_with_capacity(usize::MAX);
        let error = result.expect_err("an impossible string capacity must be rejected");

        assert_eq!(error, AmlError::BudgetExceeded);
    }

    #[test]
    fn integer_string_formatting_preserves_decimal_and_acpi_hex_forms() {
        assert_eq!(try_format_integer_string(0, false).unwrap(), "0");
        assert_eq!(try_format_integer_string(u64::MAX, false).unwrap(), "18446744073709551615");
        assert_eq!(try_format_integer_string(0xabcd, true).unwrap(), "0xABCD");
        assert_eq!(integer_string_capacity(u64::MAX, false), "18446744073709551615".len());
        assert_eq!(integer_string_capacity(u64::MAX, true), "0xFFFFFFFFFFFFFFFF".len());
    }

    #[test]
    fn buffer_string_formatting_is_bounded_and_keeps_delimiters() {
        let bytes = [0, 10, 255];

        assert_eq!(try_format_buffer_string(&bytes, false).unwrap(), "0,10,255");
        assert_eq!(try_format_buffer_string(&bytes, true).unwrap(), "0x00,0x0A,0xFF");
        assert_eq!(try_format_buffer_string(&[], false).unwrap(), "");
        assert_eq!(buffer_string_capacity(3, false), Some(11));
        assert_eq!(buffer_string_capacity(3, true), Some(14));
        assert_eq!(buffer_string_capacity(usize::MAX, true), None);
    }

    #[test]
    fn string_heap_cost_includes_header_only_when_storage_is_allocated() {
        assert_eq!(string_heap_bytes(0), Some(0));
        assert_eq!(string_heap_bytes(18), Some(42));
        assert_eq!(string_heap_bytes(usize::MAX), None);
    }

    #[test]
    fn concat_buffer_conversions_preserve_integer_width_and_source_bytes() {
        let integer_left = Object::Integer(0x1122);
        let integer_right = Object::Integer(0x3344);
        assert_eq!(concat_buffer_capacity(&integer_left, &integer_right, IntegerSize::FourBytes).unwrap(), 8);
        assert_eq!(
            try_concat_buffer_values(&integer_left, &integer_right, IntegerSize::FourBytes, |bytes| {
                assert_eq!(bytes, 8);
                Ok(())
            })
            .unwrap(),
            [0x22, 0x11, 0, 0, 0x44, 0x33, 0, 0]
        );

        let buffer_left = Object::Buffer(crate::aml_vec![0x11, 0x22]);
        let string_right = Object::String(alloc::string::String::from("Z"));
        assert_eq!(concat_buffer_capacity(&buffer_left, &string_right, IntegerSize::EightBytes).unwrap(), 3);
        assert_eq!(
            try_concat_buffer_values(&buffer_left, &string_right, IntegerSize::EightBytes, |bytes| {
                assert_eq!(bytes, 3);
                Ok(())
            })
            .unwrap(),
            [0x11, 0x22, b'Z']
        );
    }

    #[test]
    fn concat_string_conversion_is_lossy_bounded_and_reference_compatible() {
        let left = Object::Buffer(crate::aml_vec![b'A', 0xff]);
        let right = Object::Reference { kind: ReferenceKind::RefOf, inner: Object::Integer(42).wrap() };
        assert_eq!(concat_string_capacity(&left, &right).unwrap(), 6 + 2);
        assert_eq!(
            try_concat_string_values(&left, &right, |bytes| {
                assert_eq!(bytes, 32);
                Ok(())
            })
            .unwrap(),
            "A\u{fffd}42"
        );
    }

    #[test]
    fn concat_string_renders_name_paths_without_intermediate_string_allocations() {
        let name = AmlName::from_components(crate::aml_vec![
            NameComponent::Root,
            NameComponent::Segment(NameSeg::from_bytes(*b"_SB_").unwrap()),
            NameComponent::Segment(NameSeg::from_bytes(*b"PCI0").unwrap()),
        ]);
        let path = Object::NamePath { name, scope: AmlName::root() };
        let empty = Object::String(alloc::string::String::new());

        assert_eq!(concat_string_capacity(&path, &empty).unwrap(), 10);
        assert_eq!(try_concat_string_values(&path, &empty, |_| Ok(())).unwrap(), "\\_SB_.PCI0");
    }
}
