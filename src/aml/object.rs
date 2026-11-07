use crate::aml::{
    AmlError,
    ExecutionMeter,
    Handle,
    IntegerSize,
    Operation,
    namespace::AmlName,
    op_region::OpRegion,
};
use super::allocator::{self, Arc, Box, String, Vec};
use alloc::string::ToString;
use bit_field::BitField;
use core::{cell::UnsafeCell, cmp::Ordering, fmt, ops, sync::atomic::AtomicU64};
use static_assertions::assert_impl_all;

type NativeMethod =
    dyn Fn(&[WrappedObject], Option<Arc<dyn ExecutionMeter>>) -> Result<WrappedObject, AmlError> + Send + Sync;

const MAX_REFERENCE_STORE_DEPTH: usize = 256;
const MAX_REFERENCE_GRAPH_NODES: usize = 65_536;

pub type Package = Vec<WrappedObject>;

#[derive(Clone)]
pub enum Object {
    Uninitialized,
    Buffer(Vec<u8>),
    BufferField { buffer: WrappedObject, offset: usize, length: usize },
    Device,
    Event(Arc<AtomicU64>),
    FieldUnit(FieldUnit),
    Integer(u64),
    Method { code: Vec<u8>, flags: MethodFlags },
    NativeMethod { f: Arc<NativeMethod>, flags: MethodFlags },
    Mutex { mutex: Handle, sync_level: u8 },
    Reference { kind: ReferenceKind, inner: WrappedObject },
    NamePath { name: AmlName, scope: AmlName },
    OpRegion(OpRegion),
    Package(Package),
    PowerResource { system_level: u8, resource_order: u16 },
    Processor { proc_id: u8, pblk_address: u32, pblk_length: u8 },
    RawDataBuffer,
    String(String),
    ThermalZone,
    Debug,
}

impl Object {
    pub fn native_method<F>(num_args: u8, f: F) -> Object
    where
        F: Fn(&[WrappedObject]) -> Result<WrappedObject, AmlError> + 'static + Send + Sync,
    {
        Self::native_method_metered(num_args, move |args, _meter| f(args))
    }

    pub fn native_method_metered<F>(num_args: u8, f: F) -> Object
    where
        F: Fn(&[WrappedObject], Option<Arc<dyn ExecutionMeter>>) -> Result<WrappedObject, AmlError>
            + 'static
            + Send
            + Sync,
    {
        let mut flags = 0;
        flags.set_bits(0..3, num_args);
        Object::NativeMethod { f: allocator::arc_new(f), flags: MethodFlags(flags) }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Object::Uninitialized => write!(f, "[Uninitialized]"),
            Object::Buffer(bytes) => write!(f, "Buffer({bytes:x?})"),
            Object::BufferField { offset, length, .. } => {
                write!(f, "BufferField {{ offset: {offset}, length: {length} }}")
            }
            Object::Device => write!(f, "Device"),
            Object::Event(counter) => write!(f, "Event({counter:?})"),
            // TODO: include fields
            Object::FieldUnit(_) => write!(f, "FieldUnit"),
            Object::Integer(value) => write!(f, "Integer({value})"),
            // TODO: decode flags here
            Object::Method { .. } => write!(f, "Method"),
            Object::NativeMethod { .. } => write!(f, "NativeMethod"),
            Object::Mutex { .. } => write!(f, "Mutex"),
            Object::Reference { kind, inner } => write!(f, "Reference({:?} -> {})", kind, **inner),
            Object::NamePath { name, scope } => write!(f, "NamePath({name} from {scope})"),
            Object::OpRegion(region) => write!(f, "{region:?}"),
            Object::Package(elements) => {
                write!(f, "Package {{ ")?;
                for (i, element) in elements.iter().enumerate() {
                    if i == elements.len() - 1 {
                        write!(f, "{}", **element)?;
                    } else {
                        write!(f, "{}, ", **element)?;
                    }
                }
                write!(f, " }}")?;
                Ok(())
            }
            // TODO: include fields
            Object::PowerResource { .. } => write!(f, "PowerResource"),
            // TODO: include fields
            Object::Processor { .. } => write!(f, "Processor"),
            Object::RawDataBuffer => write!(f, "RawDataBuffer"),
            Object::String(value) => write!(f, "String({value:?})"),
            Object::ThermalZone => write!(f, "ThermalZone"),
            Object::Debug => write!(f, "Debug"),
        }
    }
}

/// `ObjectToken` is used to mediate mutable access to objects from a [`WrappedObject`]. It must be
/// acquired by locking the single token provided by [`super::Interpreter`].
#[non_exhaustive]
pub struct ObjectToken {
    _dont_construct_me: (),
}

impl ObjectToken {
    /// Create an [`ObjectToken`]. This should **only** be done **once** by the main interpreter,
    /// as contructing your own token allows invalid mutable access to objects.
    pub(super) unsafe fn create_interpreter_token() -> ObjectToken {
        ObjectToken { _dont_construct_me: () }
    }
}

// In order for us to assert WrappedObject is Send + Sync, Object must be.
assert_impl_all!(Object: Send, Sync);

#[derive(Clone, Debug)]
pub struct WrappedObject(Arc<WrappedObjectInner>);

pub(crate) trait ObjectHeapReservation: fmt::Debug + Send {
    fn commit_to_persistent(&mut self) -> Result<(), AmlError>;
}

#[derive(Debug)]
struct WrappedObjectInner {
    object: UnsafeCell<Object>,
    cell_heap_reservation: UnsafeCell<Option<Box<dyn ObjectHeapReservation>>>,
    heap_reservation: UnsafeCell<Option<Box<dyn ObjectHeapReservation>>>,
}

pub(crate) fn wrapped_object_inner_heap_bytes() -> Option<usize> {
    core::mem::size_of::<WrappedObjectInner>().checked_add(2 * core::mem::size_of::<usize>())?.checked_add(24)
}

pub(crate) struct MeteredObject {
    pub(crate) object: Object,
    pub(crate) heap_reservation: Option<Box<dyn ObjectHeapReservation>>,
}

unsafe impl Send for WrappedObject {}
unsafe impl Sync for WrappedObject {}

impl WrappedObject {
    pub fn new(object: Object) -> WrappedObject {
        Self::new_metered(object, None, None)
    }

    pub(crate) fn new_metered(
        object: Object,
        cell_heap_reservation: Option<Box<dyn ObjectHeapReservation>>,
        heap_reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> WrappedObject {
        #[allow(clippy::arc_with_non_send_sync)]
        WrappedObject(allocator::arc_new(WrappedObjectInner {
            object: UnsafeCell::new(object),
            cell_heap_reservation: UnsafeCell::new(cell_heap_reservation),
            heap_reservation: UnsafeCell::new(heap_reservation),
        }))
    }

    pub(crate) fn attach_heap_reservation(
        &self,
        reservation: Box<dyn ObjectHeapReservation>,
    ) -> Result<(), AmlError> {
        let slot = unsafe { &mut *self.0.heap_reservation.get() };
        if slot.is_some() {
            return Err(AmlError::InternalError("object heap reservation already attached".to_string()));
        }
        *slot = Some(reservation);
        Ok(())
    }

    pub(crate) fn commit_heap_reservation(&self, _token: &ObjectToken) -> Result<(), AmlError> {
        let mut ancestors = [core::ptr::null(); super::MAX_AML_CONTEXT_STACK_DEPTH];
        self.commit_heap_reservation_recursive(&mut ancestors, 0)
    }

    /// Check whether storing this object's reference graph at `target` would
    /// create a strong Arc cycle. Package and reference values can otherwise
    /// retain their own destination indefinitely.
    pub(crate) fn would_create_reference_cycle(
        &self,
        target: &WrappedObject,
        meter: Option<&Arc<dyn ExecutionMeter>>,
    ) -> Result<bool, AmlError> {
        let mut ancestors = [core::ptr::null(); MAX_REFERENCE_STORE_DEPTH];
        let mut visited_nodes = 0;
        Self::reference_graph_contains_target(self, target, meter, &mut ancestors, 0, &mut visited_nodes)
    }

    fn reference_graph_contains_target(
        current: &WrappedObject,
        target: &WrappedObject,
        meter: Option<&Arc<dyn ExecutionMeter>>,
        ancestors: &mut [*const WrappedObjectInner; MAX_REFERENCE_STORE_DEPTH],
        depth: usize,
        visited_nodes: &mut usize,
    ) -> Result<bool, AmlError> {
        if depth != 0 && Arc::ptr_eq(&current.0, &target.0) {
            return Ok(true);
        }
        *visited_nodes = visited_nodes.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
        if *visited_nodes > MAX_REFERENCE_GRAPH_NODES {
            return Err(AmlError::BudgetExceeded);
        }
        if let Some(meter) = meter {
            meter.loop_iteration()?;
            meter.check_deadline()?;
        }
        if ancestors[..depth].contains(&Arc::as_ptr(&current.0)) {
            return Ok(false);
        }
        if depth >= ancestors.len() {
            return Err(AmlError::BudgetExceeded);
        }
        ancestors[depth] = Arc::as_ptr(&current.0);
        let child_depth = depth + 1;
        match &**current {
            Object::Reference { inner, .. } | Object::BufferField { buffer: inner, .. } => {
                Self::reference_graph_contains_target(inner, target, meter, ancestors, child_depth, visited_nodes)
            }
            Object::Package(elements) => {
                for element in elements {
                    if Self::reference_graph_contains_target(
                        element,
                        target,
                        meter,
                        ancestors,
                        child_depth,
                        visited_nodes,
                    )? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    fn commit_heap_reservation_recursive(
        &self,
        ancestors: &mut [*const WrappedObjectInner; super::MAX_AML_CONTEXT_STACK_DEPTH],
        depth: usize,
    ) -> Result<(), AmlError> {
        let current = Arc::as_ptr(&self.0);
        if ancestors[..depth].contains(&current) {
            return Ok(());
        }
        if depth == ancestors.len() {
            return Err(AmlError::BudgetExceeded);
        }
        ancestors[depth] = current;

        for reservation in
            [unsafe { &mut *self.0.cell_heap_reservation.get() }, unsafe { &mut *self.0.heap_reservation.get() }]
        {
            if let Some(reservation) = reservation {
                reservation.commit_to_persistent()?;
            }
        }

        match &**self {
            Object::BufferField { buffer, .. } | Object::Reference { inner: buffer, .. } => {
                buffer.commit_heap_reservation_recursive(ancestors, depth + 1)?;
            }
            Object::Package(elements) => {
                for element in elements {
                    element.commit_heap_reservation_recursive(ancestors, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) unsafe fn replace_metered(&self, value: MeteredObject, _token: &ObjectToken) {
        unsafe {
            *self.0.object.get() = value.object;
            *self.0.heap_reservation.get() = value.heap_reservation;
        }
    }

    /// Gain a mutable reference to an [`Object`] from this [`WrappedObject`].
    ///
    /// # Safety
    /// This requires an [`ObjectToken`] which is protected by a lock on [`super::Interpreter`],
    /// which prevents mutable access to objects from multiple contexts. It does not, however,
    /// prevent the same object, referenced from multiple [`WrappedObject`]s, having multiple
    /// mutable (and therefore aliasing) references being made to it, and therefore care must be
    /// taken in the interpreter to prevent this.
    pub unsafe fn gain_mut<'r, 'a, 't>(&'a self, _token: &'t ObjectToken) -> &'r mut Object
    where
        't: 'r,
        'a: 'r,
    {
        unsafe { &mut *self.0.object.get() }
    }

    pub fn unwrap_reference(self) -> Result<WrappedObject, AmlError> {
        self.unwrap_reference_with(|_| true)
    }

    /// Unwraps 'transparent' references (e.g. locals, arguments, and internal usage of reference-type objects), but maintain 'real'
    /// references deliberately created by AML.
    pub fn unwrap_transparent_reference(self) -> Result<WrappedObject, AmlError> {
        self.unwrap_reference_with(|kind| {
            matches!(
                kind,
                ReferenceKind::Local | ReferenceKind::Arg | ReferenceKind::Named | ReferenceKind::ImmutableNamed
            )
        })
    }

    fn unwrap_reference_with(
        self,
        should_unwrap: impl Fn(ReferenceKind) -> bool,
    ) -> Result<WrappedObject, AmlError> {
        let mut object = self;
        let mut visited = [core::ptr::null(); MAX_REFERENCE_STORE_DEPTH];
        let mut visited_count = 0;
        loop {
            let current = Arc::as_ptr(&object.0);
            if visited[..visited_count].contains(&current) {
                return Err(AmlError::ReferenceCycle);
            }
            if visited_count == visited.len() {
                return Err(AmlError::BudgetExceeded);
            }
            visited[visited_count] = current;
            visited_count += 1;

            if let Object::Reference { kind, ref inner } = *object
                && should_unwrap(kind)
            {
                object = inner.clone();
            } else {
                return Ok(object.clone());
            }
        }
    }

    /// Unwrap a reference that is about to be stored to - find the target object.
    ///
    /// Take into account the store rules as enumerated by `Interpreter::do_store`
    ///
    /// Returns a tuple containing:
    /// - The object that should be modified
    /// - A boolean indicating whether an implicit cast should occur before the store
    pub(crate) fn unwrap_ref_for_store(self) -> Result<(WrappedObject, bool), AmlError> {
        let Object::Reference { kind: outer_kind, .. } = *self else {
            return Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::Reference, got: self.typ() });
        };

        // If we're storing directly to an Arg then the strange Windows NT behaviour for strings
        // doesn't apply. (see the comments in `Interpreter<H>::do_store` for more)
        let store_to_arg = outer_kind == ReferenceKind::Arg;

        // Set once we've unwrapped through a true reference. This helps decide whether the target
        // should be overwritten in full, or if an implicit cast to the target type is needed.
        let mut crossed_true_reference = false;

        // If we reach an Arg that directly references a Local, we should store in the Arg instead
        // of the Local (see [issue #313](https://github.com/rust-osdev/acpi/issues/313)). Except
        // for the strange Windows NT behaviour handled in the `match arg_to_local` statement below.
        let mut arg_to_local: Option<WrappedObject> = None;

        let mut target = self;
        let mut visited = [core::ptr::null(); MAX_REFERENCE_STORE_DEPTH];
        let mut visited_count = 0;

        loop {
            let current = Arc::as_ptr(&target.0);
            if visited[..visited_count].contains(&current) {
                return Err(AmlError::ReferenceCycle);
            }
            if visited_count == visited.len() {
                return Err(AmlError::BudgetExceeded);
            }
            visited[visited_count] = current;
            visited_count += 1;

            let Object::Reference { kind, ref inner } = *target else {
                return Ok(match arg_to_local {
                    Some(_) if !store_to_arg && target.typ() == ObjectType::String => (target.clone(), true),
                    Some(arg) => (arg, false),
                    None => (target.clone(), !store_to_arg && crossed_true_reference),
                });
            };

            match kind {
                ReferenceKind::ImmutableNamed => return Err(AmlError::StoreToInvalidReferenceType),
                ReferenceKind::Named | ReferenceKind::RefOf | ReferenceKind::Index => {
                    crossed_true_reference = true
                }
                ReferenceKind::Local => {}
                ReferenceKind::Arg => {
                    if arg_to_local.is_none()
                        && matches!(**inner, Object::Reference { kind: ReferenceKind::Local, .. })
                    {
                        arg_to_local = Some(inner.clone());
                    }
                }
                ReferenceKind::Unresolved => return Err(AmlError::StoreToInvalidReferenceType),
            }

            target = inner.clone();
        }
    }
}

impl ops::Deref for WrappedObject {
    type Target = Object;

    fn deref(&self) -> &Self::Target {
        /*
         * SAFETY: elided lifetime ensures reference cannot outlive at least one reference-counted
         * instance of the object. `WrappedObject::gain_mut` is unsafe, and so it is the user's
         * responsibility to ensure shared references from `Deref` do not co-exist with an
         * exclusive reference.
         */
        unsafe { &*self.0.object.get() }
    }
}

impl fmt::Display for WrappedObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wrapped({})", unsafe { &*self.0.object.get() })
    }
}

impl Object {
    pub fn wrap(self) -> WrappedObject {
        WrappedObject::new(self)
    }

    pub(crate) fn wrap_metered(
        self,
        cell_heap_reservation: Option<Box<dyn ObjectHeapReservation>>,
        heap_reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> WrappedObject {
        WrappedObject::new_metered(self, cell_heap_reservation, heap_reservation)
    }

    /// Unwraps an integer object. Errors if not already an integer.
    ///
    /// For casting to integer, use [`Object::to_integer`] instead.
    pub fn as_integer(&self) -> Result<u64, AmlError> {
        if let Object::Integer(value) = self {
            Ok(*value)
        } else {
            Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::Integer, got: self.typ() })
        }
    }

    pub fn as_string(&self) -> Result<&str, AmlError> {
        if let Object::String(value) = self {
            Ok(value)
        } else {
            Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::String, got: self.typ() })
        }
    }

    pub fn as_buffer(&self) -> Result<&[u8], AmlError> {
        if let Object::Buffer(bytes) = self {
            Ok(bytes)
        } else {
            Err(AmlError::ObjectNotOfExpectedType { expected: ObjectType::Buffer, got: self.typ() })
        }
    }

    /// Converts the object to an integer. Used for both implicit and explicit conversions.
    ///
    /// To avoid the cast, use [`Object::as_integer`] instead.
    pub fn to_integer(&self, integer_size: IntegerSize) -> Result<u64, AmlError> {
        match self {
            Object::Integer(value) => Ok(*value),
            Object::Buffer(bytes) => {
                /*
                 * The spec says this should respect the revision of the current definition block.
                 * Apparently, the NT interpreter always uses the first 8 bytes of the buffer.
                 */
                let length = usize::min(bytes.len(), integer_size as usize);
                let mut to_interpret = [0u8; 8];
                to_interpret[0..length].copy_from_slice(&bytes[0..length]);
                Ok(u64::from_le_bytes(to_interpret))
            }
            Object::String(value) => {
                /*
                 * This is about the same level of effort as ACPICA puts in. The uACPI test suite
                 * has tests that this fails - namely because of support for octal, signs, strings
                 * that won't fit in a `u64` etc. We probably need to write a more robust parser
                 * 'real' parser to handle those cases.
                 */
                let value = value.trim();
                let (value, radix): (&str, u32) =
                    match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
                        Some(value) => (value.split(|c: char| !c.is_ascii_hexdigit()).next().unwrap_or(""), 16),
                        None => (value.split(|c: char| !c.is_ascii_digit()).next().unwrap_or(""), 10),
                    };
                match value.len() {
                    0 => Ok(0),
                    _ => Ok(u64::from_str_radix(value, radix).map_err(|_| {
                        AmlError::InvalidOperationOnObject { op: Operation::ToInteger, typ: ObjectType::String }
                    })?),
                }
            }
            Object::BufferField { buffer, offset, length } => {
                let buffer = buffer.clone().unwrap_transparent_reference()?;
                let bytes = match &*buffer {
                    Object::Buffer(bytes) => bytes.as_slice(),
                    Object::String(string) => string.as_bytes(),
                    typ => {
                        return Err(AmlError::InvalidOperationOnObject {
                            op: Operation::ReadBufferField,
                            typ: typ.typ(),
                        });
                    }
                };

                // Integer conversion truncates a BufferField to the AML integer
                // width. Do not materialize the entire field as a temporary
                // Buffer when only its first 32 or 64 bits are observable.
                let source_bits = bytes.len().saturating_mul(8);
                let readable_bits = source_bits.saturating_sub(*offset);
                let integer_bits = (integer_size as usize).saturating_mul(8);
                let bit_length = usize::min(*length, integer_bits).min(readable_bits);
                let mut result = [0_u8; 8];
                copy_bits(bytes, *offset, &mut result, 0, bit_length);
                Ok(u64::from_le_bytes(result))
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ToInteger, typ: self.typ() })?,
        }
    }

    pub fn buffer_conversion_size(&self, integer_size: IntegerSize) -> Result<usize, AmlError> {
        match self {
            Object::Buffer(bytes) => Ok(bytes.len()),
            Object::Integer(_) => Ok(integer_size as usize),
            Object::String(value) => Ok(value.len()),
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::ConvertToBuffer, typ: self.typ() }),
        }
    }

    pub fn to_buffer(&self, integer_size: IntegerSize) -> Result<Vec<u8>, AmlError> {
        let size = self.buffer_conversion_size(integer_size)?;
        let mut bytes = allocator::vec_new();
        bytes.try_reserve_exact(size).map_err(|_| AmlError::BudgetExceeded)?;
        match self {
            Object::Buffer(source) => bytes.extend_from_slice(source),
            Object::Integer(value) => match integer_size {
                IntegerSize::FourBytes => bytes.extend_from_slice(&(*value as u32).to_le_bytes()),
                IntegerSize::EightBytes => bytes.extend_from_slice(&value.to_le_bytes()),
            },
            Object::String(value) => bytes.extend_from_slice(value.as_bytes()),
            _ => unreachable!("buffer_conversion_size validated the object type"),
        }
        Ok(bytes)
    }

    pub fn read_buffer_field(&self, integer_size: IntegerSize) -> Result<Object, AmlError> {
        if let Self::BufferField { buffer, offset, length } = self {
            let buffer = buffer.clone().unwrap_transparent_reference()?;
            let buffer = match &*buffer {
                Object::Buffer(buffer) => buffer.as_slice(),
                Object::String(string) => string.as_bytes(),
                typ => {
                    return Err(AmlError::InvalidOperationOnObject {
                        op: Operation::ReadBufferField,
                        typ: typ.typ(),
                    });
                }
            };
            if *length <= integer_size as usize {
                let mut dst = [0u8; 8];
                copy_bits(buffer, *offset, &mut dst, 0, *length);
                Ok(Object::Integer(u64::from_le_bytes(dst)))
            } else {
                let byte_length = length.div_ceil(8);
                let mut dst = allocator::vec_new();
                dst.try_reserve_exact(byte_length).map_err(|_| AmlError::BudgetExceeded)?;
                dst.resize(byte_length, 0);
                copy_bits(buffer, *offset, &mut dst, 0, *length);
                Ok(Object::Buffer(dst))
            }
        } else {
            Err(AmlError::InvalidOperationOnObject { op: Operation::ReadBufferField, typ: self.typ() })
        }
    }

    pub fn write_buffer_field(&mut self, value: &[u8], token: &ObjectToken) -> Result<(), AmlError> {
        if let Self::BufferField { buffer, offset, length } = self {
            let buffer = buffer.clone().unwrap_transparent_reference()?;
            match unsafe { buffer.gain_mut(token) } {
                Object::Buffer(buffer) => {
                    let destination_bits = buffer.len().checked_mul(8).ok_or(AmlError::BudgetExceeded)?;
                    let field_end = offset.checked_add(*length).ok_or(AmlError::IndexOutOfBounds)?;
                    if field_end > destination_bits {
                        return Err(AmlError::IndexOutOfBounds);
                    }
                    copy_bits(value, 0, buffer, *offset, *length);
                }
                Object::String(string) => {
                    let destination_bits = string.len().checked_mul(8).ok_or(AmlError::BudgetExceeded)?;
                    let field_end = offset.checked_add(*length).ok_or(AmlError::IndexOutOfBounds)?;
                    if field_end > destination_bits {
                        return Err(AmlError::IndexOutOfBounds);
                    }
                    let mut bytes = allocator::vec_from_slice(string.as_bytes());
                    copy_bits(value, 0, &mut bytes, *offset, *length);
                    allocator::replace_string_bytes(string, bytes)
                        .map_err(|_| AmlError::InvalidOperationOnObject {
                            op: Operation::WriteBufferField,
                            typ: ObjectType::String,
                        })?;
                }
                typ => {
                    return Err(AmlError::InvalidOperationOnObject {
                        op: Operation::WriteBufferField,
                        typ: typ.typ(),
                    });
                }
            }
            Ok(())
        } else {
            Err(AmlError::InvalidOperationOnObject { op: Operation::WriteBufferField, typ: self.typ() })
        }
    }

    /// Replace this object's contents with that of a `new` object, applying implicit casting rules
    /// as needed. This follows the NT interpreter's creative interpretation of implicit casts, which is
    /// effectively a byte-wise transmutation.
    pub fn replace_with_implicit_casting(&mut self, new: Object) -> Result<(), AmlError> {
        let new_bytes = match new {
            Object::Integer(value) => &value.to_le_bytes(),
            Object::String(ref value) => value.as_bytes(),
            Object::Buffer(ref value) => &value.clone(),
            _ => return Err(AmlError::InvalidImplicitCast { from: self.typ(), to: new.typ() }),
        };

        match self {
            Object::Integer(value) => {
                let bytes_to_copy = core::cmp::min(new_bytes.len(), 8);
                let mut bytes = [0u8; 8];
                bytes[0..bytes_to_copy].copy_from_slice(&new_bytes[0..bytes_to_copy]);
                *value = u64::from_le_bytes(bytes);
            }
            Object::String(value) => {
                *value = super::try_lossy_string_prefix(new_bytes)?;
            }
            Object::Buffer(value) => {
                *value = allocator::vec_from_slice(new_bytes);
            }
            _ => return Err(AmlError::InvalidImplicitCast { from: self.typ(), to: new.typ() }),
        }

        Ok(())
    }

    /// Returns the `ObjectType` of this object. Returns the type of the referenced object in the
    /// case of `Object::Reference`.
    pub fn typ(&self) -> ObjectType {
        match self {
            Object::Uninitialized => ObjectType::Uninitialized,
            Object::Buffer(_) => ObjectType::Buffer,
            Object::BufferField { .. } => ObjectType::BufferField,
            Object::Device => ObjectType::Device,
            Object::Event(_) => ObjectType::Event,
            Object::FieldUnit(_) => ObjectType::FieldUnit,
            Object::Integer(_) => ObjectType::Integer,
            Object::Method { .. } => ObjectType::Method,
            Object::NativeMethod { .. } => ObjectType::Method,
            Object::Mutex { .. } => ObjectType::Mutex,
            // Object::Reference { inner, .. } => inner.typ(),
            // TODO: maybe this should differentiate internal/real references?
            // An unresolved package name also behaves as a reference until dereferenced.
            Object::Reference { .. } | Object::NamePath { .. } => ObjectType::Reference,
            Object::OpRegion(_) => ObjectType::OpRegion,
            Object::Package(_) => ObjectType::Package,
            Object::PowerResource { .. } => ObjectType::PowerResource,
            Object::Processor { .. } => ObjectType::Processor,
            Object::RawDataBuffer => ObjectType::RawDataBuffer,
            Object::String(_) => ObjectType::String,
            Object::ThermalZone => ObjectType::ThermalZone,
            Object::Debug => ObjectType::Debug,
        }
    }

    /// Calculate the ordering of two objects using the AML rules
    ///
    /// This function is not intended to be used for `impl PartialOrd` because we don't want to tie
    /// the meaning of `object_a.cmp(object_b)` to those AML rules - we may want more flexibility.
    pub fn aml_cmp(&self, other: &Object) -> Result<Ordering, AmlError> {
        match (self, &other) {
            (Object::Integer(a), Object::Integer(b)) => Ok(a.cmp(b)),
            (Object::String(a), Object::String(b)) => Ok(a.cmp(b)),
            (Object::Buffer(a), Object::Buffer(b)) => {
                let size_cmp = a.len().cmp(&b.len());
                if size_cmp != Ordering::Equal {
                    return Ok(size_cmp);
                }
                Ok(a.cmp(b))
            }
            _ => Err(AmlError::InvalidOperationOnObject { op: Operation::LogicalOp, typ: self.typ() }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FieldUnit {
    pub kind: FieldUnitKind,
    pub flags: FieldFlags,
    pub bit_index: usize,
    pub bit_length: usize,
}

#[derive(Clone, Debug)]
pub enum FieldUnitKind {
    Normal { region: WrappedObject },
    Bank { region: WrappedObject, bank: WrappedObject, bank_value: u64 },
    Index { index: WrappedObject, data: WrappedObject },
}

#[derive(Clone, Copy, Debug)]
pub struct FieldFlags(pub u8);

#[derive(Clone, Copy, Debug)]
pub enum FieldAccessType {
    Any,
    Byte,
    Word,
    DWord,
    QWord,
    Buffer,
}

#[derive(Clone, Copy, Debug)]
pub enum FieldUpdateRule {
    Preserve,
    WriteAsOnes,
    WriteAsZeros,
}

impl FieldFlags {
    pub fn access_type(&self) -> Result<FieldAccessType, AmlError> {
        match self.0.get_bits(0..4) {
            0 => Ok(FieldAccessType::Any),
            1 => Ok(FieldAccessType::Byte),
            2 => Ok(FieldAccessType::Word),
            3 => Ok(FieldAccessType::DWord),
            4 => Ok(FieldAccessType::QWord),
            5 => Ok(FieldAccessType::Buffer),
            _ => Err(AmlError::InvalidFieldFlags),
        }
    }

    pub fn access_type_bytes(&self) -> Result<usize, AmlError> {
        match self.access_type()? {
            FieldAccessType::Any => {
                // TODO: given more info about the field, we might be able to make a more efficient
                // read, since all are valid in this case
                Ok(1)
            }
            FieldAccessType::Byte | FieldAccessType::Buffer => Ok(1),
            FieldAccessType::Word => Ok(2),
            FieldAccessType::DWord => Ok(4),
            FieldAccessType::QWord => Ok(8),
        }
    }

    pub fn lock_rule(&self) -> bool {
        self.0.get_bit(4)
    }

    pub fn update_rule(&self) -> Result<FieldUpdateRule, AmlError> {
        match self.0.get_bits(5..7) {
            0 => Ok(FieldUpdateRule::Preserve),
            1 => Ok(FieldUpdateRule::WriteAsOnes),
            2 => Ok(FieldUpdateRule::WriteAsZeros),
            _ => Err(AmlError::InvalidFieldFlags),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MethodFlags(pub u8);

impl MethodFlags {
    pub fn arg_count(&self) -> usize {
        self.0.get_bits(0..3) as usize
    }

    pub fn serialize(&self) -> bool {
        self.0.get_bit(3)
    }

    pub fn sync_level(&self) -> u8 {
        self.0.get_bits(4..8)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ReferenceKind {
    Named,
    ImmutableNamed,
    RefOf,
    Local,
    Arg,
    Index,
    Unresolved,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ObjectType {
    Uninitialized,
    Buffer,
    BufferField,
    Device,
    Event,
    FieldUnit,
    Integer,
    Method,
    Mutex,
    Reference,
    OpRegion,
    Package,
    PowerResource,
    Processor,
    RawDataBuffer,
    String,
    ThermalZone,
    Debug,
}

/// Helper type for decoding the result of `_STA` objects.
pub struct DeviceStatus(pub u64);

impl DeviceStatus {
    pub fn present(&self) -> bool {
        self.0.get_bit(0)
    }

    pub fn enabled(&self) -> bool {
        self.0.get_bit(1)
    }

    pub fn show_in_ui(&self) -> bool {
        self.0.get_bit(2)
    }

    pub fn functioning(&self) -> bool {
        self.0.get_bit(3)
    }

    /// This flag is only used for Battery devices (PNP0C0A), and indicates if the battery is
    /// present.
    pub fn battery_present(&self) -> bool {
        self.0.get_bit(4)
    }
}

/// Copy an arbitrary bit range of `src` to an arbitrary bit range of `dst`. This is used for
/// buffer fields. Data is zero-extended if `src` does not cover `length` bits, matching the
/// expected behaviour for buffer fields.
pub(crate) fn copy_bits(
    src: &[u8],
    mut src_index: usize,
    dst: &mut [u8],
    mut dst_index: usize,
    mut length: usize,
) {
    while length > 0 {
        let src_shift = src_index & 7;
        let mut src_bits = src.get(src_index / 8).unwrap_or(&0x00) >> src_shift;
        if src_shift > 0 && length > (8 - src_shift) {
            src_bits |= src.get(src_index / 8 + 1).unwrap_or(&0x00) << (8 - src_shift);
        }

        if length < 8 {
            src_bits &= (1 << length) - 1;
        }

        let dst_shift = dst_index & 7;
        let mut dst_mask: u16 = if length < 8 { ((1 << length) - 1) as u16 } else { 0xff_u16 } << dst_shift;
        dst[dst_index / 8] =
            (dst[dst_index / 8] & !(dst_mask as u8)) | ((src_bits << dst_shift) & (dst_mask as u8));

        if dst_shift > 0 && length > (8 - dst_shift) {
            dst_mask >>= 8;
            dst[dst_index / 8 + 1] &= !(dst_mask as u8);
            dst[dst_index / 8 + 1] |= (src_bits >> (8 - dst_shift)) & (dst_mask as u8);
        }

        if length < 8 {
            length = 0;
        } else {
            length -= 8;
            src_index += 8;
            dst_index += 8;
        }
    }
}

#[inline]
pub(crate) fn align_down(value: usize, align: usize) -> usize {
    assert!(align == 0 || align.is_power_of_two());

    if align == 0 {
        value
    } else {
        /*
         * Alignment must be a power of two.
         *
         * E.g.
         * align       =   0b00001000
         * align-1     =   0b00000111
         * !(align-1)  =   0b11111000
         * ^^^ Masks the value to the one below it with the correct align
         */
        value & !(align - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_bits() {
        let src = [0b1011_1111, 0b1111_0111, 0b1111_1111, 0b1111_1111, 0b1111_1111];
        let mut dst = [0b1110_0001, 0, 0, 0, 0];

        copy_bits(&src, 0, &mut dst, 2, 15);
        assert_eq!(dst, [0b1111_1101, 0b1101_1110, 0b0000_0001, 0b0000_0000, 0b0000_0000]);
    }

    #[test]
    fn reference_value_cannot_be_stored_into_its_own_target() {
        let target = Object::Integer(1).wrap();
        let source = Object::Reference { kind: ReferenceKind::RefOf, inner: target.clone() }.wrap();
        assert!(source.would_create_reference_cycle(&target, None).unwrap());

        let independent_value = Object::Integer(2).wrap();
        assert!(!independent_value.would_create_reference_cycle(&target, None).unwrap());
    }

    #[test]
    fn buffer_to_integer() {
        let buffer = Object::Buffer(allocator::vec_from_array([0xab, 0xcd, 0xef, 0x01, 0xff]));
        assert_eq!(buffer.to_integer(IntegerSize::FourBytes).unwrap(), 0x01efcdab);
    }

    #[test]
    fn buffer_conversion_preserves_bytes_and_reports_exact_size() {
        let string = Object::String("ACPI".to_string());
        assert_eq!(string.buffer_conversion_size(IntegerSize::EightBytes).unwrap(), 4);
        assert_eq!(string.to_buffer(IntegerSize::EightBytes).unwrap(), b"ACPI");

        let integer = Object::Integer(0x1234);
        assert_eq!(integer.buffer_conversion_size(IntegerSize::FourBytes).unwrap(), 4);
        assert_eq!(integer.to_buffer(IntegerSize::FourBytes).unwrap(), [0x34, 0x12, 0, 0]);
        assert_eq!(integer.to_buffer(IntegerSize::EightBytes).unwrap(), [0x34, 0x12, 0, 0, 0, 0, 0, 0]);

        let buffer = Object::Buffer(crate::aml_vec![0x5a, 0xa5]);
        assert_eq!(buffer.buffer_conversion_size(IntegerSize::FourBytes).unwrap(), 2);
        assert_eq!(buffer.to_buffer(IntegerSize::FourBytes).unwrap(), [0x5a, 0xa5]);
    }
    #[test]
    fn buffer_field_to_integer() {
        const BUFFER: [u8; 5] = [0xffu8; 5];
        let buffer = Object::Buffer(allocator::vec_from_slice(BUFFER)).wrap();
        let buffer_field = Object::BufferField { buffer, offset: 5, length: 9 };

        assert_eq!(buffer_field.to_integer(IntegerSize::FourBytes).unwrap(), 0x1ff);
    }

    #[test]
    fn buffer_field_to_4_byte_integer() {
        // The ones in this buffer are strategically chosen to not make it to the final integer.
        const BUFFER: [u8; 5] = [0x0f, 0x00, 0x00, 0x00, 0xf0];
        let buffer = Object::Buffer(allocator::vec_from_slice(BUFFER)).wrap();
        let buffer_field = Object::BufferField {
            buffer,
            offset: 4,
            length: 36, // This should be truncated to 32 bits in the conversion
        };

        assert_eq!(buffer_field.to_integer(IntegerSize::FourBytes).unwrap(), 0);
    }

    #[test]
    fn buffer_field_to_8_byte_integer() {
        const BUFFER: [u8; 6] = [0x0f, 0x00, 0x00, 0x00, 0xf0, 0xff];
        let buffer = Object::Buffer(allocator::vec_from_slice(BUFFER)).wrap();
        let buffer_field = Object::BufferField { buffer, offset: 4, length: 36 };

        assert_eq!(buffer_field.to_integer(IntegerSize::EightBytes).unwrap(), 0x0000000f_00000000);
    }

    #[test]
    fn oversized_buffer_field_copy_reports_allocation_failure() {
        let buffer = Object::Buffer(allocator::vec_new()).wrap();
        let buffer_field = Object::BufferField { buffer, offset: 0, length: usize::MAX };

        assert!(matches!(buffer_field.read_buffer_field(IntegerSize::EightBytes), Err(AmlError::BudgetExceeded)));
    }

    #[test]
    fn store_local_ref_to_local() {
        // As may be encountered in the last line of:
        // Local1 = RefOf(Local0)
        // Local1 = 2 (the actual store is omitted)
        let local0 = Object::Reference { kind: ReferenceKind::Local, inner: Object::Integer(1).wrap() }.wrap();
        let ref_of = Object::Reference { kind: ReferenceKind::RefOf, inner: local0 }.wrap();
        let local1 = Object::Reference { kind: ReferenceKind::Local, inner: ref_of }.wrap();

        let target = local1.unwrap_ref_for_store();
        let target = target.unwrap();

        let target_obj = &*target.0;
        let Object::Integer(x) = target_obj else {
            panic!("Incorrect type");
        };
        assert_eq!(*x, 1);
    }

    #[test]
    fn immutable_named_reference_cannot_be_a_store_target() {
        let target =
            Object::Reference { kind: ReferenceKind::ImmutableNamed, inner: Object::Integer(1).wrap() }.wrap();
        assert!(matches!(target.unwrap_ref_for_store(), Err(AmlError::StoreToInvalidReferenceType)));
    }

    #[test]
    fn store_arg_ref_to_local() {
        // As if a Local was passed as an argument to a method, and then Arg0 were stored to e.g.:
        // Local0 = 1
        // MEFD(Local0)
        // ... and inside MEFD: Arg0 = 2 (the actual store is omitted)
        let local0 = Object::Reference { kind: ReferenceKind::Local, inner: Object::Integer(1).wrap() }.wrap();
        let arg0 = Object::Reference { kind: ReferenceKind::Arg, inner: local0.clone() }.wrap();

        let target = arg0.unwrap_ref_for_store();
        let (target, implicit_cast_reqd) = target.unwrap();

        assert!(Arc::ptr_eq(&target.0, &local0.0));
        assert!(!implicit_cast_reqd);
    }

    #[test]
    fn store_arg_ref_of_local() {
        // As may be encountered in the last line of:
        // Local0 = 1
        // Arg0 = RefOf(Local0)
        // Arg0 = 2 (the actual store is omitted)
        let local0 = Object::Reference { kind: ReferenceKind::Local, inner: Object::Integer(1).wrap() }.wrap();
        let ref_of = Object::Reference { kind: ReferenceKind::RefOf, inner: local0 }.wrap();
        let arg0 = Object::Reference { kind: ReferenceKind::Arg, inner: ref_of }.wrap();

        let target = arg0.unwrap_ref_for_store();
        let (target, implicit_cast_reqd) = target.unwrap();

        let target_obj = &*target;
        let Object::Integer(x) = target_obj else {
            panic!("Incorrect type");
        };
        assert_eq!(*x, 1);
        assert!(!implicit_cast_reqd);
    }

    #[test]
    fn store_local_refof_arg_to_local_string() {
        // This covers the weird Windows NT handling of references that ultimately end up as Strings
        let string = Object::String("string".to_string()).wrap();
        let outer_local = Object::Reference { kind: ReferenceKind::Local, inner: string.clone() }.wrap();
        let arg0 = Object::Reference { kind: ReferenceKind::Arg, inner: outer_local }.wrap();
        let ref_of = Object::Reference { kind: ReferenceKind::RefOf, inner: arg0 }.wrap();
        let inner_local = Object::Reference { kind: ReferenceKind::Local, inner: ref_of }.wrap();

        let target = inner_local.unwrap_ref_for_store();
        let (target, implicit_cast_reqd) = target.unwrap();

        assert!(Arc::ptr_eq(&target.0, &string.0));
        assert!(implicit_cast_reqd);
    }
}
