use acpi::aml::{AmlError, ExecutionMeter};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const MAX_OPCODES: usize = 10_000;
const MAX_METHOD_DEPTH: usize = 32;
const MAX_LOOP_ITERATIONS: usize = 10_000;
const MAX_ACCOUNTED_BYTES: u64 = 4 * 1024 * 1024;

pub struct FuzzExecutionMeter {
    opcodes: AtomicUsize,
    method_depth: AtomicUsize,
    loop_iterations: AtomicUsize,
    accounted_bytes: AtomicU64,
}

impl FuzzExecutionMeter {
    pub fn new() -> Self {
        Self {
            opcodes: AtomicUsize::new(0),
            method_depth: AtomicUsize::new(0),
            loop_iterations: AtomicUsize::new(0),
            accounted_bytes: AtomicU64::new(0),
        }
    }

    fn charge_bytes(&self, bytes: u64) -> Result<(), AmlError> {
        let mut current = self.accounted_bytes.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(bytes).ok_or(AmlError::BudgetExceeded)?;
            if next > MAX_ACCOUNTED_BYTES {
                return Err(AmlError::BudgetExceeded);
            }
            match self.accounted_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    fn release_bytes(&self, bytes: u64) {
        let _ = self.accounted_bytes.try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            Some(current.saturating_sub(bytes))
        });
    }
}

impl ExecutionMeter for FuzzExecutionMeter {
    fn charge_opcode(&self) -> Result<(), AmlError> {
        charge_counter(&self.opcodes, MAX_OPCODES)
    }

    fn enter_method(&self) -> Result<(), AmlError> {
        charge_counter(&self.method_depth, MAX_METHOD_DEPTH)
    }

    fn leave_method(&self) {
        self.method_depth.fetch_sub(1, Ordering::AcqRel);
    }

    fn loop_iteration(&self) -> Result<(), AmlError> {
        charge_counter(&self.loop_iterations, MAX_LOOP_ITERATIONS)
    }

    fn reserve_package(&self, bytes: u64) -> Result<(), AmlError> {
        self.charge_bytes(bytes)
    }

    fn reserve_buffer(&self, bytes: u64) -> Result<(), AmlError> {
        self.charge_bytes(bytes)
    }

    fn rollback_buffer_reservation(&self, bytes: u64) {
        self.release_bytes(bytes);
    }

    fn reserve_heap(&self, bytes: u64) -> Result<(), AmlError> {
        self.charge_bytes(bytes)
    }

    fn release_heap(&self, bytes: u64) -> Result<(), AmlError> {
        self.release_bytes(bytes);
        Ok(())
    }

    fn reserve_namespace_object(&self, bytes: u64) -> Result<(), AmlError> {
        self.charge_bytes(bytes)
    }

    fn reserve_working_allocation_scoped(
        &self,
        heap_bytes: u64,
        package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<u64, AmlError> {
        let total = heap_bytes
            .checked_add(package_bytes)
            .and_then(|total| total.checked_add(buffer_bytes))
            .ok_or(AmlError::BudgetExceeded)?;
        self.charge_bytes(total)?;
        Ok(0)
    }

    fn release_working_allocation_scoped(
        &self,
        heap_bytes: u64,
        package_bytes: u64,
        buffer_bytes: u64,
        _generation: u64,
    ) {
        self.release_bytes(heap_bytes.saturating_add(package_bytes).saturating_add(buffer_bytes));
    }

    fn release_persistent_allocation(
        &self,
        heap_bytes: u64,
        package_bytes: u64,
        buffer_bytes: u64,
    ) -> Result<(), AmlError> {
        self.release_bytes(heap_bytes.saturating_add(package_bytes).saturating_add(buffer_bytes));
        Ok(())
    }
}

fn charge_counter(counter: &AtomicUsize, maximum: usize) -> Result<(), AmlError> {
    let observed = counter.fetch_add(1, Ordering::AcqRel);
    if observed < maximum {
        Ok(())
    } else {
        counter.fetch_sub(1, Ordering::AcqRel);
        Err(AmlError::BudgetExceeded)
    }
}
