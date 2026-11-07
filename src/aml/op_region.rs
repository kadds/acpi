use crate::aml::{AmlError, namespace::AmlName};
use super::allocator::{Arc as QuotaArc, Vec};
use alloc::sync::Arc;
use alloc::string::ToString;
use core::any::Any;

#[derive(Clone, Debug)]
pub struct OpRegion {
    pub space: RegionSpace,
    pub base: u64,
    pub length: u64,
    pub parent_device_path: AmlName,
}

/// Opaque authority returned by a native-region backend after validating an
/// OperationRegion. The interpreter only passes this value back to that same
/// backend; it never interprets or serializes the contents.
pub struct NativeRegionToken(Arc<dyn Any + Send + Sync>);

impl NativeRegionToken {
    pub fn new<T: Any + Send + Sync>(token: T) -> Self {
        Self(Arc::new(token))
    }

    pub fn from_arc<T: Any + Send + Sync>(token: Arc<T>) -> Self {
        Self(token)
    }

    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeRegionAccessMode {
    Read,
    ReadWrite,
}

pub trait RegionHandler {
    fn read_u8(&self, region: &OpRegion) -> Result<u8, AmlError>;
    fn read_u16(&self, region: &OpRegion) -> Result<u16, AmlError>;
    fn read_u32(&self, region: &OpRegion) -> Result<u32, AmlError>;
    fn read_u64(&self, region: &OpRegion) -> Result<u64, AmlError>;

    fn write_u8(&self, region: &OpRegion, value: u8) -> Result<(), AmlError>;
    fn write_u16(&self, region: &OpRegion, value: u16) -> Result<(), AmlError>;
    fn write_u32(&self, region: &OpRegion, value: u32) -> Result<(), AmlError>;
    fn write_u64(&self, region: &OpRegion, value: u64) -> Result<(), AmlError>;
}

/// Capability-aware entry point for native OperationRegion accesses.
///
/// The upstream `Handler` methods only receive an already-decoded address and
/// cannot express a NaOS resource lease, byte-range token, or fallible access.
/// Built-in regions therefore enter this trait with the complete region
/// descriptor before any address-space-specific implementation is selected.
/// A host implementation must perform token admission and bounds validation;
/// the interpreter never falls back to the raw `Handler` methods for these
/// spaces.
pub trait NativeRegionHandler: Send + Sync {
    fn authorize(&self, region: &OpRegion, mode: NativeRegionAccessMode) -> Result<NativeRegionToken, AmlError>;

    fn read(&self, token: &NativeRegionToken, offset: usize, length: usize) -> Result<u64, AmlError>;

    fn write(&self, token: &NativeRegionToken, offset: usize, length: usize, value: u64) -> Result<(), AmlError>;

    /// Acquire/release the firmware Global Lock through a host-owned typed
    /// service. The FACS address and its mutable state never cross this API.
    fn acquire_global_lock(&self, timeout_us: u64) -> Result<(), AmlError> {
        let _ = timeout_us;
        Err(AmlError::HostError("NOT_SUPPORTED_GLOBAL_LOCK".to_string()))
    }

    fn release_global_lock(&self) -> Result<(), AmlError> {
        Err(AmlError::HostError("NOT_SUPPORTED_GLOBAL_LOCK".to_string()))
    }

    /// Return a stable identity for the current AML execution thread. The
    /// interpreter uses it only to implement recursive `_GL_` acquisition.
    fn execution_thread_id(&self) -> Option<u64> {
        None
    }

    /// Drop all cached authority after a native-region lifecycle transition.
    fn invalidate_all(&self) -> Result<(), AmlError> {
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum RegionSpace {
    SystemMemory,
    SystemIO,
    PciConfig,
    EmbeddedControl,
    SmBus,
    SystemCmos,
    PciBarTarget,
    Ipmi,
    GeneralPurposeIo,
    GenericSerialBus,
    Pcc,
    Oem(u8),
}

#[derive(Default)]
pub struct NativeRegionRegistrations {
    connected: Option<QuotaArc<Vec<(AmlName, RegionSpace)>>>,
    connected_count: usize,
    dynamic_connected: Option<QuotaArc<Vec<(AmlName, RegionSpace)>>>,
    static_reserved_heap_bytes: u64,
    dynamic_reserved_heap_bytes: u64,
    reserved_heap_bytes: u64,
}

impl NativeRegionRegistrations {
    pub fn install(&mut self, registrations: QuotaArc<Vec<(AmlName, RegionSpace)>>) -> Result<(), AmlError> {
        if self.connected.is_some() || self.dynamic_connected.is_some() || self.reserved_heap_bytes != 0 {
            return Err(AmlError::InternalError("native region registrations already installed".to_string()));
        }
        self.connected = Some(registrations);
        self.connected_count = 0;
        Ok(())
    }

    pub fn contains(&self, scope: &AmlName, space: RegionSpace) -> bool {
        let statically_connected = self
            .connected
            .as_deref()
            .into_iter()
            .flatten()
            .take(self.connected_count)
            .any(|(existing_scope, existing_space)| existing_scope == scope && *existing_space == space);
        statically_connected
            || self
                .dynamic_connected
                .as_deref()
                .into_iter()
                .flatten()
                .any(|(existing_scope, existing_space)| existing_scope == scope && *existing_space == space)
    }

    pub fn contains_registration(&self, scope: &AmlName, space: RegionSpace) -> bool {
        let static_registration = self
            .connected
            .as_deref()
            .into_iter()
            .flatten()
            .any(|(existing_scope, existing_space)| existing_scope == scope && *existing_space == space);
        static_registration
            || self
                .dynamic_connected
                .as_deref()
                .into_iter()
                .flatten()
                .any(|(existing_scope, existing_space)| existing_scope == scope && *existing_space == space)
    }

    pub fn registration_count(&self) -> usize {
        self.connected.as_ref().map_or(0, |registrations| registrations.len())
    }

    pub fn connected_count(&self) -> usize {
        self.connected_count
    }

    pub fn mark_connected(&mut self, count: usize) -> Result<(), AmlError> {
        if count > self.registration_count() {
            return Err(AmlError::InternalError("native region registration index out of bounds".to_string()));
        }
        self.connected_count = count;
        Ok(())
    }

    pub fn disconnect_all(&mut self) {
        self.connected_count = 0;
        self.dynamic_connected = None;
    }

    pub fn snapshot(&self) -> (Option<QuotaArc<Vec<(AmlName, RegionSpace)>>>, usize) {
        (self.connected.as_ref().map(QuotaArc::clone), self.connected_count)
    }

    pub fn dynamic_snapshot(&self) -> Option<QuotaArc<Vec<(AmlName, RegionSpace)>>> {
        self.dynamic_connected.as_ref().map(QuotaArc::clone)
    }

    pub fn replace_dynamic(
        &mut self,
        registrations: Option<QuotaArc<Vec<(AmlName, RegionSpace)>>>,
        reserved_heap_bytes: u64,
    ) -> Result<(Option<QuotaArc<Vec<(AmlName, RegionSpace)>>>, u64), AmlError> {
        let total_reserved_heap_bytes = self
            .static_reserved_heap_bytes
            .checked_add(reserved_heap_bytes)
            .ok_or_else(|| AmlError::BudgetExceeded)?;
        let previous = core::mem::replace(&mut self.dynamic_connected, registrations);
        let previous_heap_bytes = core::mem::replace(&mut self.dynamic_reserved_heap_bytes, reserved_heap_bytes);
        self.reserved_heap_bytes = total_reserved_heap_bytes;
        Ok((previous, previous_heap_bytes))
    }

    pub fn maximum_path_clone_heap_upper_bound(&self) -> Option<u64> {
        self.connected
            .as_deref()?
            .iter()
            .take(self.connected_count)
            .try_fold(0_u64, |maximum, (scope, _)| Some(maximum.max(scope.clone_heap_upper_bound()?)))
            .and_then(|maximum| {
                self.dynamic_connected
                    .as_deref()
                    .into_iter()
                    .flatten()
                    .try_fold(maximum, |maximum, (scope, _)| Some(maximum.max(scope.clone_heap_upper_bound()?)))
            })
    }

    pub fn set_reserved_heap_bytes(&mut self, bytes: u64) {
        self.static_reserved_heap_bytes = bytes;
        self.reserved_heap_bytes = bytes.saturating_add(self.dynamic_reserved_heap_bytes);
    }

    pub fn dynamic_reserved_heap_bytes(&self) -> u64 {
        self.dynamic_reserved_heap_bytes
    }

    pub fn reserved_heap_bytes(&self) -> u64 {
        self.reserved_heap_bytes
    }

    pub fn clear(
        &mut self,
    ) -> (Option<QuotaArc<Vec<(AmlName, RegionSpace)>>>, Option<QuotaArc<Vec<(AmlName, RegionSpace)>>>, u64) {
        let registrations = self.connected.take();
        let dynamic_registrations = self.dynamic_connected.take();
        self.connected_count = 0;
        self.static_reserved_heap_bytes = 0;
        self.dynamic_reserved_heap_bytes = 0;
        (registrations, dynamic_registrations, core::mem::take(&mut self.reserved_heap_bytes))
    }
}

impl From<u8> for RegionSpace {
    fn from(value: u8) -> Self {
        match value {
            0 => RegionSpace::SystemMemory,
            1 => RegionSpace::SystemIO,
            2 => RegionSpace::PciConfig,
            3 => RegionSpace::EmbeddedControl,
            4 => RegionSpace::SmBus,
            5 => RegionSpace::SystemCmos,
            6 => RegionSpace::PciBarTarget,
            7 => RegionSpace::Ipmi,
            8 => RegionSpace::GeneralPurposeIo,
            9 => RegionSpace::GenericSerialBus,
            10 => RegionSpace::Pcc,
            _ => RegionSpace::Oem(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeRegionRegistrations, RegionSpace};
    use crate::aml::namespace::AmlName;
    use alloc::{sync::Arc, vec};
    use core::str::FromStr;

    #[test]
    fn native_region_registration_is_scoped_by_device_and_space() {
        let mut registrations = NativeRegionRegistrations::default();
        let device = AmlName::from_str("\\_SB.PCI0").unwrap();
        let child = AmlName::from_str("\\_SB.PCI0.DEV0").unwrap();

        let entries =
            allocator::arc_new(crate::aml_vec![(device.clone(), RegionSpace::SystemIO), (device.clone(), RegionSpace::SystemMemory)]);
        registrations.install(entries).unwrap();

        assert!(!registrations.contains(&device, RegionSpace::SystemIO));
        assert!(registrations.contains_registration(&device, RegionSpace::SystemIO));
        registrations.mark_connected(1).unwrap();

        assert!(registrations.contains(&device, RegionSpace::SystemIO));
        assert!(!registrations.contains(&device, RegionSpace::SystemMemory));
        registrations.mark_connected(2).unwrap();
        assert!(registrations.contains(&device, RegionSpace::SystemMemory));
        assert!(!registrations.contains(&child, RegionSpace::SystemIO));
        assert_eq!(registrations.snapshot().0.unwrap().len(), 2);

        registrations.clear();
        assert!(registrations.snapshot().0.is_none());
    }

    #[test]
    fn native_region_registration_snapshot_shares_entries_without_copying_paths() {
        let mut registrations = NativeRegionRegistrations::default();
        let device = AmlName::from_str("\\_SB.PCI0").unwrap();
        registrations.install(allocator::arc_new(crate::aml_vec![(device.clone(), RegionSpace::SystemIO)])).unwrap();
        registrations.mark_connected(1).unwrap();

        let (snapshot, connected_count) = registrations.snapshot();
        let snapshot = snapshot.unwrap();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(connected_count, 1);
        assert!(registrations.contains(&device, RegionSpace::SystemIO));
    }

    #[test]
    fn native_region_install_tracks_connected_prefix_and_releases_heap_accounting() {
        let mut registrations = NativeRegionRegistrations::default();
        let device = AmlName::from_str("\\_SB.PCI0").unwrap();
        registrations.install(allocator::arc_new(crate::aml_vec![(device.clone(), RegionSpace::SystemIO)])).unwrap();
        assert!(registrations.contains_registration(&device, RegionSpace::SystemIO));
        assert!(!registrations.contains(&device, RegionSpace::SystemIO));
        registrations.mark_connected(1).unwrap();
        assert!(registrations.contains(&device, RegionSpace::SystemIO));
        registrations.set_reserved_heap_bytes(128);

        let (retired, retired_dynamic, reserved_heap_bytes) = registrations.clear();
        assert_eq!(retired.unwrap().len(), 1);
        assert!(retired_dynamic.is_none());
        assert_eq!(reserved_heap_bytes, 128);
        assert_eq!(registrations.connected_count(), 0);
        assert!(!registrations.contains_registration(&device, RegionSpace::SystemIO));
    }

    #[test]
    fn native_region_install_rejects_replacing_live_registrations() {
        let mut registrations = NativeRegionRegistrations::default();
        let device = AmlName::from_str("\\_SB.PCI0").unwrap();
        registrations.install(allocator::arc_new(crate::aml_vec![(device.clone(), RegionSpace::SystemIO)])).unwrap();

        assert!(registrations.install(allocator::arc_new(crate::aml_vec![(device.clone(), RegionSpace::SystemMemory)])).is_err());
        assert!(registrations.contains_registration(&device, RegionSpace::SystemIO));
        assert!(!registrations.contains_registration(&device, RegionSpace::SystemMemory));
    }
}
