use super::{
    AmlError,
    ExecutionMeter,
    Handle,
    MeteredNameString,
    NameStringHeapReservation,
    object::{Object, ObjectHeapReservation, ObjectType, WrappedObject},
    op_region::RegionSpace,
};
use super::allocator::{self, Arc, BTreeMap, Box, String, Vec};
use alloc::string::ToString;
use bit_field::BitField;
use core::{
    fmt,
    str::{self, FromStr},
};
use log::{trace, warn};

const MAX_BOUNDED_NAMESPACE_DEPTH: usize = 256;
pub(crate) const MAX_AML_NAME_COMPONENTS: usize = MAX_BOUNDED_NAMESPACE_DEPTH + 1;

struct WorkingHeapReservation<'a> {
    meter: &'a dyn ExecutionMeter,
    bytes: u64,
    generation: u64,
}

impl Drop for WorkingHeapReservation<'_> {
    fn drop(&mut self) {
        self.meter.release_working_heap_scoped(self.bytes, self.generation);
    }
}

pub struct Namespace {
    root: NamespaceLevel,
}

impl Namespace {
    /// Conservative heap reservation for cloning the namespace's B-tree maps.
    /// It bounds every map node at the pinned Rust B-tree layout's maximum
    /// entry capacity and assumes each non-root node is only minimally full.
    pub fn clone_heap_upper_bound(&self) -> Option<u64> {
        fn map_heap_upper_bound<K, V>(entries: usize) -> Option<u64> {
            if entries == 0 {
                return Some(0);
            }

            const MIN_KEYS_PER_NON_ROOT_NODE: usize = 5;
            const MAX_KEYS_PER_NODE: usize = 11;
            const MAX_CHILDREN_PER_NODE: usize = MAX_KEYS_PER_NODE + 1;
            const NODE_HEADER_BYTES: usize = 32;

            let node_count = entries
                .checked_add(MIN_KEYS_PER_NON_ROOT_NODE - 1)?
                .checked_div(MIN_KEYS_PER_NON_ROOT_NODE)?
                .checked_add(1)?;
            let node_bytes = NODE_HEADER_BYTES
                .checked_add(MAX_KEYS_PER_NODE.checked_mul(core::mem::size_of::<K>())?)?
                .checked_add(MAX_KEYS_PER_NODE.checked_mul(core::mem::size_of::<V>())?)?
                .checked_add(MAX_CHILDREN_PER_NODE.checked_mul(core::mem::size_of::<usize>())?)?;
            u64::try_from(node_count.checked_mul(node_bytes)?).ok()
        }

        fn level_heap_upper_bound(level: &NamespaceLevel, depth: usize) -> Option<u64> {
            if depth > MAX_BOUNDED_NAMESPACE_DEPTH {
                return None;
            }
            let mut bytes = map_heap_upper_bound::<NameSeg, (ObjectFlags, WrappedObject)>(level.values.len())?
                .checked_add(map_heap_upper_bound::<NameSeg, Box<dyn ObjectHeapReservation>>(
                    level.value_heap_reservations.len(),
                )?)?
                .checked_add(map_heap_upper_bound::<NameSeg, NamespaceLevel>(level.children.len())?)?;
            for child in level.children.values() {
                bytes = bytes.checked_add(level_heap_upper_bound(child, depth.checked_add(1)?)?)?;
            }
            Some(bytes)
        }

        level_heap_upper_bound(&self.root, 0)
    }

    /// Upper bound the temporary device candidate vector and path buffers used
    /// by bounded platform discovery. Deep trees fail closed before traversal.
    pub fn device_candidate_heap_upper_bound(&self) -> Option<(usize, u64)> {
        self.device_candidate_heap_upper_bound_for(false, None).ok().flatten()
    }

    pub fn device_like_candidate_heap_upper_bound(&self) -> Option<(usize, u64)> {
        self.device_candidate_heap_upper_bound_for(true, None).ok().flatten()
    }

    pub fn device_candidate_heap_upper_bound_metered(
        &self,
        meter: &dyn ExecutionMeter,
    ) -> Result<(usize, u64), AmlError> {
        self.device_candidate_heap_upper_bound_for(false, Some(meter))?.ok_or(AmlError::BudgetExceeded)
    }

    pub fn device_like_candidate_heap_upper_bound_metered(
        &self,
        meter: &dyn ExecutionMeter,
    ) -> Result<(usize, u64), AmlError> {
        self.device_candidate_heap_upper_bound_for(true, Some(meter))?.ok_or(AmlError::BudgetExceeded)
    }

    fn device_candidate_heap_upper_bound_for(
        &self,
        include_non_device_levels: bool,
        meter: Option<&dyn ExecutionMeter>,
    ) -> Result<Option<(usize, u64)>, AmlError> {
        fn visit(
            level: &NamespaceLevel,
            depth: usize,
            candidate_count: &mut usize,
            candidate_path_bytes: &mut usize,
            maximum_depth: &mut usize,
            include_non_device_levels: bool,
            meter: Option<&dyn ExecutionMeter>,
            visited_levels: &mut usize,
        ) -> Result<Option<()>, AmlError> {
            if depth > MAX_BOUNDED_NAMESPACE_DEPTH {
                return Ok(None);
            }
            if let Some(meter) = meter {
                meter.loop_iteration()?;
                *visited_levels = visited_levels.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
                if *visited_levels % 256 == 0 {
                    meter.check_deadline()?;
                }
            }
            *maximum_depth = (*maximum_depth).max(depth);
            let candidate = level.kind == NamespaceLevelKind::Device
                || (include_non_device_levels
                    && matches!(
                        level.kind,
                        NamespaceLevelKind::Processor
                            | NamespaceLevelKind::ThermalZone
                            | NamespaceLevelKind::PowerResource
                    ));
            if candidate {
                let Some(count) = candidate_count.checked_add(1) else {
                    return Ok(None);
                };
                *candidate_count = count;
                let Some(path_buffer_bytes) = depth
                    .checked_add(1)
                    .and_then(|components| components.checked_mul(core::mem::size_of::<NameComponent>()))
                    .and_then(|bytes| bytes.checked_add(32))
                else {
                    return Ok(None);
                };
                let Some(total_path_bytes) = candidate_path_bytes.checked_add(path_buffer_bytes) else {
                    return Ok(None);
                };
                *candidate_path_bytes = total_path_bytes;
            }
            for child in level.children.values() {
                if visit(
                    child,
                    depth.checked_add(1).ok_or(AmlError::BudgetExceeded)?,
                    candidate_count,
                    candidate_path_bytes,
                    maximum_depth,
                    include_non_device_levels,
                    meter,
                    visited_levels,
                )?
                .is_none()
                {
                    return Ok(None);
                }
            }
            Ok(Some(()))
        }

        let mut candidate_count = 0_usize;
        let mut candidate_path_bytes = 0_usize;
        let mut maximum_depth = 0_usize;
        let mut visited_levels = 0_usize;
        if visit(
            &self.root,
            0,
            &mut candidate_count,
            &mut candidate_path_bytes,
            &mut maximum_depth,
            include_non_device_levels,
            meter,
            &mut visited_levels,
        )?
        .is_none()
        {
            return Ok(None);
        }
        if let Some(meter) = meter {
            meter.check_deadline()?;
        }

        let Some(candidate_records_bytes) = candidate_count
            .checked_mul(core::mem::size_of::<(AmlName, bool, Option<WrappedObject>)>())
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_add(64))
        else {
            return Ok(None);
        };
        let Some(traversal_depth_sum) = maximum_depth
            .checked_mul(maximum_depth.checked_add(3).ok_or(AmlError::BudgetExceeded)?)
            .and_then(|sum| sum.checked_div(2))
        else {
            return Ok(None);
        };
        let Some(traversal_bytes) = traversal_depth_sum
            .checked_mul(core::mem::size_of::<NameComponent>())
            .and_then(|bytes| bytes.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(maximum_depth.checked_mul(64)?))
        else {
            return Ok(None);
        };
        let Some(total_bytes) = candidate_records_bytes
            .checked_add(candidate_path_bytes.checked_mul(2).ok_or(AmlError::BudgetExceeded)?)
            .and_then(|bytes| bytes.checked_add(traversal_bytes))
        else {
            return Ok(None);
        };
        Ok(Some((candidate_count, u64::try_from(total_bytes).ok().ok_or(AmlError::BudgetExceeded)?)))
    }

    /// Upper bound the vector records and rendered `String` buffers for a
    /// complete namespace device-path snapshot. Paths are rendered as a root
    /// marker followed by four-byte segments and separators.
    pub fn device_path_string_heap_upper_bound(&self) -> Option<(usize, u64)> {
        fn visit(
            level: &NamespaceLevel,
            depth: usize,
            device_count: &mut usize,
            string_bytes: &mut usize,
        ) -> Option<()> {
            if depth > MAX_BOUNDED_NAMESPACE_DEPTH {
                return None;
            }
            if level.kind == NamespaceLevelKind::Device {
                *device_count = device_count.checked_add(1)?;
                let rendered_bytes = if depth == 0 { 1 } else { depth.checked_mul(5)? };
                *string_bytes = string_bytes.checked_add(rendered_bytes.checked_add(24)?)?;
            }
            for child in level.children.values() {
                visit(child, depth.checked_add(1)?, device_count, string_bytes)?;
            }
            Some(())
        }

        let mut device_count = 0_usize;
        let mut string_bytes = 0_usize;
        visit(&self.root, 0, &mut device_count, &mut string_bytes)?;
        let vector_bytes = device_count.checked_mul(core::mem::size_of::<String>())?.checked_add(24)?;
        let total_bytes = vector_bytes.checked_add(string_bytes)?;
        Some((device_count, u64::try_from(total_bytes).ok()?))
    }

    /// Meter the complete device-path heap scan before callers allocate the
    /// corresponding snapshot vectors and rendered strings.
    pub fn device_path_string_heap_upper_bound_metered(
        &self,
        meter: &dyn ExecutionMeter,
    ) -> Result<(usize, u64), AmlError> {
        fn visit(
            level: &NamespaceLevel,
            depth: usize,
            device_count: &mut usize,
            string_bytes: &mut usize,
            meter: &dyn ExecutionMeter,
            visited: &mut usize,
        ) -> Result<(), AmlError> {
            if depth > MAX_BOUNDED_NAMESPACE_DEPTH {
                return Err(AmlError::BudgetExceeded);
            }
            meter.loop_iteration()?;
            *visited = visited.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
            if *visited & 0xff == 0 {
                meter.check_deadline()?;
            }
            if level.kind == NamespaceLevelKind::Device {
                *device_count = device_count.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
                let rendered_bytes =
                    if depth == 0 { 1 } else { depth.checked_mul(5).ok_or(AmlError::BudgetExceeded)? };
                *string_bytes = string_bytes
                    .checked_add(rendered_bytes.checked_add(24).ok_or(AmlError::BudgetExceeded)?)
                    .ok_or(AmlError::BudgetExceeded)?;
            }
            for child in level.children.values() {
                visit(
                    child,
                    depth.checked_add(1).ok_or(AmlError::BudgetExceeded)?,
                    device_count,
                    string_bytes,
                    meter,
                    visited,
                )?;
            }
            Ok(())
        }

        let mut device_count = 0_usize;
        let mut string_bytes = 0_usize;
        let mut visited = 0_usize;
        visit(&self.root, 0, &mut device_count, &mut string_bytes, meter, &mut visited)?;
        meter.check_deadline()?;
        let vector_bytes = device_count
            .checked_mul(core::mem::size_of::<String>())
            .and_then(|bytes| bytes.checked_add(24))
            .ok_or(AmlError::BudgetExceeded)?;
        let total_bytes = vector_bytes
            .checked_add(string_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(AmlError::BudgetExceeded)?;
        Ok((device_count, total_bytes))
    }

    /// Upper bound the scratch vectors and rendered strings used to associate
    /// PCI_Config regions with device paths. Region parent strings are counted
    /// before deduplication, so the bound remains valid for the full traversal.
    pub fn pci_config_path_heap_upper_bound(&self) -> Option<(usize, usize, u64)> {
        self.pci_config_path_heap_upper_bound_for(None).ok().flatten()
    }

    pub fn pci_config_path_heap_upper_bound_metered(
        &self,
        meter: &dyn ExecutionMeter,
    ) -> Result<(usize, usize, u64), AmlError> {
        self.pci_config_path_heap_upper_bound_for(Some(meter))?.ok_or(AmlError::BudgetExceeded)
    }

    fn pci_config_path_heap_upper_bound_for(
        &self,
        meter: Option<&dyn ExecutionMeter>,
    ) -> Result<Option<(usize, usize, u64)>, AmlError> {
        fn charge_loop(meter: Option<&dyn ExecutionMeter>, visited: &mut usize) -> Result<(), AmlError> {
            let Some(meter) = meter else {
                return Ok(());
            };
            meter.loop_iteration()?;
            *visited = visited.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
            if *visited % 256 == 0 {
                meter.check_deadline()?;
            }
            Ok(())
        }

        fn visit(
            level: &NamespaceLevel,
            depth: usize,
            device_count: &mut usize,
            region_parent_count: &mut usize,
            string_bytes: &mut usize,
            meter: Option<&dyn ExecutionMeter>,
            visited: &mut usize,
        ) -> Result<Option<()>, AmlError> {
            if depth > MAX_BOUNDED_NAMESPACE_DEPTH {
                return Ok(None);
            }
            charge_loop(meter, visited)?;
            if level.kind == NamespaceLevelKind::Device {
                let Some(count) = device_count.checked_add(1) else {
                    return Ok(None);
                };
                *device_count = count;
                let Some(rendered_bytes) = (if depth == 0 { Some(1) } else { depth.checked_mul(5) })
                    .and_then(|bytes| bytes.checked_add(24))
                else {
                    return Ok(None);
                };
                let Some(total_string_bytes) = string_bytes.checked_add(rendered_bytes) else {
                    return Ok(None);
                };
                *string_bytes = total_string_bytes;
            }
            for (_, value) in level.values.values() {
                charge_loop(meter, visited)?;
                if let Object::OpRegion(region) = &**value
                    && region.space == RegionSpace::PciConfig
                {
                    let Some(parent_count) = region_parent_count.checked_add(1) else {
                        return Ok(None);
                    };
                    *region_parent_count = parent_count;
                    let Some(parent_bytes) =
                        region.parent_device_path.encoded_len().and_then(|bytes| bytes.checked_add(24))
                    else {
                        return Ok(None);
                    };
                    let Some(total_string_bytes) = string_bytes.checked_add(parent_bytes) else {
                        return Ok(None);
                    };
                    *string_bytes = total_string_bytes;
                }
            }
            for child in level.children.values() {
                let Some(next_depth) = depth.checked_add(1) else {
                    return Ok(None);
                };
                if visit(child, next_depth, device_count, region_parent_count, string_bytes, meter, visited)?
                    .is_none()
                {
                    return Ok(None);
                }
            }
            Ok(Some(()))
        }

        let mut device_count = 0_usize;
        let mut region_parent_count = 0_usize;
        let mut string_bytes = 0_usize;
        let mut visited = 0_usize;
        if visit(
            &self.root,
            0,
            &mut device_count,
            &mut region_parent_count,
            &mut string_bytes,
            meter,
            &mut visited,
        )?
        .is_none()
        {
            return Ok(None);
        }
        if let Some(meter) = meter {
            meter.check_deadline()?;
        }
        let Some(vector_bytes) = device_count
            .checked_mul(core::mem::size_of::<String>())
            .and_then(|bytes| bytes.checked_add(24))
            .and_then(|bytes| {
                region_parent_count
                    .checked_mul(core::mem::size_of::<String>())
                    .and_then(|region_bytes| bytes.checked_add(region_bytes))
            })
            .and_then(|bytes| bytes.checked_add(24))
        else {
            return Ok(None);
        };
        let Some(total_bytes) = vector_bytes.checked_add(string_bytes) else {
            return Ok(None);
        };
        let total_bytes = u64::try_from(total_bytes).map_err(|_| AmlError::BudgetExceeded)?;
        Ok(Some((device_count, region_parent_count, total_bytes)))
    }

    /// Create a new AML namespace, with the expected pre-defined objects.
    pub fn new(global_lock_mutex: Handle) -> Namespace {
        let mut namespace = Namespace { root: NamespaceLevel::new(NamespaceLevelKind::Scope) };

        namespace.add_level(AmlName::from_str("\\_GPE").unwrap(), NamespaceLevelKind::Scope).unwrap();
        namespace.add_level(AmlName::from_str("\\_SB").unwrap(), NamespaceLevelKind::Scope).unwrap();
        namespace.add_level(AmlName::from_str("\\_SI").unwrap(), NamespaceLevelKind::Scope).unwrap();
        namespace.add_level(AmlName::from_str("\\_PR").unwrap(), NamespaceLevelKind::Scope).unwrap();
        namespace.add_level(AmlName::from_str("\\_TZ").unwrap(), NamespaceLevelKind::Scope).unwrap();

        namespace
            .insert(
                AmlName::from_str("\\_GL").unwrap(),
                Object::Mutex { mutex: global_lock_mutex, sync_level: 0 }.wrap(),
            )
            .unwrap();

        /*
         * In the dark ages of ACPI 1.0, before `\_OSI`, `\_OS` was used to communicate to the firmware which OS
         * was running. This was predictably not very good, and so was replaced in ACPI 3.0 with `_OSI`, which
         * allows support for individual capabilities to be queried. `_OS` should not be used by modern firmwares;
         * we follow the NT interpreter and ACPICA by calling ourselves `Microsoft Windows NT`.
         *
         * See https://www.kernel.org/doc/html/latest/firmware-guide/acpi/osi.html for more information.
         */
        let mut default_os_name = allocator::string_new_unmetered();
        default_os_name
            .try_reserve_exact("Microsoft Windows NT".len())
            .expect("fixed ACPI compatibility string allocation failed");
        default_os_name.push_str("Microsoft Windows NT");
        namespace
            .insert(AmlName::from_str("\\_OS").unwrap(), Object::String(default_os_name).wrap())
            .unwrap();

        /*
         * `\_OSI` was introduced by ACPI 3.0 to improve the situation created by `\_OS`. Unfortunately, exactly
         * the same problem was immediately repeated by introducing capabilities reflecting that an ACPI
         * implementation is exactly the same as a particular version of Windows' (e.g. firmwares will call
         * `\_OSI("Windows 2001")`).
         *
         * We basically follow suit with whatever Linux does, as this will hopefully minimise breakage:
         *    - We always claim `Windows *` compatability
         *    - We answer 'yes' to `_OSI("Darwin")
         *    - We answer 'no' to `_OSI("Linux")`, and report that the tables are doing the wrong thing
         */
        namespace
            .insert(
                AmlName::from_str("\\_OSI").unwrap(),
                Object::native_method_metered(1, |args, meter| {
                    if args.len() != 1 {
                        return Err(AmlError::MethodArgCountIncorrect);
                    }
                    let Object::String(ref feature) = *args[0].clone().unwrap_reference()? else {
                        return Err(AmlError::ObjectNotOfExpectedType {
                            expected: ObjectType::String,
                            got: args[0].typ(),
                        });
                    };

                    let is_supported = match feature.as_str() {
                        "Windows 2000" => true,       // 2000
                        "Windows 2001" => true,       // XP
                        "Windows 2001 SP1" => true,   // XP SP1
                        "Windows 2001 SP2" => true,   // XP SP2
                        "Windows 2001.1" => true,     // Server 2003
                        "Windows 2001.1 SP1" => true, // Server 2003 SP1
                        "Windows 2006" => true,       // Vista
                        "Windows 2006 SP1" => true,   // Vista SP1
                        "Windows 2006 SP2" => true,   // Vista SP2
                        "Windows 2006.1" => true,     // Server 2008
                        "Windows 2009" => true,       // 7 and Server 2008 R2
                        "Windows 2012" => true,       // 8 and Server 2012
                        "Windows 2013" => true,       // 8.1 and Server 2012 R2
                        "Windows 2015" => true,       // 10
                        "Windows 2016" => true,       // 10 version 1607
                        "Windows 2017" => true,       // 10 version 1703
                        "Windows 2017.2" => true,     // 10 version 1709
                        "Windows 2018" => true,       // 10 version 1803
                        "Windows 2018.2" => true,     // 10 version 1809
                        "Windows 2019" => true,       // 10 version 1903
                        "Windows 2020" => true,       // 10 version 20H1
                        "Windows 2021" => true,       // 11
                        "Windows 2022" => true,       // 11 version 22H2

                        // TODO: Linux answers yes to this, NT answers no. Maybe make configurable
                        "Darwin" => false,

                        "Linux" => {
                            // TODO: should we allow users to specify that this should be true? Linux has a
                            // command line option for this.
                            warn!("ACPI evaluated `_OSI(\"Linux\")`. This is a bug. Reporting no support.");
                            false
                        }

                        "Extended Address Space Descriptor" => true,
                        "Module Device" => true,
                        "3.0 Thermal Model" => true,
                        "3.0 _SCP Extensions" => true,
                        "Processor Aggregator Device" => true,
                        _ => false,
                    };

                    super::wrap_metered_object(Object::Integer(if is_supported { u64::MAX } else { 0 }), meter)
                })
                .wrap(),
            )
            .unwrap();

        /*
         * `\_REV` evaluates to the version of the ACPI specification supported by this interpreter. Linux did this
         * correctly until 2015, but firmwares misused this to detect Linux (as even modern versions of Windows
         * return `2`), and so they switched to just returning `2` (as we'll also do). `_REV` should be considered
         * useless and deprecated (this is mirrored in newer specs, which claim `2` means "ACPI 2 or greater").
         */
        namespace.insert(AmlName::from_str("\\_REV").unwrap(), Object::Integer(2).wrap()).unwrap();

        namespace
    }

    pub fn add_level(&mut self, path: AmlName, kind: NamespaceLevelKind) -> Result<(), AmlError> {
        self.add_level_with_reservation(path, kind, None)
    }

    pub(super) fn add_level_with_reservation(
        &mut self,
        path: AmlName,
        kind: NamespaceLevelKind,
        reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> Result<(), AmlError> {
        let path = path.normalize_absolute()?;

        // Don't try to recreate the root scope
        if path != AmlName::root() {
            let (level, last_seg) = self.get_level_for_path_mut(&path)?;

            /*
             * If the level has already been added, we don't need to add it again. The parser can try to add it
             * multiple times if the ASL contains multiple blocks that add to the same scope/device.
             */
            match level.children.entry(last_seg) {
                alloc::collections::btree_map::Entry::Occupied(_) => drop(reservation),
                alloc::collections::btree_map::Entry::Vacant(entry) => {
                    let mut child = NamespaceLevel::new(kind);
                    child._heap_reservation = reservation;
                    entry.insert(child);
                }
            }
        }

        Ok(())
    }

    pub fn remove_level(&mut self, path: AmlName) -> Result<(), AmlError> {
        let path = path.normalize_absolute()?;

        // Don't try to remove the root scope
        // TODO: we probably shouldn't be able to remove the pre-defined scopes either?
        if path != AmlName::root() {
            let (level, last_seg) = self.get_level_for_path_mut(&path)?;
            level.children.remove(&last_seg);
        }

        Ok(())
    }

    pub fn insert(&mut self, path: AmlName, object: WrappedObject) -> Result<(), AmlError> {
        self.insert_if_mutable(path, object).map(|_| ())
    }

    pub(super) fn insert_if_mutable(&mut self, path: AmlName, object: WrappedObject) -> Result<bool, AmlError> {
        self.insert_if_mutable_with_reservation(path, object, None)
    }

    pub(super) fn insert_if_mutable_with_reservation(
        &mut self,
        path: AmlName,
        object: WrappedObject,
        mut reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> Result<bool, AmlError> {
        let path = path.normalize_absolute()?;

        let (level, last_seg) = self.get_level_for_path_mut(&path)?;
        if level.values.get(&last_seg).is_some_and(|(flags, _)| flags.is_immutable()) {
            return Ok(false);
        }
        if let Some(reservation) = reservation.as_mut() {
            reservation.commit_to_persistent()?;
        }
        match level.values.insert(last_seg, (ObjectFlags::new(false), object)) {
            None => Ok(true),
            Some(_) => {
                /*
                 * Real AML often has name collisions, and so we can't afford to be too strict
                 * about it. We do warn the user as it does have the potential to break stuff.
                 */
                trace!("AML name collision: {}. Replacing object.", path);
                Ok(true)
            }
        }?;
        if let Some(reservation) = reservation {
            level.value_heap_reservations.insert(last_seg, reservation);
        } else {
            level.value_heap_reservations.remove(&last_seg);
        }
        Ok(true)
    }

    /// Insert an interpreter-owned namespace value that firmware AML may
    /// query but may neither replace nor store through a named reference.
    pub fn insert_immutable(&mut self, path: AmlName, object: WrappedObject) -> Result<(), AmlError> {
        let path = path.normalize_absolute()?;
        let (level, last_seg) = self.get_level_for_path_mut(&path)?;
        level.values.insert(last_seg, (ObjectFlags::new_immutable(), object));
        level.value_heap_reservations.remove(&last_seg);
        Ok(())
    }

    pub fn create_alias(&mut self, path: AmlName, object: WrappedObject) -> Result<(), AmlError> {
        self.create_alias_with_reservation(path, object, None)
    }

    pub(super) fn create_alias_with_reservation(
        &mut self,
        path: AmlName,
        object: WrappedObject,
        mut reservation: Option<Box<dyn ObjectHeapReservation>>,
    ) -> Result<(), AmlError> {
        let path = path.normalize_absolute()?;

        let (level, last_seg) = self.get_level_for_path_mut(&path)?;
        if level.values.get(&last_seg).is_some_and(|(flags, _)| flags.is_immutable()) {
            return Err(AmlError::NameCollision(path));
        }
        if level.values.contains_key(&last_seg) {
            return Err(AmlError::NameCollision(path));
        }
        if let Some(reservation) = reservation.as_mut() {
            reservation.commit_to_persistent()?;
        }
        level.values.insert(last_seg, (ObjectFlags::new(true), object));
        if let Some(reservation) = reservation {
            level.value_heap_reservations.insert(last_seg, reservation);
        }
        Ok(())
    }

    pub fn get(&self, path: AmlName) -> Result<WrappedObject, AmlError> {
        let path = path.normalize_absolute()?;

        let (level, last_seg) = self.get_level_for_path(&path)?;
        match level.values.get(&last_seg) {
            Some((_, object)) => Ok(object.clone()),
            None => Err(AmlError::ObjectDoesNotExist(path.clone())),
        }
    }

    /// Look up a canonical absolute path without cloning it on the success path.
    pub fn get_absolute(&self, path: &AmlName) -> Result<WrappedObject, AmlError> {
        match self.get_absolute_if_present(path)? {
            Some(object) => Ok(object),
            None => Err(AmlError::ObjectDoesNotExist(path.try_clone()?)),
        }
    }

    /// Look up a canonical absolute path, returning `None` for a missing leaf
    /// without allocating an error path.
    pub fn get_absolute_if_present(&self, path: &AmlName) -> Result<Option<WrappedObject>, AmlError> {
        path.require_absolute()?;
        let (level, last_seg) = self.get_level_for_path(path)?;
        Ok(level.values.get(&last_seg).map(|(_, object)| object.clone()))
    }

    /// Search for an object at the given path of the namespace, applying the search rules described in §5.3 of the
    /// ACPI specification, if they are applicable. Returns the resolved name, and the handle of the first valid
    /// object, if found. Errors if `starting_scope` is not absolute.
    pub fn search(&self, path: &AmlName, starting_scope: &AmlName) -> Result<(AmlName, WrappedObject), AmlError> {
        let (name, object, _) = self.search_with_flags(path, starting_scope)?;
        Ok((name, object))
    }

    pub fn search_with_flags(
        &self,
        path: &AmlName,
        starting_scope: &AmlName,
    ) -> Result<(AmlName, WrappedObject, ObjectFlags), AmlError> {
        starting_scope.require_absolute()?;

        if path.search_rules_apply() {
            /*
             * If search rules apply, we need to recursively look through the namespace. If the
             * given name does not occur in the current scope, we look at the parent scope, until
             * we either find the name, or reach the root of the namespace.
             */
            let mut scope = starting_scope.clone();
            loop {
                // Search for the name at this namespace level. If we find it, we're done.
                let name = path.resolve(&scope)?;
                match self.get_level_for_path(&name) {
                    Ok((level, last_seg)) => {
                        if let Some((flags, object)) = level.values.get(&last_seg) {
                            return Ok((name, object.clone(), *flags));
                        }
                    }

                    Err(err) => return Err(err),
                }

                // If we don't find it, go up a level in the namespace and search for it there recursively
                match scope.parent() {
                    Ok(parent) => scope = parent,
                    Err(AmlError::RootHasNoParent) => return Err(AmlError::ObjectDoesNotExist(path.clone())),
                    Err(err) => return Err(err),
                }
            }
        } else {
            // If search rules don't apply, simply resolve it against the starting scope
            let name = path.resolve(starting_scope)?;
            let (level, last_seg) = self.get_level_for_path(&path.resolve(starting_scope)?)?;

            if let Some((flags, object)) = level.values.get(&last_seg) {
                Ok((name, object.clone(), *flags))
            } else {
                Err(AmlError::ObjectDoesNotExist(path.clone()))
            }
        }
    }

    pub(super) fn search_with_flags_metered(
        &self,
        path: &AmlName,
        starting_scope: &AmlName,
        meter: Arc<dyn ExecutionMeter>,
    ) -> Result<(MeteredNameString, WrappedObject, ObjectFlags), AmlError> {
        match self.search_with_flags_metered_if_present(path, starting_scope, meter)? {
            Some(result) => Ok(result),
            None => Err(AmlError::ObjectDoesNotExistWithoutPath),
        }
    }

    pub(super) fn search_with_flags_metered_if_present(
        &self,
        path: &AmlName,
        starting_scope: &AmlName,
        meter: Arc<dyn ExecutionMeter>,
    ) -> Result<Option<(MeteredNameString, WrappedObject, ObjectFlags)>, AmlError> {
        if !starting_scope.is_absolute() {
            return Err(AmlError::NameNotAbsoluteWithoutPath);
        }

        if path.search_rules_apply() {
            let mut scope = starting_scope.clone_metered(Arc::clone(&meter))?;
            loop {
                let name = path.resolve_metered(&scope, Arc::clone(&meter))?;
                if let Some((level, last_seg)) = self.get_level_for_path_if_present(&name)? {
                    if let Some((flags, object)) = level.values.get(&last_seg) {
                        return Ok(Some((name, object.clone(), *flags)));
                    }
                }

                match scope.parent_metered(Arc::clone(&meter)) {
                    Ok(parent) => scope = parent,
                    Err(AmlError::RootHasNoParent) => return Ok(None),
                    Err(err) => return Err(err),
                }
            }
        }

        let name = path.resolve_metered(starting_scope, meter)?;
        let Some((level, last_seg)) = self.get_level_for_path_if_present(&name)? else {
            return Ok(None);
        };
        Ok(level.values.get(&last_seg).map(|(flags, object)| (name, object.clone(), *flags)))
    }

    pub fn search_for_level(&self, level_name: &AmlName, starting_scope: &AmlName) -> Result<AmlName, AmlError> {
        starting_scope.require_absolute()?;

        if level_name.search_rules_apply() {
            let mut scope = starting_scope.clone().normalize()?;

            loop {
                let name = level_name.resolve(&scope)?;
                if let Ok((level, last_seg)) = self.get_level_for_path(&name)
                    && level.children.contains_key(&last_seg)
                {
                    return Ok(name);
                }

                // If we don't find it, move the scope up a level and search for it there recursively
                match scope.parent() {
                    Ok(parent) => scope = parent,
                    Err(AmlError::RootHasNoParent) => return Err(AmlError::LevelDoesNotExist(level_name.clone())),
                    Err(err) => return Err(err),
                }
            }
        } else {
            Ok(level_name.clone())
        }
    }

    /// Resolve a namespace level using ACPI search rules while charging the
    /// parent walk and every rendered path component to the active AML work
    /// unit. The returned path retains its temporary heap reservation until
    /// the caller drops it.
    pub fn search_for_level_metered(
        &self,
        level_name: &AmlName,
        starting_scope: &AmlName,
        meter: Arc<dyn ExecutionMeter>,
    ) -> Result<Option<MeteredNameString>, AmlError> {
        if !starting_scope.is_absolute() {
            return Err(AmlError::NameNotAbsoluteWithoutPath);
        }

        if !level_name.search_rules_apply() {
            meter.check_deadline()?;
            return Ok(Some(level_name.clone_metered(meter)?));
        }

        let mut scope = starting_scope.clone_metered(Arc::clone(&meter))?;
        if !scope.name.is_normal() {
            let normalization_heap_bytes = scope.name.clone_heap_upper_bound().ok_or(AmlError::BudgetExceeded)?;
            let mut normalization_reservation = NameStringHeapReservation::new(Some(Arc::clone(&meter)));
            normalization_reservation.reserve_bytes(normalization_heap_bytes)?;
            let MeteredNameString { name, _heap_reservation: source_reservation } = scope;
            let normalized_name = name.normalize()?;
            drop(source_reservation);
            scope = MeteredNameString::with_heap_reservation(normalized_name, normalization_reservation);
        }

        let mut search_iterations = 0_usize;
        loop {
            meter.loop_iteration()?;
            search_iterations = search_iterations.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
            if search_iterations & 0x3f == 0 {
                meter.check_deadline()?;
            }

            let resolved = level_name.resolve_metered(&scope.name, Arc::clone(&meter))?;
            if let Some((level, last_segment)) = self.get_level_for_path_if_present(&resolved.name)?
                && level.children.contains_key(&last_segment)
            {
                meter.check_deadline()?;
                return Ok(Some(resolved));
            }

            drop(resolved);
            match scope.name.parent_metered(Arc::clone(&meter)) {
                Ok(parent) => scope = parent,
                Err(AmlError::RootHasNoParent) => {
                    meter.check_deadline()?;
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Split an absolute path into level segments and a final segment to index into that level.
    fn get_level_for_path(&self, path: &AmlName) -> Result<(&NamespaceLevel, NameSeg), AmlError> {
        let (levels, last_seg) = split_absolute_path(path, true)?;
        let mut current_level = &self.root;
        for (index, level) in levels.iter().enumerate() {
            let NameComponent::Segment(segment) = level else {
                return Err(AmlError::InvalidName(Some(path.try_clone()?)));
            };
            let Some(child) = current_level.children.get(segment) else {
                return Err(AmlError::LevelDoesNotExist(try_path_prefix(path, index + 2)?));
            };
            current_level = child;
        }

        Ok((current_level, last_seg))
    }

    fn get_level_for_path_if_present(
        &self,
        path: &AmlName,
    ) -> Result<Option<(&NamespaceLevel, NameSeg)>, AmlError> {
        let (levels, last_seg) = split_absolute_path(path, false)?;
        let mut current_level = &self.root;
        for level in levels {
            let NameComponent::Segment(segment) = level else {
                return Err(AmlError::InvalidName(None));
            };
            let Some(child) = current_level.children.get(segment) else {
                return Ok(None);
            };
            current_level = child;
        }
        Ok(Some((current_level, last_seg)))
    }

    /// Split an absolute path into level segments and a final segment to index into that level.
    fn get_level_for_path_mut(&mut self, path: &AmlName) -> Result<(&mut NamespaceLevel, NameSeg), AmlError> {
        let (levels, last_seg) = split_absolute_path(path, true)?;

        let mut current_level = &mut self.root;
        for (index, level) in levels.iter().enumerate() {
            let NameComponent::Segment(segment) = level else {
                return Err(AmlError::InvalidName(Some(path.try_clone()?)));
            };
            let Some(child) = current_level.children.get_mut(segment) else {
                return Err(AmlError::LevelDoesNotExist(try_path_prefix(path, index + 2)?));
            };
            current_level = child;
        }

        Ok((current_level, last_seg))
    }

    /// Traverse the namespace without mutation, calling `f` on each namespace level.
    /// `f` returns a `Result<bool, AmlError>` -
    /// errors terminate the traversal and are propagated, and the `bool` on the successful path marks whether the
    /// children of the level should also be traversed.
    pub fn traverse<F>(&self, mut f: F) -> Result<(), AmlError>
    where
        F: FnMut(&AmlName, &NamespaceLevel) -> Result<bool, AmlError>,
    {
        fn traverse_level<F>(
            level: &NamespaceLevel,
            path: &mut AmlName,
            depth: usize,
            f: &mut F,
        ) -> Result<(), AmlError>
        where
            F: FnMut(&AmlName, &NamespaceLevel) -> Result<bool, AmlError>,
        {
            for (name, child) in level.children.iter() {
                let child_depth = depth.checked_add(1).ok_or_else(|| AmlError::BudgetExceeded)?;
                if child_depth > MAX_BOUNDED_NAMESPACE_DEPTH {
                    return Err(AmlError::BudgetExceeded);
                }
                path.0.push(NameComponent::Segment(*name));
                let result = (|| {
                    if f(path, child)? {
                        traverse_level(child, path, child_depth, f)?;
                    }
                    Ok(())
                })();
                path.0.pop();
                result?;
            }

            Ok(())
        }

        let path_capacity = MAX_BOUNDED_NAMESPACE_DEPTH.checked_add(1).ok_or_else(|| AmlError::BudgetExceeded)?;
        let mut components = allocator::vec_new();
        components.try_reserve_exact(path_capacity).map_err(|_| AmlError::BudgetExceeded)?;
        components.push(NameComponent::Root);
        let mut path = AmlName(components);
        if f(&path, &self.root)? {
            traverse_level(&self.root, &mut path, 0, &mut f)?;
        }

        Ok(())
    }

    /// Heap upper bound for the reusable path buffer used by `traverse`.
    pub fn traversal_path_heap_upper_bound() -> Option<u64> {
        let components = MAX_BOUNDED_NAMESPACE_DEPTH.checked_add(1)?;
        let bytes = core::mem::size_of::<Vec<NameComponent>>()
            .checked_add(components.checked_mul(core::mem::size_of::<NameComponent>())?)?;
        u64::try_from(bytes).ok()
    }

    /// Traverse with the host's current work-unit allocation and iteration budget.
    pub fn traverse_metered<F>(&self, meter: &dyn ExecutionMeter, mut f: F) -> Result<(), AmlError>
    where
        F: FnMut(&AmlName, &NamespaceLevel) -> Result<bool, AmlError>,
    {
        let path_heap_bytes = Self::traversal_path_heap_upper_bound().ok_or_else(|| AmlError::BudgetExceeded)?;
        let generation = meter.reserve_working_heap_scoped(path_heap_bytes)?;
        let _path_heap_reservation = WorkingHeapReservation { meter, bytes: path_heap_bytes, generation };
        let mut visited = 0_usize;
        self.traverse(|path, level| {
            visited = visited.checked_add(1).ok_or_else(|| AmlError::BudgetExceeded)?;
            meter.loop_iteration()?;
            if visited & 0xff == 0 {
                meter.check_deadline()?;
            }
            f(path, level)
        })?;
        meter.check_deadline()
    }

    /// Whether an AML object is below a device whose `_GLK` flag requires the
    /// firmware Global Lock. Malformed flag values fail closed.
    pub fn path_requires_global_lock(&self, path: &AmlName) -> bool {
        let Ok(global_lock_flag) = NameSeg::from_bytes(*b"_GLK") else {
            return true;
        };
        if !path.is_absolute() {
            return true;
        }

        fn requires_global_lock(level: &NamespaceLevel, flag: NameSeg) -> bool {
            let Some((_, value)) = level.values.get(&flag) else {
                return false;
            };
            match &**value {
                Object::Integer(value) => *value != 0,
                Object::Reference { inner, .. } => match &**inner {
                    Object::Integer(value) => *value != 0,
                    _ => true,
                },
                _ => true,
            }
        }

        let mut current_level = &self.root;
        if requires_global_lock(current_level, global_lock_flag) {
            return true;
        }
        for component in path.0.iter().skip(1) {
            if requires_global_lock(current_level, global_lock_flag) {
                return true;
            }
            let NameComponent::Segment(segment) = component else {
                return true;
            };
            let Some(child) = current_level.children.get(segment) else {
                break;
            };
            current_level = child;
        }
        requires_global_lock(current_level, global_lock_flag)
    }
}

fn try_path_prefix(path: &AmlName, component_count: usize) -> Result<AmlName, AmlError> {
    let components = path
        .0
        .get(..component_count)
        .ok_or_else(|| AmlError::InternalError("namespace path prefix out of range".to_string()))?;
    let mut copy = allocator::vec_new();
    copy.try_reserve_exact(components.len()).map_err(|_| AmlError::BudgetExceeded)?;
    copy.extend_from_slice(components);
    Ok(AmlName(copy))
}

fn split_absolute_path(
    path: &AmlName,
    include_diagnostic_path: bool,
) -> Result<(&[NameComponent], NameSeg), AmlError> {
    let invalid_name = || {
        if include_diagnostic_path {
            AmlError::InvalidName(Some(path.clone()))
        } else {
            AmlError::InvalidName(None)
        }
    };
    if path.0.first() != Some(&NameComponent::Root) || path.0.len() < 2 {
        return Err(invalid_name());
    }

    let (last_component, levels) = path.0[1..].split_last().ok_or_else(invalid_name)?;
    let NameComponent::Segment(last_segment) = last_component else {
        return Err(invalid_name());
    };
    if levels.iter().any(|component| !matches!(component, NameComponent::Segment(_))) {
        return Err(invalid_name());
    }

    Ok((levels, *last_segment))
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const STEM: &str = "│   ";
        const BRANCH: &str = "├── ";
        const END: &str = "└── ";

        fn print_level(f: &mut fmt::Formatter<'_>, level: &NamespaceLevel, indent_stack: String) -> fmt::Result {
            for (i, (name, (flags, object))) in level.values.iter().enumerate() {
                let end = (i == level.values.len() - 1)
                    && level.children.iter().filter(|(_, l)| l.kind == NamespaceLevelKind::Scope).count() == 0;
                writeln!(
                    f,
                    "{}{}{}: {}{}",
                    indent_stack,
                    if end { END } else { BRANCH },
                    name.as_str(),
                    if flags.is_alias() { "[A] " } else { "" },
                    **object
                )?;

                // If the object has a corresponding scope, print it here
                if let Some(child_level) = level.children.get(name) {
                    print_level(
                        f,
                        child_level,
                        if end { indent_stack.clone() + "    " } else { indent_stack.clone() + STEM },
                    )?;
                }
            }

            let mut remaining_scopes = allocator::vec_new_unmetered();
            for (name, child) in &level.children {
                if child.kind == NamespaceLevelKind::Scope {
                    remaining_scopes.push((name, child));
                }
            }
            for (i, (name, sub_level)) in remaining_scopes.iter().enumerate() {
                let end = i == remaining_scopes.len() - 1;
                writeln!(f, "{}{}{}:", indent_stack, if end { END } else { BRANCH }, name.as_str())?;
                print_level(f, sub_level, indent_stack.clone() + STEM)?;
            }

            Ok(())
        }

        writeln!(f, "\n    \\:")?;
        let mut initial_indent = allocator::string_new_unmetered();
        initial_indent.push_str("    ");
        print_level(f, &self.root, initial_indent)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NamespaceLevelKind {
    Scope,
    Device,
    Processor,
    PowerResource,
    ThermalZone,
    MethodLocals,
}

pub struct NamespaceLevel {
    pub kind: NamespaceLevelKind,
    pub values: BTreeMap<NameSeg, (ObjectFlags, WrappedObject)>,
    pub children: BTreeMap<NameSeg, NamespaceLevel>,
    value_heap_reservations: BTreeMap<NameSeg, Box<dyn ObjectHeapReservation>>,
    _heap_reservation: Option<Box<dyn ObjectHeapReservation>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ObjectFlags(u8);

impl ObjectFlags {
    pub fn new(is_alias: bool) -> ObjectFlags {
        let mut flags = 0;
        flags.set_bit(0, is_alias);
        ObjectFlags(flags)
    }

    pub fn new_immutable() -> ObjectFlags {
        ObjectFlags(1 << 1)
    }

    pub fn is_alias(&self) -> bool {
        self.0.get_bit(0)
    }

    pub fn is_immutable(&self) -> bool {
        self.0.get_bit(1)
    }
}

impl NamespaceLevel {
    pub fn new(kind: NamespaceLevelKind) -> NamespaceLevel {
        NamespaceLevel {
            kind,
            values: allocator::btree_map_new(),
            children: allocator::btree_map_new(),
            value_heap_reservations: allocator::btree_map_new(),
            _heap_reservation: None,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct AmlName(Vec<NameComponent>);

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NameComponent {
    Root,
    Prefix,
    Segment(NameSeg),
}

impl AmlName {
    /// Clone a path without turning allocator exhaustion into an infallible
    /// `Vec::clone` panic.
    pub fn try_clone(&self) -> Result<AmlName, AmlError> {
        if self.0.len() > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        let mut components = allocator::vec_new();
        components.try_reserve_exact(self.0.len()).map_err(|_| AmlError::BudgetExceeded)?;
        components.extend_from_slice(&self.0);
        Ok(AmlName(components))
    }

    /// Construct an absolute path from name segments with one fallible allocation.
    pub fn try_from_absolute_segments(segments: &[NameSeg]) -> Result<AmlName, AmlError> {
        let component_count = segments.len().checked_add(1).ok_or_else(|| AmlError::BudgetExceeded)?;
        if component_count > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        let mut components = allocator::vec_new();
        components.try_reserve_exact(component_count).map_err(|_| AmlError::BudgetExceeded)?;
        components.push(NameComponent::Root);
        components.extend(segments.iter().copied().map(NameComponent::Segment));
        Ok(AmlName(components))
    }

    /// Resolve a child name segment using one fallible allocation.
    pub fn try_resolve_child(&self, segment: NameSeg) -> Result<AmlName, AmlError> {
        if !self.is_absolute() {
            return Err(AmlError::NameNotAbsolute(self.try_clone()?));
        }
        let component_count = self.0.len().checked_add(1).ok_or_else(|| AmlError::BudgetExceeded)?;
        if component_count > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        let mut components = allocator::vec_new();
        components.try_reserve_exact(component_count).map_err(|_| AmlError::BudgetExceeded)?;
        components.extend_from_slice(&self.0);
        components.push(NameComponent::Segment(segment));
        Ok(AmlName(components))
    }

    /// Return a conservative estimate for the component vector allocation
    /// performed by `try_clone`.
    pub fn clone_heap_upper_bound(&self) -> Option<u64> {
        if self.0.len() > MAX_AML_NAME_COMPONENTS {
            return None;
        }
        let bytes = core::mem::size_of::<Vec<NameComponent>>()
            .checked_add(self.0.len().checked_mul(core::mem::size_of::<NameComponent>())?)?
            .checked_add(24)?;
        u64::try_from(bytes).ok()
    }

    /// Conservative heap bound for the component vector currently owned by
    /// this path, including spare capacity and allocator bookkeeping.
    pub fn heap_allocation_upper_bound(&self) -> Option<u64> {
        if self.0.len() > MAX_AML_NAME_COMPONENTS || self.0.capacity() > MAX_AML_NAME_COMPONENTS {
            return None;
        }
        let bytes = core::mem::size_of::<Vec<NameComponent>>()
            .checked_add(self.0.capacity().checked_mul(core::mem::size_of::<NameComponent>())?)?
            .checked_add(24)?;
        u64::try_from(bytes).ok()
    }

    pub fn root() -> AmlName {
        AmlName(crate::aml_vec![NameComponent::Root])
    }

    pub fn from_name_seg(seg: NameSeg) -> AmlName {
        AmlName(crate::aml_vec![NameComponent::Segment(seg)])
    }

    pub fn from_components(components: Vec<NameComponent>) -> AmlName {
        AmlName(components)
    }

    pub fn as_string(&self) -> String {
        self.try_to_string_bounded(MAX_AML_NAME_COMPONENTS * 5)
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    /// Append the rendered path into an output with capacity reserved from
    /// `encoded_len` by the caller.
    pub(crate) fn append_rendered_to(&self, output: &mut String) {
        for (index, component) in self.0.iter().enumerate() {
            match component {
                NameComponent::Root => output.push('\\'),
                NameComponent::Prefix => output.push('^'),
                NameComponent::Segment(segment) => {
                    output.push_str(segment.as_str());
                    if index + 1 < self.0.len() {
                        output.push('.');
                    }
                }
            }
        }
    }

    /// Return the rendered path length without allocating a `String`.
    pub fn encoded_len(&self) -> Option<usize> {
        if self.0.len() > MAX_AML_NAME_COMPONENTS {
            return None;
        }
        let mut length = 0_usize;
        let mut trailing_segment = false;
        for component in &self.0 {
            match component {
                NameComponent::Root | NameComponent::Prefix => {
                    length = length.checked_add(1)?;
                    trailing_segment = false;
                }
                NameComponent::Segment(_) => {
                    length = length.checked_add(5)?;
                    trailing_segment = true;
                }
            }
        }
        if trailing_segment {
            length = length.checked_sub(1)?;
        }
        Some(length)
    }

    /// Render an AML namespace path with an explicit output bound and a
    /// fallible exact allocation. `Ok(None)` means the encoded path is valid
    /// but exceeds the caller's wire/operation limit.
    pub fn try_to_string_bounded(&self, max_bytes: usize) -> Result<Option<String>, AmlError> {
        let length = self.encoded_len().ok_or_else(|| AmlError::BudgetExceeded)?;
        if length > max_bytes {
            return Ok(None);
        }
        let mut output = String::new();
        output.try_reserve_exact(length).map_err(|_| AmlError::BudgetExceeded)?;
        self.append_rendered_to(&mut output);
        Ok(Some(output))
    }

    /// An AML path is normal if it does not contain any prefix elements ("^" characters, when
    /// expressed as a string).
    pub fn is_normal(&self) -> bool {
        !self.0.contains(&NameComponent::Prefix)
    }

    pub fn is_absolute(&self) -> bool {
        self.0.first() == Some(&NameComponent::Root)
    }

    /// Special rules apply when searching for certain paths (specifically, those that are made up
    /// of a single name segment). Returns `true` if those rules apply.
    pub fn search_rules_apply(&self) -> bool {
        if self.0.len() != 1 {
            return false;
        }

        matches!(self.0[0], NameComponent::Segment(_))
    }

    pub fn is_descendant_of(&self, ancestor: &AmlName) -> bool {
        self.0.len() > ancestor.0.len() && self.0.starts_with(&ancestor.0)
    }

    pub fn is_rooted_single_segment(&self, segment: [u8; 4]) -> bool {
        matches!(
            self.0.as_slice(),
            [NameComponent::Root, NameComponent::Segment(NameSeg(bytes))] if *bytes == segment
        )
    }

    pub fn ends_with_segment(&self, segment: [u8; 4]) -> bool {
        matches!(self.0.last(), Some(NameComponent::Segment(NameSeg(bytes))) if *bytes == segment)
    }

    /// Normalize an AML path, resolving prefix chars. Returns `AmlError::InvalidNormalizedName` if the path
    /// normalizes to an invalid path (e.g. `\^_FOO`)
    pub fn normalize(self) -> Result<AmlName, AmlError> {
        if self.0.len() > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        /*
         * If the path is already normal, just return it as-is. This avoids an unneccessary heap allocation and
         * free.
         */
        if self.is_normal() {
            return Ok(self);
        }

        let mut normalized = allocator::vec_new();
        normalized.try_reserve_exact(self.0.len()).map_err(|_| AmlError::BudgetExceeded)?;
        for &component in &self.0 {
            match component {
                NameComponent::Segment(_) | NameComponent::Root => normalized.push(component),
                NameComponent::Prefix => {
                    if matches!(normalized.last(), Some(NameComponent::Segment(_))) {
                        normalized.pop();
                    } else {
                        return Err(AmlError::InvalidNormalizedName(self.try_clone()?));
                    }
                }
            }
        }
        Ok(AmlName(normalized))
    }

    /// Get the parent of this `AmlName`. For example, the parent of `\_SB.PCI0._PRT` is `\_SB.PCI0`. The root
    /// path has no parent, and so returns `None`.
    pub fn parent(&self) -> Result<AmlName, AmlError> {
        // Firstly, normalize the path so we don't have to deal with prefix chars
        let mut normalized_self = self.try_clone()?.normalize()?;

        match normalized_self.0.last() {
            None | Some(NameComponent::Root) => Err(AmlError::RootHasNoParent),
            Some(NameComponent::Segment(_)) => {
                normalized_self.0.pop();
                Ok(normalized_self)
            }
            Some(NameComponent::Prefix) => unreachable!(), // Prefix chars are removed by normalization
        }
    }

    /// Returns `AmlError::NameNotAbsolute` if this path is not absolute. Relative paths can reach
    /// entry points that require absolute ones from firmware input, so this is a normal error
    /// rather than a panic.
    pub fn require_absolute(&self) -> Result<(), AmlError> {
        if self.is_absolute() { Ok(()) } else { Err(AmlError::NameNotAbsolute(self.try_clone()?)) }
    }

    /// Normalize an AML path that is required to be absolute. Prefer this over `normalize` where
    /// an absolute path is required.
    pub fn normalize_absolute(self) -> Result<AmlName, AmlError> {
        self.require_absolute()?;
        self.normalize()
    }

    /// Resolve this path while accounting for the bounded component-copy and
    /// normalization scratch used by the AML executor.
    pub(super) fn resolve_metered(
        &self,
        scope: &AmlName,
        meter: Arc<dyn ExecutionMeter>,
    ) -> Result<super::MeteredNameString, AmlError> {
        if !scope.is_absolute() {
            return Err(AmlError::NameNotAbsoluteWithoutPath);
        }
        let component_count = if self.is_absolute() {
            self.0.len()
        } else {
            scope.0.len().checked_add(self.0.len()).ok_or(AmlError::BudgetExceeded)?
        };
        if component_count > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        for _ in 0..component_count {
            meter.loop_iteration()?;
        }

        let vector_bytes = core::mem::size_of::<Vec<NameComponent>>()
            .checked_add(
                component_count
                    .checked_mul(core::mem::size_of::<NameComponent>())
                    .ok_or(AmlError::BudgetExceeded)?,
            )
            .and_then(|bytes| bytes.checked_add(24))
            .ok_or(AmlError::BudgetExceeded)?;
        // A prefix path may keep the joined input alive while normalization
        // builds its output. Invalid-normalization diagnostics are converted
        // to pathless errors below, so their temporary path copy stays within
        // this reservation and is released before the error escapes.
        let allocation_count = if self.is_normal() { 1 } else { 3 };
        let heap_bytes = vector_bytes
            .checked_mul(allocation_count)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(AmlError::BudgetExceeded)?;
        let mut heap_reservation = super::NameStringHeapReservation::new(Some(meter));
        heap_reservation.reserve_bytes(heap_bytes)?;
        let resolved = match self.resolve(scope) {
            Ok(resolved) => resolved,
            Err(AmlError::InvalidNormalizedName(_)) => {
                return Err(AmlError::InvalidNormalizedNameWithoutPath);
            }
            Err(AmlError::NameNotAbsolute(_)) => return Err(AmlError::NameNotAbsoluteWithoutPath),
            Err(error) => return Err(error),
        };
        Ok(super::MeteredNameString::with_heap_reservation(resolved, heap_reservation))
    }

    pub(super) fn clone_metered(
        &self,
        meter: Arc<dyn ExecutionMeter>,
    ) -> Result<super::MeteredNameString, AmlError> {
        if self.0.len() > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        for _ in 0..self.0.len() {
            meter.loop_iteration()?;
        }
        let bytes = self.clone_heap_upper_bound().ok_or(AmlError::BudgetExceeded)?;
        let mut heap_reservation = super::NameStringHeapReservation::new(Some(meter));
        heap_reservation.reserve_bytes(bytes)?;
        let name = self.try_clone()?;
        Ok(super::MeteredNameString::with_heap_reservation(name, heap_reservation))
    }

    pub(super) fn parent_metered(
        &self,
        meter: Arc<dyn ExecutionMeter>,
    ) -> Result<super::MeteredNameString, AmlError> {
        if self.0.len() > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        for _ in 0..self.0.len() {
            meter.loop_iteration()?;
        }
        let bytes = self
            .clone_heap_upper_bound()
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(AmlError::BudgetExceeded)?;
        let mut heap_reservation = super::NameStringHeapReservation::new(Some(meter));
        heap_reservation.reserve_bytes(bytes)?;
        let parent = self.parent()?;
        Ok(super::MeteredNameString::with_heap_reservation(parent, heap_reservation))
    }

    /// Resolve this path against a given scope, making it absolute. If the path is absolute, it is
    /// returned directly. The path is also normalized. Errors if `scope` is not absolute.
    pub fn resolve(&self, scope: &AmlName) -> Result<AmlName, AmlError> {
        scope.require_absolute()?;

        if self.is_absolute() {
            return self.try_clone()?.normalize();
        }

        let component_count = scope.0.len().checked_add(self.0.len()).ok_or_else(|| AmlError::BudgetExceeded)?;
        if component_count > MAX_AML_NAME_COMPONENTS {
            return Err(AmlError::BudgetExceeded);
        }
        let mut components = allocator::vec_new();
        components.try_reserve_exact(component_count).map_err(|_| AmlError::BudgetExceeded)?;
        components.extend_from_slice(&scope.0);
        components.extend_from_slice(&self.0);
        AmlName(components).normalize()
    }
}

impl FromStr for AmlName {
    type Err = AmlError;

    fn from_str(mut string: &str) -> Result<Self, Self::Err> {
        if string.is_empty() {
            return Err(AmlError::EmptyNamesAreInvalid);
        }

        let absolute = string.starts_with('\\');
        if absolute {
            string = &string[1..];
        }

        let mut component_count: usize = if absolute { 1 } else { 0 };
        if !string.is_empty() {
            for part in string.split('.') {
                let mut prefix_count = 0_usize;
                while part.as_bytes().get(prefix_count) == Some(&b'^') {
                    prefix_count = prefix_count
                        .checked_add(1)
                        .filter(|count| *count <= MAX_AML_NAME_COMPONENTS)
                        .ok_or_else(|| AmlError::BudgetExceeded)?;
                }
                component_count = component_count
                    .checked_add(prefix_count)
                    .and_then(|count| count.checked_add(1))
                    .filter(|count| *count <= MAX_AML_NAME_COMPONENTS)
                    .ok_or_else(|| AmlError::BudgetExceeded)?;
            }
        }

        let mut components = allocator::vec_new();
        components.try_reserve_exact(component_count).map_err(|_| AmlError::BudgetExceeded)?;

        if absolute {
            components.push(NameComponent::Root);
        }

        if !string.is_empty() {
            // Divide the rest of it into segments, and parse those
            for mut part in string.split('.') {
                // Handle prefix chars
                while part.starts_with('^') {
                    components.push(NameComponent::Prefix);
                    part = &part[1..];
                }

                components.push(NameComponent::Segment(NameSeg::from_str(part)?));
            }
        }

        Ok(Self(components))
    }
}

impl fmt::Display for AmlName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (index, component) in self.0.iter().enumerate() {
            match component {
                NameComponent::Root => f.write_str("\\")?,
                NameComponent::Prefix => f.write_str("^")?,
                NameComponent::Segment(segment) => {
                    f.write_str(segment.as_str())?;
                    if index + 1 < self.0.len() {
                        f.write_str(".")?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NameSeg(pub(crate) [u8; 4]);

impl NameSeg {
    pub fn from_bytes(bytes: [u8; 4]) -> Result<NameSeg, AmlError> {
        if !is_lead_name_char(bytes[0]) {
            return Err(AmlError::InvalidNameSeg(bytes));
        }
        if !is_name_char(bytes[1]) {
            return Err(AmlError::InvalidNameSeg(bytes));
        }
        if !is_name_char(bytes[2]) {
            return Err(AmlError::InvalidNameSeg(bytes));
        }
        if !is_name_char(bytes[3]) {
            return Err(AmlError::InvalidNameSeg(bytes));
        }
        Ok(NameSeg(bytes))
    }

    pub fn as_str(&self) -> &str {
        // We should only construct valid ASCII name segments
        unsafe { str::from_utf8_unchecked(&self.0) }
    }
}

impl FromStr for NameSeg {
    type Err = AmlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Each NameSeg can only have four chars, and must have at least one
        if s.is_empty() || s.len() > 4 {
            return Err(AmlError::InvalidNameSeg([0xff, 0xff, 0xff, 0xff]));
        }

        // We pre-fill the array with '_', so it will already be correct if the length is < 4
        let mut seg = [b'_'; 4];
        let bytes = s.as_bytes();
        let invalid_bytes = || {
            let mut value = [0xff; 4];
            value[..bytes.len()].copy_from_slice(bytes);
            value
        };

        // Manually do the first one, because we have to check it's a LeadNameChar
        if !is_lead_name_char(bytes[0]) {
            return Err(AmlError::InvalidNameSeg(invalid_bytes()));
        }
        seg[0] = bytes[0];

        // Copy the rest of the chars, checking that they're NameChars
        for i in 1..bytes.len() {
            if !is_name_char(bytes[i]) {
                return Err(AmlError::InvalidNameSeg(invalid_bytes()));
            }
            seg[i] = bytes[i];
        }

        Ok(NameSeg(seg))
    }
}

pub fn is_lead_name_char(c: u8) -> bool {
    c.is_ascii_uppercase() || c == b'_'
}

pub fn is_name_char(c: u8) -> bool {
    is_lead_name_char(c) || c.is_ascii_digit()
}

impl fmt::Debug for NameSeg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    #[derive(Debug)]
    struct DropCountReservation(Arc<AtomicUsize>);

    impl ObjectHeapReservation for DropCountReservation {
        fn commit_to_persistent(&mut self) -> Result<(), AmlError> {
            Ok(())
        }
    }

    impl Drop for DropCountReservation {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn namespace_heap_reservations_follow_value_and_level_lifetimes() {
        let dropped = allocator::arc_new(AtomicUsize::new(0));
        let mut namespace = Namespace::new(Handle(1));
        let path = AmlName::from_str("\\VALU").unwrap();

        assert!(
            namespace
                .insert_if_mutable_with_reservation(
                    path.clone(),
                    Object::Integer(1).wrap(),
                    Some(allocator::box_new(DropCountReservation(Arc::clone(&dropped)))),
                )
                .unwrap()
        );
        assert!(
            namespace
                .insert_if_mutable_with_reservation(
                    path.clone(),
                    Object::Integer(2).wrap(),
                    Some(allocator::box_new(DropCountReservation(Arc::clone(&dropped)))),
                )
                .unwrap()
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 1);

        let immutable_path = AmlName::from_str("\\IMMU").unwrap();
        namespace.insert_immutable(immutable_path.clone(), Object::Integer(3).wrap()).unwrap();
        assert!(
            !namespace
                .insert_if_mutable_with_reservation(
                    immutable_path,
                    Object::Integer(4).wrap(),
                    Some(allocator::box_new(DropCountReservation(Arc::clone(&dropped)))),
                )
                .unwrap()
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 2);

        let level_path = AmlName::from_str("\\LEVL").unwrap();
        namespace
            .add_level_with_reservation(
                level_path.clone(),
                NamespaceLevelKind::Scope,
                Some(allocator::box_new(DropCountReservation(Arc::clone(&dropped)))),
            )
            .unwrap();
        namespace.remove_level(level_path).unwrap();
        assert_eq!(dropped.load(Ordering::Relaxed), 3);
    }

    struct CountingMeter {
        iterations: AtomicUsize,
        deadline_checks: AtomicUsize,
        heap_bytes: AtomicU64,
        reserved_heap_bytes: AtomicU64,
        released_heap_bytes: AtomicU64,
        reject_heap: bool,
        reject_deadline: bool,
    }

    struct GenerationMeter {
        generation: AtomicU64,
        released_generation: AtomicU64,
    }

    impl ExecutionMeter for GenerationMeter {
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

        fn reserve_working_heap(&self, _bytes: u64) -> Result<(), AmlError> {
            Ok(())
        }

        fn reserve_working_heap_scoped(&self, _bytes: u64) -> Result<u64, AmlError> {
            Ok(self.generation.load(Ordering::Acquire))
        }

        fn release_working_heap_scoped(&self, _bytes: u64, generation: u64) {
            self.released_generation.store(generation, Ordering::Release);
        }
    }

    impl ExecutionMeter for CountingMeter {
        fn charge_opcode(&self) -> Result<(), AmlError> {
            Ok(())
        }

        fn enter_method(&self) -> Result<(), AmlError> {
            Ok(())
        }

        fn leave_method(&self) {}

        fn loop_iteration(&self) -> Result<(), AmlError> {
            self.iterations.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn check_deadline(&self) -> Result<(), AmlError> {
            self.deadline_checks.fetch_add(1, Ordering::Relaxed);
            if self.reject_deadline {
                Err(AmlError::HostError("test deadline exceeded".to_string()))
            } else {
                Ok(())
            }
        }

        fn reserve_working_heap(&self, bytes: u64) -> Result<(), AmlError> {
            if self.reject_heap {
                return Err(AmlError::HostError("test heap budget exhausted".to_string()));
            }
            self.heap_bytes.fetch_add(bytes, Ordering::Relaxed);
            self.reserved_heap_bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(())
        }

        fn release_working_heap(&self, bytes: u64) {
            self.heap_bytes.fetch_sub(bytes, Ordering::Relaxed);
            self.released_heap_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    #[test]
    fn inserting_an_object_at_the_namespace_root_returns_an_error() {
        let mut namespace = Namespace { root: NamespaceLevel::new(NamespaceLevelKind::Scope) };
        assert!(matches!(namespace.get_level_for_path(&AmlName::root()), Err(AmlError::InvalidName(Some(_)))));
        let result = namespace.insert(AmlName::root(), Object::Integer(1).wrap());

        assert!(matches!(result, Err(AmlError::InvalidName(Some(_)))));
    }

    #[test]
    fn invalid_short_name_segment_returns_error_without_indexing_past_input() {
        assert!(matches!(NameSeg::from_str("q"), Err(AmlError::InvalidNameSeg(_))));
        assert!(matches!(NameSeg::from_str("A-"), Err(AmlError::InvalidNameSeg(_))));
    }

    #[test]
    fn immutable_namespace_values_reject_replacement_and_mark_named_targets() {
        let mut namespace = Namespace::new(Handle(1));
        let path = AmlName::from_str("\\_REV").unwrap();
        namespace.insert_immutable(path.clone(), Object::Integer(7).wrap()).unwrap();
        namespace.insert(path.clone(), Object::Integer(9).wrap()).unwrap();

        let (_, value, immutable) = namespace.search_with_flags(&path, &AmlName::root()).unwrap();
        assert!(immutable.is_immutable());
        assert!(matches!(*value, Object::Integer(7)));
    }

    #[test]
    fn insert_if_mutable_reports_when_an_immutable_slot_keeps_its_value() {
        let mut namespace = Namespace::new(Handle(1));
        let path = AmlName::from_str("\\_REV").unwrap();
        namespace.insert_immutable(path.try_clone().unwrap(), Object::Integer(2).wrap()).unwrap();

        assert!(!namespace.insert_if_mutable(path, Object::Integer(9).wrap()).unwrap());
        assert!(matches!(
            *namespace.get_absolute(&AmlName::from_str("\\_REV").unwrap()).unwrap(),
            Object::Integer(2)
        ));
    }

    #[test]
    fn namespace_read_apis_accept_shared_borrows() {
        let namespace = Namespace::new(Handle(1));
        let path = AmlName::from_str("\\_REV").unwrap();
        assert!(matches!(*namespace.get(path).unwrap(), Object::Integer(2)));

        let path = AmlName::try_from_absolute_segments(&[NameSeg::from_str("_REV").unwrap()]).unwrap();
        assert!(matches!(*namespace.get_absolute(&path).unwrap(), Object::Integer(2)));

        let absent = AmlName::try_from_absolute_segments(&[
            NameSeg::from_str("_GPE").unwrap(),
            NameSeg::from_str("_L2A").unwrap(),
        ])
        .unwrap();
        assert!(namespace.get_absolute_if_present(&absent).unwrap().is_none());

        let mut visited_root = false;
        namespace
            .traverse(|path, level| {
                visited_root |= path == &AmlName::root() && level.kind == NamespaceLevelKind::Scope;
                Ok(true)
            })
            .unwrap();
        assert!(visited_root);
    }

    #[test]
    fn metered_namespace_search_charges_and_releases_resolved_path_scratch() {
        let namespace = Namespace::new(Handle(1));
        let path = AmlName::from_str("_REV").unwrap();
        let scope = AmlName::from_str("\\_SB_").unwrap();
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        });

        let (resolved, object, _) = namespace
            .search_with_flags_metered(&path, &scope, meter.clone())
            .expect("relative name should resolve in a parent namespace");
        assert_eq!(resolved.to_string(), "\\_REV");
        assert!(matches!(*object, Object::Integer(2)));
        assert!(meter.iterations.load(Ordering::Relaxed) > 0);
        let reserved_bytes = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        assert!(reserved_bytes > 0);
        let live_bytes = meter.heap_bytes.load(Ordering::Relaxed);
        assert!(live_bytes > 0);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed) + live_bytes, reserved_bytes);

        drop(resolved);
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed), reserved_bytes);
    }

    #[test]
    fn metered_optional_namespace_search_miss_returns_without_diagnostic_clone() {
        let namespace = Namespace::new(Handle(1));
        let path = AmlName::from_str("MISS").unwrap();
        let scope = AmlName::root();
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        });

        let result = namespace.search_with_flags_metered_if_present(&path, &scope, meter.clone()).unwrap();

        assert!(result.is_none());
        assert!(meter.reserved_heap_bytes.load(Ordering::Relaxed) > 0);
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            meter.reserved_heap_bytes.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn metered_level_search_charges_parent_walk_and_releases_path_scratch() {
        let mut namespace = Namespace::new(Handle(1));
        namespace.add_level(AmlName::from_str("\\_SB_.PCI0").unwrap(), NamespaceLevelKind::Scope).unwrap();
        namespace.add_level(AmlName::from_str("\\_SB_.PCI0.LNKA").unwrap(), NamespaceLevelKind::Device).unwrap();
        let name = AmlName::from_str("LNKA").unwrap();
        let scope = AmlName::from_str("\\_SB_.PCI0.DEV0").unwrap();
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        });

        let resolved = namespace.search_for_level_metered(&name, &scope, meter.clone()).unwrap().unwrap();

        assert_eq!(resolved.as_name().to_string(), "\\_SB_.PCI0.LNKA");
        assert!(meter.iterations.load(Ordering::Relaxed) > name.0.len());
        assert!(meter.heap_bytes.load(Ordering::Relaxed) > 0);
        drop(resolved);
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            meter.reserved_heap_bytes.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn metered_level_search_propagates_deadline_failure() {
        let namespace = Namespace::new(Handle(1));
        let name = AmlName::from_str("MISS").unwrap();
        let scope = AmlName::root();
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: true,
        });

        assert!(matches!(
            namespace.search_for_level_metered(&name, &scope, meter.clone()),
            Err(AmlError::HostError(_))
        ));
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn metered_required_namespace_search_miss_does_not_return_an_unmetered_path_copy() {
        let namespace = Namespace::new(Handle(1));
        let path = AmlName::from_str("MISS").unwrap();
        let scope = AmlName::root();
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        });

        let error = namespace.search_with_flags_metered(&path, &scope, meter.clone()).unwrap_err();

        assert!(matches!(error, AmlError::ObjectDoesNotExistWithoutPath));
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            meter.reserved_heap_bytes.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn metered_namespace_search_rejects_malformed_absolute_path_without_copying_it() {
        let namespace = Namespace::new(Handle(1));
        let path = AmlName::from_components(crate::aml_vec![
            NameComponent::Root,
            NameComponent::Root,
            NameComponent::Segment(NameSeg::from_str("MISS").unwrap()),
        ]);
        let scope = AmlName::root();
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        });

        assert_eq!(
            namespace.search_with_flags_metered(&path, &scope, meter.clone()).unwrap_err(),
            AmlError::InvalidName(None)
        );
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            meter.reserved_heap_bytes.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn namespace_clone_heap_upper_bound_tracks_map_growth() {
        let mut namespace = Namespace::new(Handle(1));
        let empty_bound = namespace.clone_heap_upper_bound().unwrap();

        for index in 0_u64..16 {
            let name = AmlName::from_str(&format!("\\N{index:03X}")).unwrap();
            namespace.insert(name, Object::Integer(index).wrap()).unwrap();
        }

        assert!(namespace.clone_heap_upper_bound().unwrap() > empty_bound);
    }

    #[test]
    fn device_candidate_heap_upper_bound_covers_path_storage() {
        let mut namespace = Namespace::new(Handle(1));
        namespace.add_level(AmlName::from_str("\\_SB.DEV0").unwrap(), NamespaceLevelKind::Device).unwrap();
        let (one_device, one_device_bytes) = namespace.device_candidate_heap_upper_bound().unwrap();
        assert_eq!(one_device, 1);

        namespace.add_level(AmlName::from_str("\\_SB.DEV0.CHLD").unwrap(), NamespaceLevelKind::Device).unwrap();
        let (two_devices, two_device_bytes) = namespace.device_candidate_heap_upper_bound().unwrap();
        assert_eq!(two_devices, 2);
        assert!(two_device_bytes > one_device_bytes);
    }

    #[test]
    fn metered_device_candidate_heap_upper_bound_charges_namespace_levels() {
        let mut namespace = Namespace::new(Handle(1));
        namespace.add_level(AmlName::from_str("\\_SB.DEV0").unwrap(), NamespaceLevelKind::Device).unwrap();
        let meter = CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        };

        let bounded = namespace.device_candidate_heap_upper_bound_metered(&meter).unwrap();

        assert_eq!(bounded, namespace.device_candidate_heap_upper_bound().unwrap());
        assert!(meter.iterations.load(Ordering::Relaxed) > 0);
        assert!(meter.deadline_checks.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn device_like_candidate_heap_upper_bound_includes_all_initializer_levels() {
        let mut namespace = Namespace::new(Handle(1));
        namespace.add_level(AmlName::from_str("\\_SB.DEV0").unwrap(), NamespaceLevelKind::Device).unwrap();
        namespace.add_level(AmlName::from_str("\\_SB.PROC").unwrap(), NamespaceLevelKind::Processor).unwrap();
        namespace.add_level(AmlName::from_str("\\_SB.THZ0").unwrap(), NamespaceLevelKind::ThermalZone).unwrap();
        namespace.add_level(AmlName::from_str("\\_SB.PWR0").unwrap(), NamespaceLevelKind::PowerResource).unwrap();

        assert_eq!(namespace.device_candidate_heap_upper_bound().unwrap().0, 1);
        assert_eq!(namespace.device_like_candidate_heap_upper_bound().unwrap().0, 4);
    }

    #[test]
    fn aml_path_descendant_check_uses_component_boundaries() {
        let parent = AmlName::from_str("\\_SB.DEV0").unwrap();

        assert!(AmlName::from_str("\\_SB.DEV0.CHLD").unwrap().is_descendant_of(&parent));
        assert!(!AmlName::from_str("\\_SB.DEV1.CHLD").unwrap().is_descendant_of(&parent));
        assert!(!parent.is_descendant_of(&parent));
        assert!(AmlName::from_str("\\_PIC").unwrap().is_rooted_single_segment(*b"_PIC"));
        assert!(!parent.is_rooted_single_segment(*b"_PIC"));
        assert!(parent.ends_with_segment(*b"DEV0"));
    }

    #[test]
    fn device_path_string_heap_upper_bound_covers_rendered_path_storage() {
        let mut namespace = Namespace::new(Handle(1));
        namespace.add_level(AmlName::from_str("\\_SB.DEV0").unwrap(), NamespaceLevelKind::Device).unwrap();
        let (one_device, one_device_bytes) = namespace.device_path_string_heap_upper_bound().unwrap();
        assert_eq!(one_device, 1);

        namespace.add_level(AmlName::from_str("\\_SB.DEV0.CHLD").unwrap(), NamespaceLevelKind::Device).unwrap();
        let (two_devices, two_device_bytes) = namespace.device_path_string_heap_upper_bound().unwrap();
        assert_eq!(two_devices, 2);
        assert!(two_device_bytes > one_device_bytes);
    }

    #[test]
    fn pci_config_path_heap_upper_bound_covers_device_and_region_parent_strings() {
        let mut namespace = Namespace::new(Handle(1));
        let device_path = AmlName::from_str("\\_SB.DEV0").unwrap();
        namespace.add_level(device_path.clone(), NamespaceLevelKind::Device).unwrap();
        namespace
            .insert(
                AmlName::from_str("\\_SB.DEV0.REG0").unwrap(),
                Object::OpRegion(crate::aml::op_region::OpRegion {
                    space: crate::aml::op_region::RegionSpace::PciConfig,
                    base: 0,
                    length: 4,
                    parent_device_path: device_path,
                })
                .wrap(),
            )
            .unwrap();

        let (device_paths, region_parents, one_device_bytes) =
            namespace.pci_config_path_heap_upper_bound().unwrap();
        assert_eq!(device_paths, 1);
        assert_eq!(region_parents, 1);

        let child_path = AmlName::from_str("\\_SB.DEV0.CHLD").unwrap();
        namespace.add_level(child_path.clone(), NamespaceLevelKind::Device).unwrap();
        namespace
            .insert(
                AmlName::from_str("\\_SB.DEV0.CHLD.REG0").unwrap(),
                Object::OpRegion(crate::aml::op_region::OpRegion {
                    space: crate::aml::op_region::RegionSpace::PciConfig,
                    base: 0,
                    length: 4,
                    parent_device_path: child_path,
                })
                .wrap(),
            )
            .unwrap();
        let (device_paths, region_parents, two_device_bytes) =
            namespace.pci_config_path_heap_upper_bound().unwrap();
        assert_eq!(device_paths, 2);
        assert_eq!(region_parents, 2);
        assert!(two_device_bytes > one_device_bytes);

        let meter = CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        };
        assert_eq!(
            namespace.pci_config_path_heap_upper_bound_metered(&meter).unwrap(),
            (device_paths, region_parents, two_device_bytes)
        );
        assert!(meter.iterations.load(Ordering::Relaxed) > 0);
        assert!(meter.deadline_checks.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn namespace_traversal_rejects_excessive_depth_before_recursing() {
        let mut namespace = Namespace { root: NamespaceLevel::new(NamespaceLevelKind::Scope) };
        let child_name = NameSeg::from_str("DEEP").unwrap();
        let mut level = &mut namespace.root;
        for _ in 0..=MAX_BOUNDED_NAMESPACE_DEPTH {
            level.children.insert(child_name, NamespaceLevel::new(NamespaceLevelKind::Scope));
            level = level.children.get_mut(&child_name).unwrap();
        }

        assert!(namespace.traverse(|_, _| Ok(true)).is_err());
    }

    #[test]
    fn metered_namespace_traversal_reserves_path_scratch_before_visiting() {
        let namespace = Namespace::new(Handle(1));
        let meter = CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        };
        let mut visits = 0;

        namespace
            .traverse_metered(&meter, |_, _| {
                visits += 1;
                Ok(true)
            })
            .unwrap();

        assert_eq!(visits, 6);
        assert_eq!(meter.iterations.load(Ordering::Relaxed), 6);
        assert_eq!(meter.deadline_checks.load(Ordering::Relaxed), 1);
        assert_eq!(
            meter.reserved_heap_bytes.load(Ordering::Relaxed),
            Namespace::traversal_path_heap_upper_bound().unwrap()
        );
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            Namespace::traversal_path_heap_upper_bound().unwrap()
        );

        let rejecting_meter = CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: true,
            reject_deadline: false,
        };
        let mut rejected_visits = 0;
        assert!(
            namespace
                .traverse_metered(&rejecting_meter, |_, _| {
                    rejected_visits += 1;
                    Ok(true)
                })
                .is_err()
        );
        assert_eq!(rejected_visits, 0);
        assert_eq!(rejecting_meter.iterations.load(Ordering::Relaxed), 0);
        assert_eq!(rejecting_meter.released_heap_bytes.load(Ordering::Relaxed), 0);

        let deadline_meter = CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: true,
        };
        assert!(namespace.traverse_metered(&deadline_meter, |_, _| Ok(true)).is_err());
        assert_eq!(deadline_meter.deadline_checks.load(Ordering::Relaxed), 1);
        assert_eq!(deadline_meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            deadline_meter.released_heap_bytes.load(Ordering::Relaxed),
            Namespace::traversal_path_heap_upper_bound().unwrap()
        );

        let callback_error_meter = CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        };
        assert!(
            namespace
                .traverse_metered(&callback_error_meter, |_, _| {
                    Err(AmlError::HostError("test traversal error".to_string()))
                })
                .is_err()
        );
        assert_eq!(callback_error_meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            callback_error_meter.released_heap_bytes.load(Ordering::Relaxed),
            Namespace::traversal_path_heap_upper_bound().unwrap()
        );
    }

    #[test]
    fn metered_namespace_traversal_releases_scratch_to_its_originating_work_unit() {
        let namespace = Namespace::new(Handle(1));
        let meter = GenerationMeter { generation: AtomicU64::new(7), released_generation: AtomicU64::new(0) };

        namespace
            .traverse_metered(&meter, |_, _| {
                meter.generation.store(8, Ordering::Release);
                Ok(false)
            })
            .unwrap();

        assert_eq!(meter.released_generation.load(Ordering::Acquire), 7);
    }

    #[test]
    fn encoded_name_length_matches_rendered_path_length() {
        for raw in ["\\", "\\_SB_.PCI0", "^DEV0.CHLD", "\\_SB_.^PCI0.CHLD"] {
            let name = AmlName::from_str(raw).unwrap();
            assert_eq!(name.encoded_len(), Some(name.to_string().len()), "{raw}");
        }
    }

    #[test]
    fn bounded_try_render_matches_path_and_rejects_before_copy_when_too_long() {
        for raw in ["\\", "\\_SB_.PCI0", "^DEV0.CHLD", "\\_SB_.^PCI0.CHLD"] {
            let name = AmlName::from_str(raw).unwrap();
            assert_eq!(name.try_to_string_bounded(raw.len()).unwrap().as_deref(), Some(raw));
            if !raw.is_empty() {
                assert_eq!(name.try_to_string_bounded(raw.len() - 1).unwrap(), None);
            }
        }
    }

    #[test]
    fn clone_heap_upper_bound_covers_name_component_storage() {
        let name = AmlName::from_str("\\_SB_.PCI0.DEV0").unwrap();
        let upper_bound = name.clone_heap_upper_bound().unwrap();
        let expected =
            core::mem::size_of::<Vec<NameComponent>>() + name.0.len() * core::mem::size_of::<NameComponent>() + 24;
        assert_eq!(upper_bound, expected as u64);
    }

    #[test]
    fn aml_name_string_rejects_paths_over_component_limit() {
        let mut path = String::new();
        path.extend(core::iter::repeat_n('^', MAX_BOUNDED_NAMESPACE_DEPTH + 1));
        path.push_str("ABCD");

        assert_eq!(AmlName::from_str(&path), Err(AmlError::BudgetExceeded));
    }

    #[test]
    fn aml_name_resolution_rejects_combined_component_overflow() {
        let scope = AmlName::from_str("\\_SB_").unwrap();
        let name = AmlName::from_components(crate::aml_vec![
            NameComponent::Segment(NameSeg::from_str("AAAA").unwrap());
            MAX_BOUNDED_NAMESPACE_DEPTH
        ]);

        assert_eq!(name.resolve(&scope), Err(AmlError::BudgetExceeded));
    }

    #[test]
    fn metered_aml_name_resolution_reserves_then_releases_scratch() {
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        });
        let name = AmlName::from_str("^DEV0").unwrap();
        let scope = AmlName::from_str("\\_SB_.PCI0").unwrap();

        let resolved =
            name.resolve_metered(&scope, meter.clone()).expect("bounded path scratch should be admitted");
        assert_eq!(resolved.to_string(), "\\_SB_.DEV0");
        let reserved_bytes = meter.reserved_heap_bytes.load(Ordering::Relaxed);
        assert!(reserved_bytes > 0);
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), reserved_bytes);

        drop(resolved);

        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed), reserved_bytes);
    }

    #[test]
    fn metered_aml_name_resolution_rejects_before_path_allocation() {
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: true,
            reject_deadline: false,
        });
        let name = AmlName::from_str("^DEV0").unwrap();
        let scope = AmlName::from_str("\\_SB_.PCI0").unwrap();

        assert!(name.resolve_metered(&scope, meter.clone()).is_err());
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(meter.released_heap_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn metered_invalid_normalization_returns_pathless_error_and_releases_scratch() {
        let meter = allocator::arc_new(CountingMeter {
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            heap_bytes: AtomicU64::new(0),
            reserved_heap_bytes: AtomicU64::new(0),
            released_heap_bytes: AtomicU64::new(0),
            reject_heap: false,
            reject_deadline: false,
        });
        let name = AmlName::from_components(crate::aml_vec![
            NameComponent::Root,
            NameComponent::Prefix,
            NameComponent::Segment(NameSeg::from_str("DEV0").unwrap()),
        ]);
        let scope = AmlName::root();

        assert_eq!(
            name.resolve_metered(&scope, meter.clone()).unwrap_err(),
            AmlError::InvalidNormalizedNameWithoutPath
        );
        assert_eq!(meter.heap_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(
            meter.released_heap_bytes.load(Ordering::Relaxed),
            meter.reserved_heap_bytes.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn try_resolve_child_appends_segment_to_absolute_scope() {
        let scope = AmlName::from_str("\\_SB_.PCI0").unwrap();
        let segment = NameSeg::from_bytes(*b"_REG").unwrap();

        assert_eq!(scope.try_resolve_child(segment).unwrap().to_string(), "\\_SB_.PCI0._REG");
        assert!(AmlName::from_str("PCI0").unwrap().try_resolve_child(segment).is_err());
    }

    #[test]
    fn try_from_absolute_segments_builds_canonical_rooted_path() {
        let path = AmlName::try_from_absolute_segments(&[
            NameSeg::from_str("_GPE").unwrap(),
            NameSeg::from_str("_L2A").unwrap(),
        ])
        .unwrap();

        assert_eq!(path.to_string(), "\\_GPE._L2A");
    }
}
