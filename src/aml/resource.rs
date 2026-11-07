use super::{object::WrappedObject, try_string_from_str, try_vec_from_slice};
use crate::aml::{AmlError, ExecutionMeter, Operation, object::Object};
use super::allocator::{self, String, Vec};
use bit_field::BitField;
use byteorder::{ByteOrder, LittleEndian};
use core::mem;
pub use smallvec;

#[derive(Debug, PartialEq, Eq)]
pub enum Resource {
    Irq(IrqDescriptor),
    AddressSpace(AddressSpaceDescriptor),
    MemoryRange(MemoryRangeDescriptor),
    IOPort(IOPortDescriptor),
    Dma(DMADescriptor),
    SerialBus(SerialBusDescriptor),
    GPIO(GPIOConnectionDescriptor),
}

const UNSUPPORTED_LARGE_RESOURCE_DESCRIPTORS: &[(u8, &str)] = &[
    (0x01, "24-bit Memory Range Descriptor"),
    (0x02, "Generic Register Descriptor"),
    (0x03, "0x03 Reserved"),
    (0x04, "Vendor-defined Descriptor"),
    (0x05, "32-bit Memory Range Descriptor"),
    (0x0d, "Pin Function Descriptor"),
    (0x0f, "Pin Configuration Descriptor"),
    (0x10, "Pin Group Descriptor"),
    (0x11, "Pin Group Function Descriptor"),
    (0x12, "Pin Group Configuration Descriptor"),
];

const UNSUPPORTED_SMALL_RESOURCE_DESCRIPTORS: &[(u8, &str)] = &[
    (0x06, "Start Dependent Functions Descriptor"),
    (0x07, "End Dependent Functions Descriptor"),
    (0x09, "Fixed Location IO Port Descriptor"),
    (0x0a, "Fixed DMA Descriptor"),
    (0x0e, "Vendor Defined Descriptor"),
];

const MAX_RESOURCE_DESCRIPTORS: usize = 256;
const INITIAL_RESOURCE_CAPACITY: usize = 4;
const RESOURCE_DEADLINE_SAMPLE_INTERVAL: usize = 256;

struct ResourceScanBudget<'a> {
    meter: Option<&'a dyn ExecutionMeter>,
    iterations: usize,
}

impl ResourceScanBudget<'_> {
    fn charge_iteration(&mut self) -> Result<(), AmlError> {
        if let Some(meter) = self.meter {
            meter.loop_iteration()?;
        }
        self.iterations = self.iterations.checked_add(1).ok_or(AmlError::BudgetExceeded)?;
        if self.iterations % RESOURCE_DEADLINE_SAMPLE_INTERVAL == 0 {
            if let Some(meter) = self.meter {
                meter.check_deadline()?;
            }
        }
        Ok(())
    }

    fn charge_bytes(&mut self, bytes: usize) -> Result<(), AmlError> {
        for _ in 0..bytes {
            self.charge_iteration()?;
        }
        Ok(())
    }

    fn check_deadline(&self) -> Result<(), AmlError> {
        if let Some(meter) = self.meter {
            meter.check_deadline()?;
        }
        Ok(())
    }
}

/// Parse a `ResourceDescriptor` buffer into a list of resources.
pub fn resource_descriptor_list(descriptor: WrappedObject) -> Result<Vec<Resource>, AmlError> {
    resource_descriptor_list_with_meter(descriptor, None)
}

/// Parse a resource template while charging descriptor scans to an AML
/// execution work unit. Unsupported descriptors consume fuel too.
#[doc(hidden)]
pub fn resource_descriptor_list_metered(
    descriptor: WrappedObject,
    meter: &dyn ExecutionMeter,
) -> Result<Vec<Resource>, AmlError> {
    resource_descriptor_list_with_meter(descriptor, Some(meter))
}

fn resource_descriptor_list_with_meter(
    descriptor: WrappedObject,
    meter: Option<&dyn ExecutionMeter>,
) -> Result<Vec<Resource>, AmlError> {
    if let Object::Buffer(ref bytes) = *descriptor {
        let mut descriptors = allocator::vec_new();
        let mut bytes = bytes.as_slice();
        let mut scan_budget = ResourceScanBudget { meter, iterations: 0 };
        let mut unsupported_warning_emitted = false;

        while !bytes.is_empty() {
            scan_budget.charge_iteration()?;

            let (descriptor, remaining_bytes) =
                resource_descriptor(bytes, &mut unsupported_warning_emitted, &mut scan_budget)?;

            if let Some(descriptor) = descriptor {
                if descriptors.len() == MAX_RESOURCE_DESCRIPTORS {
                    return Err(AmlError::HostError(alloc::string::String::from(
                        "ACPI_RESOURCE_COUNT_LIMIT",
                    )));
                }
                reserve_resource_slot(&mut descriptors)?;
                descriptors.push(descriptor);
            }
            bytes = remaining_bytes;
        }

        scan_budget.check_deadline()?;
        Ok(descriptors)
    } else {
        Err(AmlError::InvalidOperationOnObject { op: Operation::ParseResource, typ: descriptor.typ() })
    }
}

fn reserve_resource_slot(resources: &mut Vec<Resource>) -> Result<(), AmlError> {
    if resources.len() < resources.capacity() {
        return Ok(());
    }
    let doubled = resources.capacity().checked_mul(2).unwrap_or(MAX_RESOURCE_DESCRIPTORS);
    let target_capacity = usize::min(MAX_RESOURCE_DESCRIPTORS, usize::max(INITIAL_RESOURCE_CAPACITY, doubled));
    let additional = target_capacity
        .checked_sub(resources.len())
        .filter(|additional| *additional != 0)
        .ok_or(AmlError::BudgetExceeded)?;
    resources.try_reserve_exact(additional).map_err(|_| AmlError::BudgetExceeded)
}

fn unsupported_resource_descriptor_name(
    descriptors: &[(u8, &'static str)],
    descriptor_type: u8,
) -> Option<&'static str> {
    descriptors.iter().find_map(|(typ, name)| (*typ == descriptor_type).then_some(*name))
}

fn skip_unsupported_resource<'a>(
    size: &str,
    descriptor_type: u8,
    descriptor_name: &str,
    remaining_bytes: &'a [u8],
    warning_emitted: &mut bool,
) -> Result<(Option<Resource>, &'a [u8]), AmlError> {
    if !*warning_emitted {
        log::warn!("skipping unsupported {size} resource descriptor type {descriptor_type:#x}: {descriptor_name}");
        *warning_emitted = true;
    }
    Ok((None, remaining_bytes))
}

fn resource_descriptor<'a>(
    bytes: &'a [u8],
    unsupported_warning_emitted: &mut bool,
    scan_budget: &mut ResourceScanBudget<'_>,
) -> Result<(Option<Resource>, &'a [u8]), AmlError> {
    /*
     * If bit 7 of Byte 0 is set, it's a large descriptor. If not, it's a small descriptor.
     */
    if bytes.is_empty() {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    if bytes[0].get_bit(7) {
        /*
         * We're parsing a large item. The descriptor type is encoded in Bits 0-6 of Byte 0. Valid types:
         *      0x00: Reserved
         *      0x01: 24-bit Memory Range Descriptor
         *      0x02: Generic Register Descriptor
         *      0x03: Reserved
         *      0x04: Vendor-defined Descriptor
         *      0x05: 32-bit Memory Range Descriptor
         *      0x06: 32-bit Fixed Memory Range Descriptor
         *      0x07: Address Space Resource Descriptor
         *      0x08: Word Address Space Descriptor
         *      0x09: Extended Interrupt Descriptor
         *      0x0a: QWord Address Space Descriptor
         *      0x0b: Extended Address Space Descriptor
         *      0x0c: GPIO Connection Descriptor
         *      0x0d: Pin Function Descriptor
         *      0x0e: GenericSerialBus Connection Descriptor
         *      0x0f: Pin Configuration Descriptor
         *      0x10: Pin Group Descriptor
         *      0x11: Pin Group Function Descriptor
         *      0x12: Pin Group Configuration Descriptor
         *      0x13-0x7f: Reserved
         *
         * Byte 1 contains bits 0-7 of the length, and Byte 2 contains bits 8-15 of the length. Subsequent
         * bytes contain the actual data items.
         */
        if bytes.len() < 3 {
            return Err(AmlError::InvalidResourceDescriptor);
        }
        let descriptor_type = bytes[0].get_bits(0..7);
        let length = LittleEndian::read_u16(&bytes[1..=2]) as usize;
        let descriptor_length = length.checked_add(3).ok_or(AmlError::InvalidResourceDescriptor)?;
        if descriptor_length > bytes.len() {
            return Err(AmlError::InvalidResourceDescriptor);
        }
        let (descriptor_bytes, remaining_bytes) = bytes.split_at(descriptor_length);

        let descriptor = match descriptor_type {
            0x06 => fixed_memory_descriptor(descriptor_bytes),
            0x07 => address_space_descriptor::<u32>(descriptor_bytes, scan_budget),
            0x08 => address_space_descriptor::<u16>(descriptor_bytes, scan_budget),
            0x09 => extended_interrupt_descriptor(descriptor_bytes, scan_budget),
            0x0a => address_space_descriptor::<u64>(descriptor_bytes, scan_budget),
            0x0b => extended_address_space_descriptor(descriptor_bytes, scan_budget),
            0x0c => gpio_connection_descriptor(descriptor_bytes, scan_budget),
            0x0e => serial_bus_descriptor(descriptor_bytes, scan_budget),

            0x00 | 0x13..=0x7f => Err(AmlError::InvalidResourceDescriptor),
            0x80..=0xff => unreachable!(),
            _ => {
                let Some(descriptor_name) =
                    unsupported_resource_descriptor_name(UNSUPPORTED_LARGE_RESOURCE_DESCRIPTORS, descriptor_type)
                else {
                    return Err(AmlError::InvalidResourceDescriptor);
                };
                return skip_unsupported_resource(
                    "large",
                    descriptor_type,
                    descriptor_name,
                    remaining_bytes,
                    unsupported_warning_emitted,
                );
            }
        }?;

        Ok((Some(descriptor), remaining_bytes))
    } else {
        /*
         * We're parsing a small descriptor. Byte 0 has the format:
         *    | Bits        | Field             |
         *    |-------------|-------------------|
         *    | 0-2         | Length - n bytes  |
         *    | 3-6         | Small item type   |
         *    | 7           | 0 = small item    |
         *
         * The valid types are:
         *      0x00-0x03: Reserved
         *      0x04: IRQ Format Descriptor
         *      0x05: DMA Format Descriptor
         *      0x06: Start Dependent Functions Descriptor
         *      0x07: End Dependent Functions Descriptor
         *      0x08: IO Port Descriptor
         *      0x09: Fixed Location IO Port Descriptor
         *      0x0A: Fixed DMA Descriptor
         *      0x0B-0x0D: Reserved
         *      0x0E: Vendor Defined Descriptor
         *      0x0F: End Tag Descriptor
         */
        let descriptor_type = bytes[0].get_bits(3..=6);
        let length: usize = bytes[0].get_bits(0..=2) as usize;
        let descriptor_length = length + 1;
        if descriptor_length > bytes.len() {
            return Err(AmlError::InvalidResourceDescriptor);
        }
        let (descriptor_bytes, remaining_bytes) = bytes.split_at(descriptor_length);

        let descriptor = match descriptor_type {
            0x00..=0x03 => Err(AmlError::InvalidResourceDescriptor),
            0x04 => irq_format_descriptor(descriptor_bytes, scan_budget),
            0x05 => dma_format_descriptor(descriptor_bytes),
            0x08 => io_port_descriptor(descriptor_bytes),
            0x0B..=0x0D => Err(AmlError::InvalidResourceDescriptor),
            0x0F => return Ok((None, &[])),
            0x10..=0xFF => unreachable!(),
            _ => {
                let Some(descriptor_name) =
                    unsupported_resource_descriptor_name(UNSUPPORTED_SMALL_RESOURCE_DESCRIPTORS, descriptor_type)
                else {
                    return Err(AmlError::InvalidResourceDescriptor);
                };
                return skip_unsupported_resource(
                    "small",
                    descriptor_type,
                    descriptor_name,
                    remaining_bytes,
                    unsupported_warning_emitted,
                );
            }
        }?;

        Ok((Some(descriptor), remaining_bytes))
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InterruptTrigger {
    Edge,
    Level,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InterruptPolarity {
    ActiveHigh,
    ActiveLow,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum AddressSpaceResourceType {
    MemoryRange,
    IORange,
    BusNumberRange,
    VendorDefined(u8),
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum AddressSpaceDecodeType {
    Additive,
    Subtractive,
}

#[derive(Debug, PartialEq, Eq)]
pub struct AddressSpaceDescriptor {
    pub resource_type: AddressSpaceResourceType,
    pub is_maximum_address_fixed: bool,
    pub is_minimum_address_fixed: bool,
    pub decode_type: AddressSpaceDecodeType,
    /// Raw General Flags and Type Specific Flags from the AML descriptor.
    /// Keeping these fields prevents policy layers from losing producer/
    /// consumer, memory, cacheability, and bridge-translation information.
    pub general_flags: u8,
    pub type_specific_flags: u8,

    pub granularity: u64,
    pub address_range: (u64, u64),
    pub translation_offset: u64,
    pub length: u64,
    /// Present only for an Extended Address Space descriptor.
    pub revision_id: Option<u8>,
    /// Present only for an Extended Address Space descriptor.
    pub type_specific_attributes: Option<u64>,
    pub resource_source_index: Option<u8>,
    pub resource_source: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum MemoryRangeDescriptor {
    FixedLocation { is_writable: bool, base_address: u32, range_length: u32 },
}

fn fixed_memory_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * -- 32-bit Fixed Memory Descriptor ---
     * Offset     Field Name                              Definition
     * Byte 0     32-bit Fixed Memory Range Descriptor    Value = 0x86 (10000110B) – Type = 1, Large item name = 0x06
     * Byte 1     Length, bits [7:0]                      Value = 0x09 (9)
     * Byte 2     Length, bits [15:8]                     Value = 0x00
     * Byte 3     Information                             This field provides extra information about this memory.
     *                                                    Bit [7:1]   Ignored
     *                                                    Bit [0]     Write status, _RW
     *                                                        1  writeable (read/write)
     *                                                        0  non-writeable (read-only)
     * Byte 4     Range base address, _BAS bits [7:0]     Address bits [7:0] of the base memory address for which the card may be configured.
     * Byte 5     Range base address, _BAS bits [15:8]    Address bits [15:8] of the base memory address for which the card may be configured.
     * Byte 6     Range base address, _BAS bits [23:16]   Address bits [23:16] of the base memory address for which the card may be configured.
     * Byte 7     Range base address, _BAS bits [31:24]   Address bits [31:24] of the base memory address for which the card may be configured.
     * Byte 8     Range length, _LEN bits [7:0]           This field contains bits [7:0] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     * Byte 9     Range length, _LEN bits [15:8]          This field contains bits [15:8] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     * Byte 10    Range length, _LEN bits [23:16]         This field contains bits [23:16] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     * Byte 11    Range length, _LEN bits [31:24]         This field contains bits [31:24] of the memory range length. The range length provides the length of the memory range in 1-byte blocks.
     */
    if bytes.len() != 12 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let information = bytes[3];
    let is_writable = information.get_bit(0);

    let base_address = LittleEndian::read_u32(&bytes[4..=7]);
    let range_length = LittleEndian::read_u32(&bytes[8..=11]);

    Ok(Resource::MemoryRange(MemoryRangeDescriptor::FixedLocation { is_writable, base_address, range_length }))
}

fn address_space_resource_type(value: u8) -> Result<AddressSpaceResourceType, AmlError> {
    match value {
        0 => Ok(AddressSpaceResourceType::MemoryRange),
        1 => Ok(AddressSpaceResourceType::IORange),
        2 => Ok(AddressSpaceResourceType::BusNumberRange),
        3..=191 => Err(AmlError::InvalidResourceDescriptor),
        192..=255 => Ok(AddressSpaceResourceType::VendorDefined(value)),
    }
}

fn parse_resource_source(
    bytes: &[u8],
    scan_budget: &mut ResourceScanBudget<'_>,
) -> Result<(Option<u8>, Option<String>), AmlError> {
    if bytes.is_empty() {
        return Ok((None, None));
    }
    let (&index, source_bytes) = bytes.split_first().ok_or(AmlError::InvalidResourceDescriptor)?;
    let mut terminator = None;
    for (offset, byte) in source_bytes.iter().enumerate() {
        scan_budget.charge_iteration()?;
        if *byte == 0 {
            terminator = Some(offset);
            break;
        }
    }
    let terminator = terminator.ok_or(AmlError::InvalidResourceDescriptor)?;
    if terminator == 0 || terminator > 255 || terminator + 1 != source_bytes.len() {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let source_result = core::str::from_utf8(&source_bytes[..terminator]);
    scan_budget.check_deadline()?;
    let source = source_result.map_err(|_| AmlError::InvalidResourceDescriptor)?;
    Ok((Some(index), Some(try_string_from_str(source)?)))
}

fn address_space_descriptor<T>(
    bytes: &[u8],
    scan_budget: &mut ResourceScanBudget<'_>,
) -> Result<Resource, AmlError> {
    /*
     * WORD Address Space Descriptor Definition
     * Note: The definitions for DWORD and QWORD are the same other than the width of the address fields.
     *
     * Offset  Field Name                                   Definition
     * Byte 0  WORD Address Space Descriptor                Value = 0x88 (10001000B) – Type = 1, Large item name = 0x08
     * Byte 1  Length, bits [7:0]                           Variable length, minimum value = 0x0D (13)
     * Byte 2  Length, bits [15:8]                          Variable length, minimum value = 0x00
     * Byte 3  Resource Type                                Indicates which type of resource this descriptor describes. Defined values are:
     *                                                        0         Memory range
     *                                                        1         I/O range
     *                                                        2         Bus number range
     *                                                        3–191     Reserved
     *                                                        192-255   Hardware Vendor Defined
     * Byte 4  General Flags                                Flags that are common to all resource types:
     *                                                        Bits [7:4]   Reserved (must be 0)
     *                                                        Bit [3]     Max Address Fixed, _MAF:
     *                                                          1  The specified maximum address is fixed
     *                                                          0  The specified maximum address is not fixed
     *                                                             and can be changed
     *                                                        Bit [2]      Min Address Fixed,_MIF:
     *                                                          1   The specified minimum address is fixed
     *                                                          0   The specified minimum address is not fixed
     *                                                              and can be changed
     *                                                        Bit [1]      Decode Type, _DEC:
     *                                                          1   This bridge subtractively decodes this address          (top level bridges only)
     *                                                          0   This bridge positively decodes this address
     *                                                        Bit [0]      Ignored
     * Byte 5  Type Specific Flags                           Flags that are specific to each resource type. The meaning of the flags in this field depends on the value of the Resource Type field (see above).
     * Byte 6  Address space granularity, _GRA bits[7:0]     A set bit in this mask means that this bit is decoded. All bits less significant than the most significant set bit must be set. (In other words, the value of the full Address Space Granularity field (all 16 bits) must be a number (2n-1).
     * Byte 7  Address space granularity, _GRA bits[15:8]
     * Byte 8  Address range minimum, _MIN, bits [7:0]       For bridges that translate addresses, this is the address space on the secondary side of the bridge.
     * Byte 9  Address range minimum, _MIN, bits [15:8]
     * Byte 10 Address range maximum, _MAX, bits [7:0]       For bridges that translate addresses, this is the address space on the secondary side of the bridge.
     * Byte 11 Address range maximum, _MAX, bits [15:8]
     * Byte 12 Address Translation offset, _TRA, bits [7:0]  For bridges that translate addresses across the bridge, this is the offset that must be added to the address on the secondary side to obtain the address on the primary side. Non-bridge devices must list 0 for all Address Translation offset bits.
     * Byte 13 Address Translation offset, _TRA, bits [15:8]
     * Byte 14 Address Length, _LEN, bits [7:0]
     * Byte 15 Address Length, _LEN, bits [15:8]
     * Byte 16 Resource Source Index (Optional)              Only present if Resource Source (below) is present. This field gives an index to the specific resource descriptor that this device consumes from in the current resource template for the device object pointed to in Resource Source.
     * String  Resource Source (Optional)                    If present, the device that uses this descriptor consumes its resources from the resources produced by the named device object. If not present, the device consumes its resources out of a global pool. If not present, the device consumes this resource from its hierarchical parent.
     */
    let size = mem::size_of::<T>();

    let fields_end = 6 + size * 5;
    if bytes.len() < fields_end || bytes[4] & 0xf0 != 0 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let resource_type = address_space_resource_type(bytes[3])?;

    let general_flags = bytes[4];
    let is_maximum_address_fixed = general_flags.get_bit(3);
    let is_minimum_address_fixed = general_flags.get_bit(2);
    let decode_type = if general_flags.get_bit(1) {
        AddressSpaceDecodeType::Subtractive
    } else {
        AddressSpaceDecodeType::Additive
    };

    let mut address_fields = bytes[6..].chunks_exact(size);

    // it's safe to unwrap because we check the length at the top
    let granularity = LittleEndian::read_uint(address_fields.next().unwrap(), size);
    let address_range_min = LittleEndian::read_uint(address_fields.next().unwrap(), size);
    let address_range_max = LittleEndian::read_uint(address_fields.next().unwrap(), size);
    let translation_offset = LittleEndian::read_uint(address_fields.next().unwrap(), size);
    let length = LittleEndian::read_uint(address_fields.next().unwrap(), size);
    let (resource_source_index, resource_source) = parse_resource_source(&bytes[fields_end..], scan_budget)?;

    Ok(Resource::AddressSpace(AddressSpaceDescriptor {
        resource_type,
        is_maximum_address_fixed,
        is_minimum_address_fixed,
        decode_type,
        general_flags,
        type_specific_flags: bytes[5],
        granularity,
        address_range: (address_range_min, address_range_max),
        translation_offset,
        length,
        revision_id: None,
        type_specific_attributes: None,
        resource_source_index,
        resource_source,
    }))
}

fn extended_address_space_descriptor(
    bytes: &[u8],
    scan_budget: &mut ResourceScanBudget<'_>,
) -> Result<Resource, AmlError> {
    const EXTENDED_FIELDS_END: usize = 56;
    if bytes.len() < EXTENDED_FIELDS_END || bytes[4] & 0xf0 != 0 || bytes[6] != 1 || bytes[7] != 0 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let resource_type = address_space_resource_type(bytes[3])?;
    let general_flags = bytes[4];
    let mut address_fields = bytes[8..48].chunks_exact(8);
    let granularity = LittleEndian::read_u64(address_fields.next().ok_or(AmlError::InvalidResourceDescriptor)?);
    let address_range_min =
        LittleEndian::read_u64(address_fields.next().ok_or(AmlError::InvalidResourceDescriptor)?);
    let address_range_max =
        LittleEndian::read_u64(address_fields.next().ok_or(AmlError::InvalidResourceDescriptor)?);
    let translation_offset =
        LittleEndian::read_u64(address_fields.next().ok_or(AmlError::InvalidResourceDescriptor)?);
    let length = LittleEndian::read_u64(address_fields.next().ok_or(AmlError::InvalidResourceDescriptor)?);
    let type_specific_attributes = Some(LittleEndian::read_u64(&bytes[48..56]));
    let (resource_source_index, resource_source) =
        parse_resource_source(&bytes[EXTENDED_FIELDS_END..], scan_budget)?;

    Ok(Resource::AddressSpace(AddressSpaceDescriptor {
        resource_type,
        is_maximum_address_fixed: general_flags.get_bit(3),
        is_minimum_address_fixed: general_flags.get_bit(2),
        decode_type: if general_flags.get_bit(1) {
            AddressSpaceDecodeType::Subtractive
        } else {
            AddressSpaceDecodeType::Additive
        },
        general_flags,
        type_specific_flags: bytes[5],
        granularity,
        address_range: (address_range_min, address_range_max),
        translation_offset,
        length,
        revision_id: Some(bytes[6]),
        type_specific_attributes,
        resource_source_index,
        resource_source,
    }))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Irqs(Vec<u32>);

impl Irqs {
    pub fn new() -> Self {
        Self(allocator::vec_new())
    }

    pub fn from_buf<const N: usize>(values: [u32; N]) -> Self {
        Self(allocator::vec_from_array(values))
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), alloc::collections::TryReserveError> {
        self.0.try_reserve(additional)
    }

    pub fn push(&mut self, value: u32) {
        self.0.push(value);
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
}

impl Default for Irqs {
    fn default() -> Self {
        Self::new()
    }
}

impl core::ops::Deref for Irqs {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IrqDescriptor {
    pub is_consumer: bool,
    pub trigger: InterruptTrigger,
    pub polarity: InterruptPolarity,
    pub is_shared: bool,
    pub is_wake_capable: bool,
    pub irqs: Irqs,
}

fn irq_format_descriptor(bytes: &[u8], scan_budget: &mut ResourceScanBudget<'_>) -> Result<Resource, AmlError> {
    /*
     * IRQ Descriptor Definition
     *
     * Offset   Field Name
     * Byte 0   Value = 0x22 or 0x23 (0010001nB)– Type = 0, Small item name = 0x4, Length = 2 or 3
     * Byte 1   IRQ mask bits[7:0], _INT
     *          Bit [0] represents IRQ0, bit[1] is IRQ1, and so on.
     * Byte 2   IRQ mask bits[15:8], _INT
     *          Bit [0] represents IRQ8, bit[1] is IRQ9, and so on.
     * Byte 3   IRQ Information. Each bit, when set, indicates this device is capable of driving a certain type of interrupt.
     *          (Optional—if not included then assume edge sensitive, high true interrupts.)
     *          These bits can be used both for reporting and setting IRQ resources.
     *          Note: This descriptor is meant for describing interrupts that are connected to PIC-compatible interrupt
     *                controllers, which can only be programmed for Active-High-Edge-Triggered or Active-Low-Level-Triggered
     *                interrupts. Any other combination is invalid. The Extended Interrupt Descriptor can be used to describe
     *                other combinations.
     *            Bit [7:6]  Reserved (must be 0)
     *            Bit [5]    Wake Capability, _WKC
     *              0x0 = Not Wake Capable: This interrupt is not capable of waking the system.
     *              0x1 = Wake Capable: This interrupt is capable of waking the system from a
     *                    low-power idle state or a system sleep state.
     *            Bit [4]    Interrupt Sharing, _SHR
     *              0x0 = Exclusive: This interrupt is not shared with other devices.
     *              0x1 = Shared: This interrupt is shared with other devices.
     *            Bit [3]    Interrupt Polarity, _LL
     *              0  Active-High – This interrupt is sampled when the signal is high, or true
     *              1  Active-Low – This interrupt is sampled when the signal is low, or false.
     *            Bit [2:1]  Ignored
     *            Bit [0]    Interrupt Mode, _HE
     *              0  Level-Triggered – Interrupt is triggered in response to signal in a low state.
     *              1  Edge-Triggered – Interrupt is triggered in response to a change in signal state from low to high.
     */
    fn irqs_from_bit_mask(bit_mask: u16, scan_budget: &mut ResourceScanBudget<'_>) -> Result<Irqs, AmlError> {
        let mut irqs = Irqs::new();
        irqs.try_reserve(bit_mask.count_ones() as usize).map_err(|_| AmlError::BudgetExceeded)?;
        for bit in 0..16 {
            scan_budget.charge_iteration()?;
            if bit_mask.get_bit(bit) {
                irqs.push(bit as u32);
            }
        }
        Ok(irqs)
    }

    match bytes.len() {
        0..=2 => Err(AmlError::InvalidResourceDescriptor),
        3 => {
            // no IRQ information ("length 2" in spec)
            let irq = LittleEndian::read_u16(&bytes[1..=2]);

            Ok(Resource::Irq(IrqDescriptor {
                irqs: irqs_from_bit_mask(irq, scan_budget)?,
                is_wake_capable: false,
                is_shared: false,
                polarity: InterruptPolarity::ActiveHigh,
                trigger: InterruptTrigger::Edge,

                is_consumer: false, // assumed to be producer
            }))
        }
        4 => {
            // with IRQ information ("length 3" in spec)
            let irq = LittleEndian::read_u16(&bytes[1..=2]);

            let information = bytes[3];
            let is_wake_capable = information.get_bit(5);
            let is_shared = information.get_bit(4);
            let polarity = match information.get_bit(3) {
                false => InterruptPolarity::ActiveHigh,
                true => InterruptPolarity::ActiveLow,
            };
            let trigger = match information.get_bit(0) {
                false => InterruptTrigger::Level,
                true => InterruptTrigger::Edge,
            };

            Ok(Resource::Irq(IrqDescriptor {
                irqs: irqs_from_bit_mask(irq, scan_budget)?,
                is_wake_capable,
                is_shared,
                polarity,
                trigger,

                is_consumer: false, // assumed to be producer
            }))
        }
        _ => Err(AmlError::InvalidResourceDescriptor),
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DMASupportedSpeed {
    CompatibilityMode,
    TypeA, // as described by the EISA
    TypeB,
    TypeF,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DMATransferTypePreference {
    _8BitOnly,
    _8And16Bit,
    _16Bit,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DMADescriptor {
    pub channel_mask: u8,
    pub supported_speeds: DMASupportedSpeed,
    pub is_bus_master: bool,
    pub transfer_type_preference: DMATransferTypePreference,
}

pub fn dma_format_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * DMA Descriptor Definition
     * Offset  Field Name
     * Byte 0  Value = 0x2A (00101010B) – Type = 0, Small item name = 0x5, Length = 2
     * Byte 1  DMA channel mask bits [7:0] (channels 0 – 7), _DMA
     *         Bit [0] is channel 0, etc.
     * Byte 2  Bit [7]           Reserved (must be 0)
     *         Bits [6:5]        DMA channel speed supported, _TYP
     *           00    Indicates compatibility mode
     *           01    Indicates Type A DMA as described in the EISA
     *           10    Indicates Type B DMA
     *           11    Indicates Type F
     *         Bits [4:3]        Ignored
     *         Bit [2]           Logical device bus master status, _BM
     *           0      Logical device is not a bus master
     *           1      Logical device is a bus master
     *         Bits [1:0]       DMA transfer type preference, _SIZ
     *           00    8-bit only
     *           01    8- and 16-bit
     *           10    16-bit only
     *           11    Reserved
     */
    if bytes.len() != 3 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let channel_mask = bytes[1];
    let options = bytes[2];
    let supported_speeds = match options.get_bits(5..=6) {
        0 => DMASupportedSpeed::CompatibilityMode,
        1 => DMASupportedSpeed::TypeA,
        2 => DMASupportedSpeed::TypeB,
        3 => DMASupportedSpeed::TypeF,
        _ => unreachable!(),
    };
    let is_bus_master = options.get_bit(2);
    let transfer_type_preference = match options.get_bits(0..=1) {
        0 => DMATransferTypePreference::_8BitOnly,
        1 => DMATransferTypePreference::_8And16Bit,
        2 => DMATransferTypePreference::_16Bit,
        3 => return Err(AmlError::InvalidResourceDescriptor),
        _ => unreachable!(),
    };

    Ok(Resource::Dma(DMADescriptor { channel_mask, supported_speeds, is_bus_master, transfer_type_preference }))
}

#[derive(Debug, PartialEq, Eq)]
pub struct IOPortDescriptor {
    pub decodes_full_address: bool,
    pub memory_range: (u16, u16),
    pub base_alignment: u8,
    pub range_length: u8,
}

fn io_port_descriptor(bytes: &[u8]) -> Result<Resource, AmlError> {
    /*
     * I/O Port Descriptor Definition
     * Offset   Field Name                                  Definition
     * Byte 0   I/O Port Descriptor                         Value = 0x47 (01000111B) –
     *                                                      Type = 0, Small item name = 0x8, Length = 7
     * Byte 1   Information                                 Bits [7:1]     Reserved and must be 0
     *                                                      Bit [0]          (_DEC)
     *                                                        1    The logical device decodes 16-bit addresses
     *                                                        0    The logical device only decodes address bits[9:0]
     * Byte 2   Range minimum base address, _MIN bits[7:0]  Address bits [7:0] of the minimum base I/O address that the card may be configured for.
     * Byte 3   Range minimum base address, _MIN bits[15:8] Address bits [15:8] of the minimum base I/O address that the card may be configured for.
     * Byte 4   Range maximum base address, _MAX bits[7:0]  Address bits [7:0] of the maximum base I/O address that the card may be configured for.
     * Byte 5   Range maximum base address, _MAX bits[15:8] Address bits [15:8] of the maximum base I/O address that the card may be configured for.
     * Byte 6   Base alignment, _ALN                        Alignment for minimum base address, increment in 1-byte blocks.
     * Byte 7   Range length, _LEN                          The number of contiguous I/O ports requested.
     */
    if bytes.len() != 8 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let information = bytes[1];
    let decodes_full_address = information.get_bit(0);

    let memory_range_min = LittleEndian::read_u16(&bytes[2..=3]);
    let memory_range_max = LittleEndian::read_u16(&bytes[4..=5]);
    let memory_range = (memory_range_min, memory_range_max);

    let base_alignment = bytes[6];
    let range_length = bytes[7];

    Ok(Resource::IOPort(IOPortDescriptor { decodes_full_address, memory_range, base_alignment, range_length }))
}

fn extended_interrupt_descriptor(
    bytes: &[u8],
    scan_budget: &mut ResourceScanBudget<'_>,
) -> Result<Resource, AmlError> {
    /*
     * --- Extended Interrupt Descriptor ---
     * Byte 3 contains the Interrupt Vector Flags:
     *      Bit 0: 1 if device consumes the resource, 0 if it produces it
     *      Bit 1: 1 if edge-triggered, 0 if level-triggered
     *      Bit 2: 1 = active-high, 0 = active-low
     *      Bit 3: 1 if interrupt is shared with other devices
     *      Bit 4: 1 if this interrupt is capable of waking the system, 0 if it is not
     * Byte 4 contains the number of interrupt numbers that follow. When this descriptor is
     * returned from `_CRS` or send to `_SRS`, this field must be 1.
     *
     * From Byte 5 onwards, there are `n` interrupt numbers, each of which is encoded as a
     * 4-byte little-endian number.
     *
     * NOTE: We only support the case where there is a single interrupt number.
     */
    if bytes.len() < 9 {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let number_of_interrupts = bytes[4] as usize;
    let interrupt_bytes = number_of_interrupts
        .checked_mul(size_of::<u32>())
        .and_then(|length| 5_usize.checked_add(length))
        .ok_or(AmlError::InvalidResourceDescriptor)?;
    if interrupt_bytes > bytes.len() {
        return Err(AmlError::InvalidResourceDescriptor);
    }

    let irqs = if number_of_interrupts == 1 {
        let irq = LittleEndian::read_u32(&bytes[5..9]);

        Irqs::from_buf([irq])
    } else {
        let mut irqs = Irqs::new();
        irqs.try_reserve(number_of_interrupts).map_err(|_| AmlError::BudgetExceeded)?;

        for i in 0..number_of_interrupts {
            scan_budget.charge_iteration()?;
            let start = 5 + i * size_of::<u32>();
            let end = start + 4;

            let irq = LittleEndian::read_u32(&bytes[start..end]);

            irqs.push(irq);
        }

        irqs
    };

    Ok(Resource::Irq(IrqDescriptor {
        is_consumer: bytes[3].get_bit(0),
        trigger: if bytes[3].get_bit(1) { InterruptTrigger::Edge } else { InterruptTrigger::Level },
        polarity: if bytes[3].get_bit(2) { InterruptPolarity::ActiveLow } else { InterruptPolarity::ActiveHigh },
        is_shared: bytes[3].get_bit(3),
        is_wake_capable: bytes[3].get_bit(4),
        irqs,
    }))
}

#[derive(Debug, PartialEq, Eq)]
pub enum ShareType {
    Shared,
    Exclusive,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ResourceUsage {
    ResourceConsumer,
    ResourceProducer,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SlaveMode {
    ControllerInitiated,
    DeviceInitiated,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AddressingMode {
    AddressingMode7Bit,
    AddressingMode10Bit,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SerialBusDescriptor {
    pub sharing: ShareType,
    pub usage: ResourceUsage,
    pub slave_mode: SlaveMode,
    pub source: String,
    pub source_index: u8,
    pub vendor_data: Vec<u8>,
    pub bus: SerialBus,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SerialBus {
    I2C { addressing_mode: AddressingMode, speed: u32, address: u16 },
}

fn i2c_bus_descriptor(bytes: &[u8]) -> Result<(SerialBus, (usize, usize)), AmlError> {
    if bytes.len() < 18 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    // revision
    if bytes[9] > 1 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let type_length = LittleEndian::read_u16(&bytes[10..=11]) as usize;
    if type_length < 6 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let speed = LittleEndian::read_u32(&bytes[12..=15]);
    let address = LittleEndian::read_u16(&bytes[16..=17]);
    Ok((
        SerialBus::I2C {
            addressing_mode: if bytes[7].get_bit(0) {
                AddressingMode::AddressingMode10Bit
            } else {
                AddressingMode::AddressingMode7Bit
            },
            speed,
            address,
        },
        (18, type_length - 6),
    ))
}

fn serial_bus_descriptor(bytes: &[u8], scan_budget: &mut ResourceScanBudget<'_>) -> Result<Resource, AmlError> {
    if bytes.len() < 12 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    // revision
    if bytes[3] > 2 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let (descriptor, (vendor_data_idx, vendor_data_length)) = match bytes[5] {
        1 => i2c_bus_descriptor(bytes)?,
        _ => return Err(AmlError::LibUnimplemented),
    };
    let vendor_data_end = vendor_data_idx
        .checked_add(vendor_data_length)
        .filter(|end| *end < bytes.len())
        .ok_or(AmlError::InvalidResourceDescriptor)?;
    let vendor_data = try_vec_from_slice(&bytes[vendor_data_idx..vendor_data_end])?;
    if bytes.last() != Some(&0) {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let source_bytes = &bytes[vendor_data_end..bytes.len() - 1];
    scan_budget.charge_bytes(source_bytes.len())?;
    let source_result = str::from_utf8(source_bytes);
    scan_budget.check_deadline()?;
    let source = source_result.map_err(|_| AmlError::InvalidResourceDescriptor)?;

    Ok(Resource::SerialBus(SerialBusDescriptor {
        sharing: if bytes[6].get_bit(2) { ShareType::Shared } else { ShareType::Exclusive },
        usage: if bytes[6].get_bit(1) { ResourceUsage::ResourceConsumer } else { ResourceUsage::ResourceProducer },
        slave_mode: if bytes[6].get_bit(0) { SlaveMode::DeviceInitiated } else { SlaveMode::ControllerInitiated },
        source_index: bytes[4],
        source: try_string_from_str(source)?,
        vendor_data,
        bus: descriptor,
    }))
}

#[derive(Debug, PartialEq, Eq)]
pub enum GPIOInterruptPolarity {
    ActiveHigh,
    ActiveLow,
    ActiveBoth,
}

#[derive(Debug, PartialEq, Eq)]
pub enum WakeCapability {
    NotWakeCapable,
    WakeCapable,
}

#[derive(Debug, PartialEq, Eq)]
pub enum IORestriction {
    None,
    InputOnly,
    OutputOnly,
    NoneAndPreserve,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PinConfiguration {
    Default,
    PullUp,
    PullDown,
    NoPull,
}

#[derive(Debug, PartialEq, Eq)]
pub struct GPIOConnectionDescriptor {
    pub usage: ResourceUsage,
    pub sharing: ShareType,
    pub pin_configuration: PinConfiguration,
    pub output_drive_strength: u16,
    pub debounce_timeout: u16,
    pub pins: Vec<u16>,
    pub source: String,
    pub vendor_data: Vec<u8>,
    pub connection: GPIOConnection,
}

#[derive(Debug, PartialEq, Eq)]
pub enum GPIOConnection {
    Interrupt { trigger: InterruptTrigger, wake_capability: WakeCapability, polarity: GPIOInterruptPolarity },
    IO { io_restriction: IORestriction },
}

fn gpio_connection_descriptor(
    bytes: &[u8],
    scan_budget: &mut ResourceScanBudget<'_>,
) -> Result<Resource, AmlError> {
    if bytes.len() < 23 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    // revision
    if bytes[3] != 1 {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let pin_table_offset = LittleEndian::read_u16(&bytes[14..=15]) as usize;
    let source_name_offset = LittleEndian::read_u16(&bytes[17..=18]) as usize;
    let vendor_data_offset = LittleEndian::read_u16(&bytes[19..=20]) as usize;
    let vendor_data_length = LittleEndian::read_u16(&bytes[21..=22]) as usize;

    let pin_table_bytes = source_name_offset
        .checked_sub(pin_table_offset)
        .filter(|length| length % size_of::<u16>() == 0)
        .ok_or(AmlError::InvalidResourceDescriptor)?;
    let pin_table_end = pin_table_offset
        .checked_add(pin_table_bytes)
        .filter(|end| *end <= bytes.len())
        .ok_or(AmlError::InvalidResourceDescriptor)?;
    if pin_table_offset < 23 || pin_table_end != source_name_offset {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let source_end = bytes
        .len()
        .checked_sub(vendor_data_length)
        .and_then(|end| end.checked_sub(1))
        .filter(|end| *end >= source_name_offset)
        .ok_or(AmlError::InvalidResourceDescriptor)?;
    let vendor_data_end = vendor_data_offset
        .checked_add(vendor_data_length)
        .filter(|end| *end <= bytes.len())
        .ok_or(AmlError::InvalidResourceDescriptor)?;
    if bytes[source_end] != 0
        || (vendor_data_length != 0 && (vendor_data_offset != source_end + 1 || vendor_data_end != bytes.len()))
    {
        return Err(AmlError::InvalidResourceDescriptor);
    }
    let vendor_data = try_vec_from_slice(&bytes[vendor_data_offset..vendor_data_end])?;
    let source_bytes = &bytes[source_name_offset..source_end];
    scan_budget.charge_bytes(source_bytes.len())?;
    let source_result = str::from_utf8(source_bytes);
    scan_budget.check_deadline()?;
    let source = source_result.map_err(|_| AmlError::InvalidResourceDescriptor)?;
    let mut pins = allocator::vec_new();
    let pin_count = pin_table_bytes / size_of::<u16>();
    pins.try_reserve_exact(pin_count).map_err(|_| AmlError::BudgetExceeded)?;
    for pin in bytes[pin_table_offset..pin_table_end].chunks_exact(size_of::<u16>()) {
        scan_budget.charge_iteration()?;
        pins.push(LittleEndian::read_u16(pin));
    }

    Ok(Resource::GPIO(GPIOConnectionDescriptor {
        usage: if bytes[5].get_bit(0) { ResourceUsage::ResourceConsumer } else { ResourceUsage::ResourceProducer },
        sharing: if bytes[7].get_bit(3) { ShareType::Shared } else { ShareType::Exclusive },
        pin_configuration: match bytes[9] {
            0x00 => PinConfiguration::Default,
            0x01 => PinConfiguration::PullUp,
            0x02 => PinConfiguration::PullDown,
            0x03 => PinConfiguration::NoPull,
            _ => {
                return Err(AmlError::InvalidResourceDescriptor);
            }
        },
        output_drive_strength: LittleEndian::read_u16(&bytes[10..=11]),
        debounce_timeout: LittleEndian::read_u16(&bytes[12..=13]),
        pins,
        source: try_string_from_str(source)?,
        vendor_data,
        connection: match bytes[4] {
            0 => GPIOConnection::Interrupt {
                trigger: match bytes[7].get_bit(0) {
                    true => InterruptTrigger::Edge,
                    false => InterruptTrigger::Level,
                },
                wake_capability: match bytes[7].get_bit(4) {
                    true => WakeCapability::WakeCapable,
                    false => WakeCapability::NotWakeCapable,
                },
                polarity: match bytes[7].get_bits(1..=2) {
                    0x0 => GPIOInterruptPolarity::ActiveHigh,
                    0x1 => GPIOInterruptPolarity::ActiveLow,
                    0x2 => GPIOInterruptPolarity::ActiveBoth,
                    _ => return Err(AmlError::InvalidResourceDescriptor),
                },
            },
            1 => GPIOConnection::IO {
                io_restriction: match bytes[7].get_bits(0..=1) {
                    0b00 => IORestriction::None,
                    0b01 => IORestriction::InputOnly,
                    0b10 => IORestriction::OutputOnly,
                    0b11 => IORestriction::NoneAndPreserve,
                    _ => unreachable!(),
                },
            },
            _ => {
                return Err(AmlError::InvalidResourceDescriptor);
            }
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aml::ExecutionMeter;
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct LoopBudgetMeter {
        maximum_iterations: usize,
        iterations: AtomicUsize,
        deadline_checks: AtomicUsize,
        reject_deadline: bool,
    }

    impl ExecutionMeter for LoopBudgetMeter {
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
    fn test_parses_keyboard_crs() {
        let bytes: Vec<u8> = [
            // Generated from `iasl -l pc-bios_acpi-dsdt.asl`
            //
            //         315:                   IO (Decode16,
            //         316:                       0x0060,             // Range Minimum
            //         317:                       0x0060,             // Range Maximum
            //         318:                       0x01,               // Alignment
            //         319:                       0x01,               // Length
            //         320:                       )

            //    0000040A:  47 01 60 00 60 00 01 01     "G.`.`..."
            0x47, 0x01, 0x60, 0x00, 0x60, 0x00, 0x01, 0x01,
            //         321:                   IO (Decode16,
            //         322:                       0x0064,             // Range Minimum
            //         323:                       0x0064,             // Range Maximum
            //         324:                       0x01,               // Alignment
            //         325:                       0x01,               // Length
            //         326:                       )

            //    00000412:  47 01 64 00 64 00 01 01     "G.d.d..."
            0x47, 0x01, 0x64, 0x00, 0x64, 0x00, 0x01, 0x01,
            //         327:                   IRQNoFlags ()
            //         328:                       {1}

            //    0000041A:  22 02 00 ...............    "".."
            0x22, 0x02, 0x00, //    0000041D:  79 00 ..................    "y."
            0x79, 0x00,
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            allocator::vec_from_array([
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x60, 0x60),
                    base_alignment: 1,
                    range_length: 1
                }),
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x64, 0x64),
                    base_alignment: 1,
                    range_length: 1
                }),
                Resource::Irq(IrqDescriptor {
                    is_consumer: false,
                    trigger: InterruptTrigger::Edge,
                    polarity: InterruptPolarity::ActiveHigh,
                    is_shared: false,
                    is_wake_capable: false,
                irqs: Irqs::from_buf([1])
                })
            ])
        );
    }

    #[test]
    fn test_pci_crs() {
        let bytes: Vec<u8> = [
            // Generated from `iasl -l pc-bios_acpi-dsdt.asl`
            //
            //      98:               WordBusNumber (ResourceProducer, MinFixed, MaxFixed, PosDecode,
            //      99:                   0x0000,             // Granularity
            //     100:                   0x0000,             // Range Minimum
            //     101:                   0x00FF,             // Range Maximum
            //     102:                   0x0000,             // Translation Offset
            //     103:                   0x0100,             // Length
            //     104:                   ,, )

            // 000000F3:  88 0D 00 02 0C 00 00 00     "........"
            // 000000FB:  00 00 FF 00 00 00 00 01     "........"
            0x88, 0x0D, 0x00, 0x02, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x01,
            //     105:               IO (Decode16,
            //     106:                   0x0CF8,             // Range Minimum
            //     107:                   0x0CF8,             // Range Maximum
            //     108:                   0x01,               // Alignment
            //     109:                   0x08,               // Length
            //     110:                   )

            // 00000103:  47 01 F8 0C F8 0C 01 08     "G......."
            0x47, 0x01, 0xF8, 0x0C, 0xF8, 0x0C, 0x01, 0x08,
            //     111:               WordIO (ResourceProducer, MinFixed, MaxFixed, PosDecode, EntireRange,
            //     112:                   0x0000,             // Granularity
            //     113:                   0x0000,             // Range Minimum
            //     114:                   0x0CF7,             // Range Maximum
            //     115:                   0x0000,             // Translation Offset
            //     116:                   0x0CF8,             // Length
            //     117:                   ,, , TypeStatic, DenseTranslation)

            // 0000010B:  88 0D 00 01 0C 03 00 00     "........"
            // 00000113:  00 00 F7 0C 00 00 F8 0C     "........"
            0x88, 0x0D, 0x00, 0x01, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x00, 0xF7, 0x0C, 0x00, 0x00, 0xF8, 0x0C,
            //     118:               WordIO (ResourceProducer, MinFixed, MaxFixed, PosDecode, EntireRange,
            //     119:                   0x0000,             // Granularity
            //     120:                   0x0D00,             // Range Minimum
            //     121:                   0xFFFF,             // Range Maximum
            //     122:                   0x0000,             // Translation Offset
            //     123:                   0xF300,             // Length
            //     124:                   ,, , TypeStatic, DenseTranslation)

            // 0000011B:  88 0D 00 01 0C 03 00 00     "........"
            // 00000123:  00 0D FF FF 00 00 00 F3     "........"
            0x88, 0x0D, 0x00, 0x01, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x0D, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xF3,
            //     125:               DWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, Cacheable, ReadWrite,
            //     126:                   0x00000000,         // Granularity
            //     127:                   0x000A0000,         // Range Minimum
            //     128:                   0x000BFFFF,         // Range Maximum
            //     129:                   0x00000000,         // Translation Offset
            //     130:                   0x00020000,         // Length
            //     131:                   ,, , AddressRangeMemory, TypeStatic)

            // 0000012B:  87 17 00 00 0C 03 00 00     "........"
            // 00000133:  00 00 00 00 0A 00 FF FF     "........"
            // 0000013B:  0B 00 00 00 00 00 00 00     "........"
            // 00000143:  02 00 ..................    ".."
            0x87, 0x17, 0x00, 0x00, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0xFF, 0xFF, 0x0B,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
            //     132:               DWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, NonCacheable, ReadWrite,
            //     133:                   0x00000000,         // Granularity
            //     134:                   0xE0000000,         // Range Minimum
            //     135:                   0xFEBFFFFF,         // Range Maximum
            //     136:                   0x00000000,         // Translation Offset
            //     137:                   0x1EC00000,         // Length
            //     138:                   ,, _Y00, AddressRangeMemory, TypeStatic)

            // 00000145:  87 17 00 00 0C 01 00 00     "........"
            // 0000014D:  00 00 00 00 00 E0 FF FF     "........"
            // 00000155:  BF FE 00 00 00 00 00 00     "........"
            // 0000015D:  C0 1E ..................    ".."
            0x87, 0x17, 0x00, 0x00, 0x0C, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0xFF, 0xFF, 0xBF,
            0xFE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC0, 0x1E,
            // 0000015F:  79 00 ..................    "y."
            0x79, 0x00,
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            allocator::vec_from_array([
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::BusNumberRange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    general_flags: 0x0c,
                    type_specific_flags: 0,
                    granularity: 0,
                    address_range: (0x00, 0xFF),
                    translation_offset: 0,
                    length: 0x100,
                    revision_id: None,
                    type_specific_attributes: None,
                    resource_source_index: None,
                    resource_source: None,
                }),
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0xCF8, 0xCF8),
                    base_alignment: 1,
                    range_length: 8
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::IORange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    general_flags: 0x0c,
                    type_specific_flags: 0x03,
                    granularity: 0,
                    address_range: (0x0000, 0x0CF7),
                    translation_offset: 0,
                    length: 0xCF8,
                    revision_id: None,
                    type_specific_attributes: None,
                    resource_source_index: None,
                    resource_source: None,
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::IORange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    general_flags: 0x0c,
                    type_specific_flags: 0x03,
                    granularity: 0,
                    address_range: (0x0D00, 0xFFFF),
                    translation_offset: 0,
                    length: 0xF300,
                    revision_id: None,
                    type_specific_attributes: None,
                    resource_source_index: None,
                    resource_source: None,
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::MemoryRange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    general_flags: 0x0c,
                    type_specific_flags: 0x03,
                    granularity: 0,
                    address_range: (0xA0000, 0xBFFFF),
                    translation_offset: 0,
                    length: 0x20000,
                    revision_id: None,
                    type_specific_attributes: None,
                    resource_source_index: None,
                    resource_source: None,
                }),
                Resource::AddressSpace(AddressSpaceDescriptor {
                    resource_type: AddressSpaceResourceType::MemoryRange,
                    is_maximum_address_fixed: true,
                    is_minimum_address_fixed: true,
                    decode_type: AddressSpaceDecodeType::Additive,
                    general_flags: 0x0c,
                    type_specific_flags: 0x01,
                    granularity: 0,
                    address_range: (0xE0000000, 0xFEBFFFFF),
                    translation_offset: 0,
                    length: 0x1EC00000,
                    revision_id: None,
                    type_specific_attributes: None,
                    resource_source_index: None,
                    resource_source: None,
                }),
            ])
        );
    }

    #[test]
    fn parses_extended_address_space_descriptor_without_losing_window_values() {
        let mut bytes = crate::aml_vec![0x8b, 0x35, 0, 0, 0x0c, 0x01, 1, 0];
        for value in [0x0fff_u64, 0x1000_0000, 0x1fff_ffff, 0x1000, 0x1000_0000, 0xfeed_face_cafe_beef] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0x79, 0]);

        let resources = resource_descriptor_list(Object::Buffer(bytes).wrap()).unwrap();
        assert_eq!(resources.len(), 1);
        let Resource::AddressSpace(window) = &resources[0] else {
            panic!("expected extended address-space resource");
        };
        assert_eq!(window.resource_type, AddressSpaceResourceType::MemoryRange);
        assert_eq!(window.general_flags, 0x0c);
        assert_eq!(window.type_specific_flags, 0x01);
        assert_eq!(window.address_range, (0x1000_0000, 0x1fff_ffff));
        assert_eq!(window.translation_offset, 0x1000);
        assert_eq!(window.length, 0x1000_0000);
        assert_eq!(window.revision_id, Some(1));
        assert_eq!(window.type_specific_attributes, Some(0xfeed_face_cafe_beef));
    }

    #[test]
    fn preserves_vendor_address_space_type_without_panicking() {
        let mut bytes = crate::aml_vec![0x8a, 0x2b, 0, 0xc0];
        bytes.resize(46, 0);
        bytes.extend_from_slice(&[0x79, 0]);

        let resources = resource_descriptor_list(Object::Buffer(bytes).wrap()).unwrap();
        let [Resource::AddressSpace(window)] = resources.as_slice() else {
            panic!("expected one vendor-defined address-space descriptor");
        };
        assert_eq!(window.resource_type, AddressSpaceResourceType::VendorDefined(0xc0));
    }

    #[test]
    fn rejects_reserved_dma_transfer_preference_without_panicking() {
        let bytes = crate::aml_vec![0x2a, 0x01, 0x03, 0x79, 0];
        assert!(resource_descriptor_list(Object::Buffer(bytes).wrap()).is_err());
    }

    #[test]
    fn rejects_truncated_large_resource_header_without_panicking() {
        let bytes = crate::aml_vec![0x8a];
        assert!(resource_descriptor_list(Object::Buffer(bytes).wrap()).is_err());
    }

    #[test]
    fn single_bit_resource_descriptor_mutations_never_panic() {
        let templates = [
            crate::aml_vec![
                0x47, 0x01, 0x60, 0x00, 0x60, 0x00, 0x01, 0x01, // IO
                0x22, 0x02, 0x00, // IRQ
                0x79, 0x00, // End tag
            ],
            {
                let mut bytes = crate::aml_vec![0x8b, 0x35, 0, 0, 0x0c, 0x01, 1, 0];
                for value in [0x0fff_u64, 0x1000_0000, 0x1fff_ffff, 0x1000, 0x1000_0000, 0xfeed_face_cafe_beef] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes.extend_from_slice(&[0x79, 0]);
                bytes
            },
        ];

        for (template_index, template) in templates.iter().enumerate() {
            for byte_index in 0..template.len() {
                for bit in 0..u8::BITS {
                    let mut mutated = template.clone();
                    mutated[byte_index] ^= 1_u8 << bit;

                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        resource_descriptor_list(Object::Buffer(mutated).wrap())
                    }));
                    assert!(
                        outcome.is_ok(),
                        "resource mutation panicked: template={template_index}, byte={byte_index}, bit={bit}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_large_resource_length_overrun_without_panicking() {
        let bytes = crate::aml_vec![0x8a, 0x2b, 0x00];
        assert!(resource_descriptor_list(Object::Buffer(bytes).wrap()).is_err());
    }

    #[test]
    fn skips_unsupported_large_resource_descriptors() {
        let bytes: Vec<u8> = [
            0x82, 0x01, 0x00, 0x00, // Generic Register descriptor with one byte of payload.
            0x86, 0x09, 0x00, 0x01, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00,
            0x00, // Memory32Fixed(ReadWrite, 0x1000, 0x1000)
            0x79, 0x00, // End tag.
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            allocator::vec_from_array([Resource::MemoryRange(MemoryRangeDescriptor::FixedLocation {
                is_writable: true,
                base_address: 0x1000,
                range_length: 0x1000,
            })])
        );
    }

    #[test]
    fn skips_unsupported_small_resource_descriptors() {
        let bytes: Vec<u8> = [
            0x71, 0x00, // Vendor-defined small descriptor with one byte of payload.
            0x86, 0x09, 0x00, 0x01, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00,
            0x00, // Memory32Fixed(ReadWrite, 0x1000, 0x1000)
            0x79, 0x00, // End tag.
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            allocator::vec_from_array([Resource::MemoryRange(MemoryRangeDescriptor::FixedLocation {
                is_writable: true,
                base_address: 0x1000,
                range_length: 0x1000,
            })])
        );
    }

    #[test]
    fn metered_parser_charges_unsupported_descriptors_before_scanning_all_input() {
        const DESCRIPTOR_COUNT: usize = 4096;
        let mut bytes = allocator::vec_new();
        bytes.try_reserve_exact(DESCRIPTOR_COUNT * 2).unwrap();
        for _ in 0..DESCRIPTOR_COUNT {
            bytes.extend_from_slice(&[0x71, 0x00]); // Unsupported vendor descriptor.
        }
        let meter = LoopBudgetMeter {
            maximum_iterations: 8,
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            reject_deadline: false,
        };

        let result = resource_descriptor_list_metered(Object::Buffer(bytes).wrap(), &meter);

        assert_eq!(result, Err(AmlError::BudgetExceeded));
        assert_eq!(meter.iterations.load(Ordering::Relaxed), 9);
        assert_eq!(meter.deadline_checks.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn metered_parser_checks_deadline_during_long_descriptor_scans() {
        const DESCRIPTOR_COUNT: usize = 1024;
        let mut bytes = allocator::vec_new();
        bytes.try_reserve_exact(DESCRIPTOR_COUNT * 2).unwrap();
        for _ in 0..DESCRIPTOR_COUNT {
            bytes.extend_from_slice(&[0x71, 0x00]);
        }
        let meter = LoopBudgetMeter {
            maximum_iterations: DESCRIPTOR_COUNT,
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            reject_deadline: true,
        };

        let result = resource_descriptor_list_metered(Object::Buffer(bytes).wrap(), &meter);

        assert_eq!(result, Err(AmlError::DeadlineExceeded));
        assert_eq!(meter.iterations.load(Ordering::Relaxed), RESOURCE_DEADLINE_SAMPLE_INTERVAL);
        assert_eq!(meter.deadline_checks.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn metered_parser_checks_deadline_while_scanning_resource_source() {
        const FIXED_DESCRIPTOR_BYTES: usize = 46;
        const SOURCE_BYTES: usize = 1024;
        let total_length = FIXED_DESCRIPTOR_BYTES + 1 + SOURCE_BYTES;
        let mut bytes = crate::aml_vec![b'S'; total_length];
        bytes[0] = 0x8a; // QWord Address Space descriptor.
        let payload_length = (total_length - 3) as u16;
        bytes[1..3].copy_from_slice(&payload_length.to_le_bytes());
        bytes[3] = 0; // Memory resource.
        bytes[4] = 0;
        bytes[5] = 0;
        bytes[FIXED_DESCRIPTOR_BYTES] = 0; // Resource Source Index.

        let meter = LoopBudgetMeter {
            maximum_iterations: SOURCE_BYTES,
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            reject_deadline: true,
        };

        let result = resource_descriptor_list_metered(Object::Buffer(bytes).wrap(), &meter);

        assert_eq!(result, Err(AmlError::DeadlineExceeded));
        assert_eq!(meter.iterations.load(Ordering::Relaxed), RESOURCE_DEADLINE_SAMPLE_INTERVAL);
        assert_eq!(meter.deadline_checks.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn metered_parser_charges_gpio_pin_scans() {
        const PIN_COUNT: usize = 8;
        let source_name_offset = 23 + PIN_COUNT * size_of::<u16>();
        let total_length = source_name_offset + 1;
        let mut bytes = crate::aml_vec![0; total_length];
        bytes[0] = 0x8c;
        let payload_length = (total_length - 3) as u16;
        bytes[1..3].copy_from_slice(&payload_length.to_le_bytes());
        bytes[3] = 1; // Revision.
        bytes[4] = 1; // GPIO I/O connection.
        bytes[9] = 0; // Default pin configuration.
        bytes[14..16].copy_from_slice(&(23_u16).to_le_bytes());
        bytes[17..19].copy_from_slice(&(source_name_offset as u16).to_le_bytes());
        for index in 0..PIN_COUNT {
            let pin_offset = 23 + index * size_of::<u16>();
            bytes[pin_offset..pin_offset + 2].copy_from_slice(&(index as u16).to_le_bytes());
        }

        let meter = LoopBudgetMeter {
            maximum_iterations: 4,
            iterations: AtomicUsize::new(0),
            deadline_checks: AtomicUsize::new(0),
            reject_deadline: false,
        };

        let result = resource_descriptor_list_metered(Object::Buffer(bytes).wrap(), &meter);

        assert_eq!(result, Err(AmlError::BudgetExceeded));
        assert_eq!(meter.iterations.load(Ordering::Relaxed), 5);
        assert_eq!(meter.deadline_checks.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_fdc_crs() {
        let bytes: Vec<u8> = [
            //         365:                   IO (Decode16,
            //         366:                       0x03F2,             // Range Minimum
            //         367:                       0x03F2,             // Range Maximum
            //         368:                       0x00,               // Alignment
            //         369:                       0x04,               // Length
            //         370:                       )

            //    0000047C:  47 01 F2 03 F2 03 00 04     "G......."
            0x47, 0x01, 0xF2, 0x03, 0xF2, 0x03, 0x00, 0x04,
            //         371:                   IO (Decode16,
            //         372:                       0x03F7,             // Range Minimum
            //         373:                       0x03F7,             // Range Maximum
            //         374:                       0x00,               // Alignment
            //         375:                       0x01,               // Length
            //         376:                       )

            //    00000484:  47 01 F7 03 F7 03 00 01     "G......."
            0x47, 0x01, 0xF7, 0x03, 0xF7, 0x03, 0x00, 0x01,
            //         377:                   IRQNoFlags ()
            //         378:                       {6}

            //    0000048C:  22 40 00 ...............    ""@."
            0x22, 0x40, 0x00,
            //         379:                   DMA (Compatibility, NotBusMaster, Transfer8, )
            //         380:                       {2}

            //    0000048F:  2A 04 00 ...............    "*.."
            0x2A, 0x04, 0x00, //    00000492:  79 00 ..................    "y."
            0x79, 0x00,
        ]
        .to_vec();

        let value = Object::Buffer(bytes).wrap();
        let resources = resource_descriptor_list(value).unwrap();

        assert_eq!(
            resources,
            allocator::vec_from_array([
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x03F2, 0x03F2),
                    base_alignment: 0,
                    range_length: 4
                }),
                Resource::IOPort(IOPortDescriptor {
                    decodes_full_address: true,
                    memory_range: (0x03F7, 0x03F7),
                    base_alignment: 0,
                    range_length: 1
                }),
                Resource::Irq(IrqDescriptor {
                    is_consumer: false,
                    trigger: InterruptTrigger::Edge,
                    polarity: InterruptPolarity::ActiveHigh,
                    is_shared: false,
                    is_wake_capable: false,
                    irqs: Irqs::from_buf([6])
                }),
                Resource::Dma(DMADescriptor {
                    channel_mask: 1 << 2,
                    supported_speeds: DMASupportedSpeed::CompatibilityMode,
                    is_bus_master: false,
                    transfer_type_preference: DMATransferTypePreference::_8BitOnly
                })
            ])
        );
    }
}
