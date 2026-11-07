//! Native OperationRegion adapter for interpreter tests.

use acpi::{
    Handler,
    aml::{
        AmlError,
        namespace::AmlName,
        op_region::{NativeRegionAccessMode, NativeRegionHandler, NativeRegionToken, OpRegion, RegionSpace},
    },
};
use pci_types::PciAddress;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub struct TestNativeRegionHandler<H> {
    handler: H,
    pci_devices: Vec<(AmlName, PciAddress)>,
    invalidation_count: Arc<AtomicUsize>,
}

impl<H> TestNativeRegionHandler<H> {
    pub fn new(handler: H, pci_devices: Vec<(AmlName, PciAddress)>) -> Self {
        Self { handler, pci_devices, invalidation_count: Arc::new(AtomicUsize::new(0)) }
    }

    pub fn invalidation_count(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.invalidation_count)
    }
}

impl<H> NativeRegionHandler for TestNativeRegionHandler<H>
where
    H: Handler + Send + Sync,
{
    fn acquire_global_lock(&self, _timeout_us: u64) -> Result<(), AmlError> {
        Ok(())
    }

    fn release_global_lock(&self) -> Result<(), AmlError> {
        Ok(())
    }

    fn execution_thread_id(&self) -> Option<u64> {
        Some(1)
    }

    fn invalidate_all(&self) -> Result<(), AmlError> {
        self.invalidation_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn authorize(&self, region: &OpRegion, _mode: NativeRegionAccessMode) -> Result<NativeRegionToken, AmlError> {
        Ok(NativeRegionToken::new(region.clone()))
    }

    fn read(&self, token: &NativeRegionToken, offset: usize, length: usize) -> Result<u64, AmlError> {
        let region = token
            .downcast_ref::<OpRegion>()
            .ok_or_else(|| AmlError::HostError("TEST_REGION_TOKEN".to_string()))?;
        let address = checked_address(region, offset, length)?;
        match (region.space, length) {
            (RegionSpace::SystemMemory, 1) => Ok(u64::from(self.handler.read_u8(address as usize))),
            (RegionSpace::SystemMemory, 2) => Ok(u64::from(self.handler.read_u16(address as usize))),
            (RegionSpace::SystemMemory, 4) => Ok(u64::from(self.handler.read_u32(address as usize))),
            (RegionSpace::SystemMemory, 8) => Ok(self.handler.read_u64(address as usize)),
            (RegionSpace::SystemIO, 1) => Ok(u64::from(self.handler.read_io_u8(to_port(address)?))),
            (RegionSpace::SystemIO, 2) => Ok(u64::from(self.handler.read_io_u16(to_port(address)?))),
            (RegionSpace::SystemIO, 4) => Ok(u64::from(self.handler.read_io_u32(to_port(address)?))),
            (RegionSpace::PciConfig, 1) => {
                let device = self.pci_device(region)?;
                Ok(u64::from(self.handler.read_pci_u8(device, to_config_offset(address)?)))
            }
            (RegionSpace::PciConfig, 2) => {
                let device = self.pci_device(region)?;
                Ok(u64::from(self.handler.read_pci_u16(device, to_config_offset(address)?)))
            }
            (RegionSpace::PciConfig, 4) => {
                let device = self.pci_device(region)?;
                Ok(u64::from(self.handler.read_pci_u32(device, to_config_offset(address)?)))
            }
            _ => Err(AmlError::NoHandlerForRegionAccess(region.space)),
        }
    }

    fn write(&self, token: &NativeRegionToken, offset: usize, length: usize, value: u64) -> Result<(), AmlError> {
        let region = token
            .downcast_ref::<OpRegion>()
            .ok_or_else(|| AmlError::HostError("TEST_REGION_TOKEN".to_string()))?;
        let address = checked_address(region, offset, length)?;
        match (region.space, length) {
            (RegionSpace::SystemMemory, 1) => self.handler.write_u8(address as usize, value as u8),
            (RegionSpace::SystemMemory, 2) => self.handler.write_u16(address as usize, value as u16),
            (RegionSpace::SystemMemory, 4) => self.handler.write_u32(address as usize, value as u32),
            (RegionSpace::SystemMemory, 8) => self.handler.write_u64(address as usize, value),
            (RegionSpace::SystemIO, 1) => self.handler.write_io_u8(to_port(address)?, value as u8),
            (RegionSpace::SystemIO, 2) => self.handler.write_io_u16(to_port(address)?, value as u16),
            (RegionSpace::SystemIO, 4) => self.handler.write_io_u32(to_port(address)?, value as u32),
            (RegionSpace::PciConfig, 1) => {
                let device = self.pci_device(region)?;
                self.handler.write_pci_u8(device, to_config_offset(address)?, value as u8);
            }
            (RegionSpace::PciConfig, 2) => {
                let device = self.pci_device(region)?;
                self.handler.write_pci_u16(device, to_config_offset(address)?, value as u16);
            }
            (RegionSpace::PciConfig, 4) => {
                let device = self.pci_device(region)?;
                self.handler.write_pci_u32(device, to_config_offset(address)?, value as u32);
            }
            _ => return Err(AmlError::NoHandlerForRegionAccess(region.space)),
        }
        Ok(())
    }
}

impl<H> TestNativeRegionHandler<H> {
    fn pci_device(&self, region: &OpRegion) -> Result<PciAddress, AmlError> {
        self.pci_devices
            .iter()
            .find(|(path, _)| path == &region.parent_device_path)
            .map(|(_, address)| *address)
            .ok_or(AmlError::NoHandlerForRegionAccess(RegionSpace::PciConfig))
    }
}

fn checked_address(region: &OpRegion, offset: usize, length: usize) -> Result<u64, AmlError> {
    let offset = u64::try_from(offset).map_err(|_| AmlError::HostError("TEST_REGION_RANGE".to_string()))?;
    let length = u64::try_from(length).map_err(|_| AmlError::HostError("TEST_REGION_RANGE".to_string()))?;
    let end = offset.checked_add(length).ok_or_else(|| AmlError::HostError("TEST_REGION_RANGE".to_string()))?;
    if length == 0 || end > region.length {
        return Err(AmlError::HostError("TEST_REGION_RANGE".to_string()));
    }
    region.base.checked_add(offset).ok_or_else(|| AmlError::HostError("TEST_REGION_RANGE".to_string()))
}

fn to_port(address: u64) -> Result<u16, AmlError> {
    u16::try_from(address).map_err(|_| AmlError::HostError("TEST_REGION_PORT".to_string()))
}

fn to_config_offset(address: u64) -> Result<u16, AmlError> {
    u16::try_from(address).map_err(|_| AmlError::HostError("TEST_REGION_PCI_OFFSET".to_string()))
}
