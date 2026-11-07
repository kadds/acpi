#[cfg(not(target_os = "naos"))]
use alloc::alloc::Global;

#[cfg(target_os = "naos")]
pub type Allocator = naos_runtime::HeapQuotaAllocator;
#[cfg(not(target_os = "naos"))]
pub type Allocator = Global;

pub type Vec<T> = alloc::vec::Vec<T, Allocator>;
pub type Arc<T> = alloc::sync::Arc<T, Allocator>;
pub type Box<T> = alloc::boxed::Box<T, Allocator>;
pub type BTreeMap<K, V> = alloc::collections::btree_map::BTreeMap<K, V, Allocator>;

fn allocator() -> Allocator {
    #[cfg(target_os = "naos")]
    {
        Allocator::current()
    }
    #[cfg(not(target_os = "naos"))]
    {
        Global
    }
}
fn unmetered_allocator() -> Allocator {
    #[cfg(target_os = "naos")]
    {
        Allocator::default()
    }
    #[cfg(not(target_os = "naos"))]
    {
        Global
    }
}

pub fn vec_new<T>() -> Vec<T> {
    Vec::new_in(allocator())
}

pub fn vec_new_unmetered<T>() -> Vec<T> {
    Vec::new_in(unmetered_allocator())
}

pub fn vec_with_capacity<T>(capacity: usize) -> Vec<T> {
    Vec::with_capacity_in(capacity, allocator())
}

pub fn vec_from_slice<T: Clone>(source: impl AsRef<[T]>) -> Vec<T> {
    let source = source.as_ref();
    let mut result = vec_with_capacity(source.len());
    result.extend_from_slice(source);
    result
}

pub fn vec_from_array<T, const N: usize>(source: [T; N]) -> Vec<T> {
    let mut result = vec_with_capacity(N);
    result.extend(source);
    result
}

pub fn try_string_from_str(source: &str) -> Result<String, alloc::collections::TryReserveError> {
    let mut result = String::new();
    result.try_reserve_exact(source.len())?;
    result.push_str(source);
    Ok(result)
}

pub fn arc_new<T>(value: T) -> Arc<T> {
    Arc::new_in(value, allocator())
}

pub fn arc_new_unmetered<T>(value: T) -> Arc<T> {
    Arc::new_in(value, unmetered_allocator())
}

pub fn box_new<T>(value: T) -> Box<T> {
    Box::new_in(value, allocator())
}

pub fn btree_map_new<K, V>() -> BTreeMap<K, V> {
    BTreeMap::new_in(allocator())
}

pub fn string_new_unmetered() -> String {
    #[cfg(target_os = "naos")]
    {
        String(Vec::new_in(unmetered_allocator()))
    }
    #[cfg(not(target_os = "naos"))]
    {
        String::new()
    }
}

pub fn replace_string_bytes(
    target: &mut String,
    bytes: Vec<u8>,
) -> Result<(), core::str::Utf8Error> {
    #[cfg(target_os = "naos")]
    {
        target.replace_bytes(bytes)
    }
    #[cfg(not(target_os = "naos"))]
    {
        *target = String::from_utf8(bytes).map_err(|error| error.utf8_error())?;
        Ok(())
    }
}

#[cfg(not(target_os = "naos"))]
pub type String = alloc::string::String;

#[cfg(target_os = "naos")]
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct String(Vec<u8>);

#[cfg(target_os = "naos")]
impl String {
    pub fn new() -> Self {
        Self(vec_new())
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), alloc::collections::TryReserveError> {
        self.0.try_reserve(additional)
    }

    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), alloc::collections::TryReserveError> {
        self.0.try_reserve_exact(additional)
    }

    pub fn push_str(&mut self, value: &str) {
        self.0.extend_from_slice(value.as_bytes());
    }

    pub fn push(&mut self, value: char) {
        let mut encoded = [0; 4];
        self.push_str(value.encode_utf8(&mut encoded));
    }

    pub fn truncate(&mut self, new_len: usize) {
        if new_len <= self.len() && self.as_str().is_char_boundary(new_len) {
            self.0.truncate(new_len);
        }
    }

    pub fn as_str(&self) -> &str {
        // SAFETY: all mutation methods append valid UTF-8 and the byte vector
        // is private, so the invariant is maintained for the string's life.
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn replace_bytes(&mut self, bytes: Vec<u8>) -> Result<(), core::str::Utf8Error> {
        core::str::from_utf8(&bytes)?;
        self.0 = bytes;
        Ok(())
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

#[cfg(target_os = "naos")]
impl Default for String {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "naos")]
impl core::ops::Deref for String {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

#[cfg(target_os = "naos")]
impl core::ops::DerefMut for String {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: `str`'s mutable methods preserve UTF-8 validity.
        unsafe { core::str::from_utf8_unchecked_mut(&mut self.0) }
    }
}

#[cfg(target_os = "naos")]
impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[cfg(target_os = "naos")]
impl core::ops::Add<&str> for String {
    type Output = Self;

    fn add(mut self, value: &str) -> Self::Output {
        self.push_str(value);
        self
    }
}

#[cfg(target_os = "naos")]
impl core::fmt::Write for String {
    fn write_str(&mut self, value: &str) -> core::fmt::Result {
        self.push_str(value);
        Ok(())
    }
}

#[cfg(target_os = "naos")]
impl core::fmt::Display for String {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(target_os = "naos")]
impl core::fmt::Debug for String {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_str(), formatter)
    }
}

#[cfg(target_os = "naos")]
impl PartialEq<str> for String {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

#[cfg(target_os = "naos")]
impl PartialEq<&str> for String {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[macro_export]
macro_rules! aml_vec {
    () => {
        $crate::aml::allocator::vec_new()
    };
    ($element:expr; $count:expr) => {{
        let mut result = $crate::aml::allocator::vec_new();
        result.resize($count, $element);
        result
    }};
    ($($element:expr),+ $(,)?) => {{
        let mut result = $crate::aml::allocator::vec_new();
        result.extend([$($element),+]);
        result
    }};
}
