/// This is the service definition. It looks a lot like a trait definition.
// TODO(asalkeld) auto generate this interface. https://github.com/xertai/sdac/issues/40
#[tarpc::service]
pub trait Cuda {
    async fn cuInit(flags: u32) -> u32;
    async fn cuDeviceGet(ordinal: i32) -> (i32, u32);
    async fn cuDeviceGetCount() -> (i32, u32);
    async fn cuDeviceGetName(maxLen: i32, dev: i32) -> (String, u32);
}
