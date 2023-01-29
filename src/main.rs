#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/cuda_driver_bindings.rs"));

mod service;
use futures_util::StreamExt;
use std::ffi::CString;
use anyhow;
use tarpc::{
    context,    server::{BaseChannel, Channel},
    tokio_serde::formats::Json,
};
use crate::service::Cuda as CudaService;

// This is the type that implements the generated World trait. It is the business logic
// and is used to start the server.
#[derive(Clone)]
struct RemoteCuda;

#[tarpc::server]
impl service::Cuda for RemoteCuda {
    async fn cuInit(self, _: context::Context, flags: u32) -> u32 {
        // TODO(asalkeld) This does not look right. cuInit should be called once per process or once for the server.
        // https://github.com/xertai/sdac/issues/38
        unsafe { cuInit(flags) }
    }
    async fn cuDeviceGet(self, _: context::Context, ordinal: i32) -> (i32, u32) {
        let mut dev: CUdevice = 0;
        let devPtr: *mut CUdevice = &mut dev;

        unsafe {
            let res = cuDeviceGet(devPtr, ordinal);
            (*devPtr, res)
        }
    }
    async fn cuDeviceGetCount(self, _: context::Context) -> (i32, u32) {
        let mut count: i32 = 0;
        let countPtr: *mut i32 = &mut count;

        unsafe {
            let res = cuDeviceGetCount(countPtr);

            (*countPtr, res)
        }
    }
    async fn cuDeviceGetName(self, _: context::Context, max_len: i32, dev: i32) -> (String, u32) {
        let mut name = [0; 1024];
        let mut len = max_len;
        if len > 1024 {
            len = 1024
        }

        unsafe {
            let res = cuDeviceGetName(name.as_mut_ptr(), len, dev);

            let strName = CString::from_raw(name.as_mut_ptr())
                .into_string()
                .expect("failed to convert name");

            (strName, res)
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // JSON transport is provided by the json_transport tarpc module. It makes it easy
    // to start up a serde-powered json serialization strategy over TCP.
    // TODO(asalkeld) we should bind to a particular interface.
    // https://github.com/xertai/sdac/issues/39
    let listener = tarpc::serde_transport::tcp::listen("[::1]:50055", Json::default).await?;

    println!("Sdac listening on `{}`", listener.local_addr());
    tokio::spawn(async move {
        listener
            .filter_map(|r| async move { r.ok() })
            .map(BaseChannel::with_defaults)
            .map(|channel| channel.execute(RemoteCuda.serve()))
            .buffer_unordered(10)
            .for_each(|()| async {})
            .await;
    });

    Ok(())
}
