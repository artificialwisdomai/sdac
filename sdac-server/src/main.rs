#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/cuda_driver_bindings.rs"));

use futures_util::StreamExt;
use service::Cuda as CudaService;
use std::ffi::CString;
use tarpc::{
    context,
    server::{BaseChannel, Channel},
    tokio_serde::formats::Json,
};

// This is the type that implements the generated World trait. It is the business logic
// and is used to start the server.
#[derive(Clone)]
struct RemoteCuda;

#[tarpc::server]
impl service::Cuda for RemoteCuda {
    async fn cuGetErrorName(self, _: context::Context, error: u32) -> (String, u32) {
        let mut nameV: i8 = 1;
        let name: *mut i8 = &mut nameV;

        unsafe {
            let res = cuGetErrorName(error, name as *mut *const i8);
            if res != cudaError_enum_CUDA_SUCCESS {
                return (String::new(), res);
            }

            let str = CString::from_raw(name)
                .into_string()
                .expect("failed to convert name");

            (str, res)
        }
    }
    async fn cuGetErrorString(self, _: context::Context, error: u32) -> (String, u32) {
        let mut descrV: i8 = 1;
        let descr: *mut i8 = &mut descrV;

        unsafe {
            let res = cuGetErrorString(error, descr as *mut *const i8);
            if res != cudaError_enum_CUDA_SUCCESS {
                return (String::new(), res);
            }

            let str = CString::from_raw(descr)
                .into_string()
                .expect("failed to convert name");

            (str, res)
        }
    }
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
    async fn cuDeviceTotalMem_v2(dev: CUdevice) -> (usize, CUresult) {
        let mut bytes: usize = 0;
        unsafe {
            let res = cuDeviceTotalMem_v2(&bytes, dev);

            (bytes, res)
        }
    }
    async fn cuDeviceGetAttribute(self, _: context::Context, attrib: u32, dev: i32) -> (i32, u32) {
        let mut value: i32 = 0;

        unsafe {
            let res = cuDeviceGetAttribute(&mut value, attrib, dev);

            (value, res)
        }
    }
    async fn cuModuleLoadData(self, _: context::Context, image: Vec::<u8>) -> (i32, u32) {
        let mut module: CUmodule = 0;
        let modulePtr: *mut CUmodule = &mut module;

        unsafe {
            let res = cuModuleLoadData(modulePtr, image.as_ptr());

            (*modulePtr, res)
        }
    }
    async fn cuModuleLoadFatBinary(self, _: context::Context, fatCubin: Vec::<u8>) -> (i32, u32) {
        let mut module: CUmodule = 0;
        let modulePtr: *mut CUmodule = &mut module;

        unsafe {
            let res = cuModuleLoadFatBinary(modulePtr, fatCubin.as_ptr());

            (*modulePtr, res)
        }
    }
    async fn cudaGetLastError(self, _: context::Context) -> u32 {
        unsafe { cudaGetLastError() }
    }
    async fn cudaPeekAtLastError(self, _: context::Context) -> u32 {
        unsafe { cudaPeekAtLastError() }
    }
    async fn cudaGetErrorName(self, _: context::Context, error: cudaError_t) -> String{
        unsafe { cudaGetErrorName(error)
    }
    async fn cudaGetErrorString(self, _: context::Context, error: cudaError_t) -> String{
        unsafe { cudaGetErrorString(error)
    }
    async fn cudaMalloc(self, _: context::Context, size: usize) -> (usize, cudaError_t) {
        unsafe { cudaMalloc(devPtr as *mut *mut std::os::raw::c_void, size) }
    }
    async fn cudaFree(self, _: context::Context, devPtr: usize) -> u32 {
        unsafe { cudaFree(devPtr as *mut std::os::raw::c_void) }
    }
    async fn cudaMemcpyDtoH(dst: usize, src: usize, count: usize) -> (Vec::<u8>, CUresult){
        let mut data = vec![0; count];
        unsafe {
            let res = cudaMemcpyDtoH(data.as_mut_ptr(), src, count);
            (data, res)
        }
    }
    async fn cudaMemcpyHtoD(dst: usize, data: Vec::<u8>,size: usize) -> CUresult{
        unsafe { cudaMemcpyHtoD(dst, data.as_ptr(), size)
    }
    async fn cudaMemcpyDtoD(dst: usize, src: usize,size: usize) -> CUresult{
        unsafe { cudaMemcpyDtoD(dst, src, size)
    }
    async fn cudaMemset(devPtr: usize, value: c_int, count: usize) -> cudaError_t{
        unsafe { cudaMemset(devPtr, value, count)
    }
    async fn cudaGetDeviceCount() -> (i32, cudaError_t){
        let mut count: i32 = 0;
        unsafe { cudaGetDeviceCount(&mut count) }
    }
    async fn cudaSetDevice(device: c_int) -> cudaError_t{
        unsafe { cudaSetDevice(device) }
    }
    async fn cudaDeviceGetAttribute(attr: cudaDeviceAttr, device: c_int) -> (c_int,cudaError_t){
        let mut value: c_int = 0;
        unsafe { cudaDeviceGetAttribute(&mut value, attr, device) }
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
