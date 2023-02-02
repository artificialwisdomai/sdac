#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/cuda_types.rs"));

use ::std::os::raw::*;

/// This is the service definition. It looks a lot like a trait definition.
// TODO(asalkeld) auto generate this interface. https://github.com/xertai/sdac/issues/40
#[tarpc::service]
pub trait Cuda {
    // CUDA Driver API
    async fn cuGetErrorName(error: CUresult) -> (String, CUresult);
    async fn cuGetErrorString(error: CUresult) -> (String, CUresult);
    async fn cuInit(flags: u32) -> CUresult;
    async fn cuDeviceGet(ordinal: i32) -> (i32, CUresult);
    async fn cuDeviceGetCount() -> (i32, CUresult);
    async fn cuDeviceGetName(maxLen: i32, dev: CUdevice) -> (String, CUresult);
    async fn cuDeviceTotalMem_v2(dev: CUdevice) -> (usize, CUresult); 

    async fn cuMemAlloc_v2(size: usize) -> (CUdeviceptr, CUresult);
    async fn cuMemFree_v2(devPtr: CUdeviceptr) -> CUresult;

    async fn cuMemcpyDtoH_v2(src: CUdeviceptr, size: usize) -> (Vec::<u8>, CUresult);
    async fn cuMemcpyHtoD_v2(dst: CUdeviceptr, data: Vec::<u8>,size: usize) -> CUresult;

    // Runtime API
    async fn cudaGetLastError() -> cudaError_t;
    async fn cudaPeekAtLastError() -> cudaError_t;
    async fn cudaGetErrorName(error: cudaError_t) -> String;
    async fn cudaGetErrorString(error: cudaError_t) -> String;

    async fn cudaGetDeviceCount() -> (i32, cudaError_t);
    async fn cudaSetDevice(device: c_int) -> cudaError_t;
    async fn cudaDeviceGetAttribute(attr: cudaDeviceAttr, device: c_int) -> (c_int,cudaError_t);
    
    async fn cudaMalloc(size: usize) -> (usize,cudaError_t);
    async fn cudaFree(devPtr: usize) -> cudaError_t;

    async fn cudaMemcpyDtoH(dst: usize, src: usize, count: usize) -> (Vec::<u8>, CUresult);
    async fn cudaMemcpyHtoD(dst: usize, data: Vec::<u8>,size: usize) -> CUresult;
    async fn cudaMemcpyDtoD(dst: usize, src: usize,size: usize) -> CUresult;

    async fn cudaMemset(devPtr: usize, value: c_int, count: usize) -> cudaError_t;

    async fn cuModuleLoadData(data: Vec::<u8>) -> (usize, cudaError_t);
    async fn cuModuleLoadFatBinary(data: Vec::<u8>) -> (usize, cudaError_t);
    async fn cuModuleUnload(hmod: usize) -> cudaError_t; 
}
