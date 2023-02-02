#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use service::*;

mod device;
mod global;
mod runtime;

// CUDA Driver API
#[no_mangle]
unsafe extern "C" fn cuInit(flags: ::std::os::raw::c_uint) -> CUresult {
    device::cuInit(global::client(), flags)
}

#[no_mangle]
unsafe extern "C" fn cuGetErrorString(
    error: CUresult,
    pStr: *mut ::std::os::raw::c_char,
) -> CUresult {
    device::cuGetErrorString(global::client(), error, pStr)
}

#[no_mangle]
unsafe extern "C" fn cuGetErrorName(
    error: CUresult,
    pStr: *mut ::std::os::raw::c_char,
) -> CUresult {
    device::cuGetErrorName(global::client(), error, pStr)
}

#[no_mangle]
unsafe extern "C" fn cuDeviceGetName(
    name: *mut ::std::os::raw::c_char,
    len: ::std::os::raw::c_int,
    dev: CUdevice,
) -> CUresult {
    device::cuDeviceGetName(global::client(), name, len, dev)
}

#[no_mangle]
unsafe extern "C" fn cuDeviceGetCount(count: *mut ::std::os::raw::c_int) -> CUresult {
    device::cuDeviceGetCount(global::client(), count)
}

#[no_mangle]
unsafe extern "C" fn cuDeviceGet(
    device: *mut CUdevice,
    ordinal: ::std::os::raw::c_int,
) -> CUresult {
    device::cuDeviceGet(global::client(), device, ordinal)
}

#[no_mangle]
unsafe extern "C" fn cuDeviceTotalMem_v2(
    bytes: *mut usize,
    dev: CUdevice,
) -> CUresult {
    device::cuDeviceTotalMem_v2(global::client(), bytes, dev)
}

// ======================
// Runtime API

#[no_mangle]
unsafe extern "C" fn cudaMalloc(
    devPtr: *mut *mut ::std::os::raw::c_void,
    size: usize,
) -> cudaError_t {
    runtime::cudaMalloc(global::client(), devPtr, size)
}

#[no_mangle]
unsafe extern "C" fn cudaFree(devPtr: *mut ::std::os::raw::c_void) -> cudaError_t {
    runtime::cudaFree(global::client(), devPtr)
}

#[no_mangle]
unsafe extern "C" fn cudaMemcpy(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    runtime::cudaMemcpy(global::client(), dst, src, count, kind)
}

#[no_mangle]
unsafe extern "C" fn cudaMemset(
    devPtr: *mut ::std::os::raw::c_void,
    value: ::std::os::raw::c_int,
    count: usize,
) -> cudaError_t {
    runtime::cudaMemset(global::client(), devPtr, value, count)
}

#[no_mangle]
unsafe extern "C" fn cudaGetLastError() -> cudaError_t {
    runtime::cudaGetLastError(global::client())
}

#[no_mangle]
unsafe extern "C" fn cudaPeekAtLastError() -> cudaError_t {
    runtime::cudaPeekAtLastError(global::client())
}

#[no_mangle]
unsafe extern "C" fn cudaGetErrorName(error: cudaError_t) -> *const ::std::os::raw::c_char {
    runtime::cudaGetErrorName(global::client(), error)
}

#[no_mangle]
unsafe extern "C" fn cudaGetErrorString(error: cudaError_t) -> *const ::std::os::raw::c_char {
    runtime::cudaGetErrorString(global::client(), error)
}

#[no_mangle]
unsafe extern "C" fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t {
    runtime::cudaGetDeviceCount(global::client(), count)
}

#[no_mangle]
unsafe extern "C" fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t {
    runtime::cudaSetDevice(global::client(), device)
}

#[no_mangle]
unsafe extern "C" fn cudaDeviceGetAttribute(
    value: *mut ::std::os::raw::c_int,
    attr: cudaDeviceAttr,
    device: ::std::os::raw::c_int,
) -> cudaError_t {
    runtime::cudaDeviceGetAttribute(global::client(), value, attr, device)
}

#[no_mangle]
unsafe extern "C" fn cuModuleLoad(
    module: *mut CUmodule,
    fname: *const ::std::os::raw::c_char,
) -> cudaError_t {
    runtime::cuModuleLoad(global::client(), module, fname)
}

#[no_mangle]
unsafe extern "C" fn cuModuleLoadData(
    module: *mut CUmodule,
    image: *const ::std::os::raw::c_void,
) -> cudaError_t {
    runtime::cuModuleLoadData(global::client(), module, image)
}

#[no_mangle]
unsafe extern "C" fn cuModuleLoadFatBinary(
    module: *mut CUmodule,
    fatCubin: *const ::std::os::raw::c_void,
) -> cudaError_t {
    runtime::cuModuleLoadFatBinary(global::client(), module, fatCubin)
}

#[no_mangle]
unsafe extern "C" fn cuModuleUnload(module: CUmodule) -> cudaError_t {
    runtime::cuModuleUnload(global::client(), module)
}
