#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use service::*;
use futures::executor::block_on;
use std::mem::size_of;
use std::ffi::CString;
use std::sync::{Mutex};
use tarpc::{context};

pub fn cuGetErrorString(
    client: &Mutex<service::CudaClient>,
    error: CUresult,
    pStr: *mut ::std::os::raw::c_char,
) -> CUresult {
    let (strName, res) = block_on(
            client
            .lock()
            .unwrap()
            .cuGetErrorString(context::current(), error),
    )
    .unwrap();

    if res != cudaError_enum_CUDA_SUCCESS {
        return res;
    }

    let cs = CString::new(strName).unwrap();
    unsafe {
        libc::strcpy(pStr, cs.as_ptr());
    }

    res
}

pub fn cuGetErrorName(
    client: &Mutex<service::CudaClient>,
    error: CUresult,
    pStr: *mut ::std::os::raw::c_char,
) -> CUresult {
    let (strName, res) = block_on(
        client
            .lock()
            .unwrap()
            .cuGetErrorName(context::current(), error),
    )
    .unwrap();

    if res != cudaError_enum_CUDA_SUCCESS {
        return res;
    }

    let cs = CString::new(strName).unwrap();
    unsafe {
        libc::strcpy(pStr, cs.as_ptr());
    }

    res
}

pub fn cuInit(client: &Mutex<service::CudaClient>,flags: ::std::os::raw::c_uint) -> CUresult {
    block_on(
       client
            .lock()
            .unwrap()
            .cuInit(context::current(), flags),
    )
    .unwrap()
}

pub fn cuDeviceGetName(client: &Mutex<service::CudaClient>,
    name: *mut ::std::os::raw::c_char,
    len: ::std::os::raw::c_int,
    dev: CUdevice,
) -> CUresult {
    let (strName, res) = block_on(client.lock().unwrap().cuDeviceGetName(
        context::current(),
        len,
        dev,
    ))
    .unwrap();

    let cs = CString::new(strName).unwrap();
    unsafe {
        libc::strcpy(name, cs.as_ptr());
    }

    res
}

pub fn cuDeviceGetCount(client: &Mutex<service::CudaClient>,count: *mut ::std::os::raw::c_int) -> CUresult {
    let (cnt, res) = block_on(
        client
            .lock()
            .unwrap()
            .cuDeviceGetCount(context::current()),
    )
    .unwrap();

    unsafe {
        *count = cnt;
    }

    res
}

pub fn cuDeviceGet(client: &Mutex<service::CudaClient>,
    device: *mut CUdevice,
    ordinal: ::std::os::raw::c_int,
) -> CUresult {
    let (dev, res) = block_on(
        client
            .lock()
            .unwrap()
            .cuDeviceGet(context::current(), ordinal),
    )
    .unwrap();

    unsafe {
        *device = dev;
    }

    res
}

pub fn cuMemAlloc_v2(client: &Mutex<service::CudaClient>,
    dptr: *mut CUdeviceptr,
    bytesize: ::std::os::raw::c_ulonglong,
) -> CUresult {
    let (ptr, res) = block_on(
        client
            .lock()
            .unwrap()
            .cuMemAlloc_v2(context::current(), bytesize as usize),
    )
    .unwrap();

    unsafe {
        *dptr = ptr;
    }

    res
}

pub fn cuMemcpyDtoH_v2(client: &Mutex<service::CudaClient>,
    dstHost: *mut ::std::os::raw::c_void,
    srcDevice: CUdeviceptr,
    ByteCount: ::std::os::raw::c_ulonglong,
) -> CUresult {
let (data, res) =   block_on(
        client
            .lock()
            .unwrap()
            .cuMemcpyDtoH_v2(context::current(), srcDevice, ByteCount as usize),
    )
    .unwrap();

    if res!= cudaError_enum_CUDA_SUCCESS {
        return res;
    }

    unsafe {
        libc::memcpy(dstHost, data.as_ptr() as *const libc::c_void, ByteCount as usize);
    }

    0
}

pub fn cuMemcpyHtoD_v2(client: &Mutex<service::CudaClient>,
    dstDevice: CUdeviceptr,
    srcHost: *const ::std::os::raw::c_void,
    ByteCount: ::std::os::raw::c_ulonglong,
) -> CUresult {
    let data = unsafe{ Vec::<u8>::from_raw_parts(srcHost, ByteCount, size_of(u8))};
    block_on(
        client
            .lock()
            .unwrap()
            .cuMemcpyHtoD_v2(context::current(), dstDevice, data, ByteCount as usize),
    )
    .unwrap()
}

pub fn cuMemFree_v2(client: &Mutex<service::CudaClient>,dptr: CUdeviceptr) -> CUresult {
    block_on(
        client
            .lock()
            .unwrap()
            .cuMemFree_v2(context::current(), dptr),
    )
    .unwrap()
}

pub fn cuDeviceTotalMem_v2(
    client: &Mutex<service::CudaClient>,
    bytes: *mut usize,
    dev: CUdevice,
) -> CUresult{
    let (cnt, res) = block_on(
        client
        .lock()
            .unwrap()
            .cuDeviceTotalMem_v2(context::current(), dev),
    )
    .unwrap();

    unsafe {
        *bytes = cnt;
    }

    res
}