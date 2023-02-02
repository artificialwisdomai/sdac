#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use ::std::os::raw::*;
use std::fs;
use clib::memcpy;
use futures::executor::block_on;
use service::*;
use std::ffi::CString;
use std::sync::Mutex;
use tarpc::context;

/// Allocate memory on the device
/// Allocates \\p size bytes of linear memory on the device and returns in
/// *devPtr a pointer to the allocated memory. The allocated memory is
/// suitably aligned for any kind of variable. The memory is not cleared.
/// ::cudaMalloc() returns ::cudaErrorMemoryAllocation in case of failure.
///
/// The device version of ::cudaFree cannot be used with a \\p *devPtr
/// allocated using the host API, and vice versa.
/// \\param devPtr - Pointer to allocated device memory
/// \\param size   - Requested allocation size in bytes\n\n \\return
/// ::cudaSuccess,\n ::cudaErrorInvalidValue,\n ::cudaErrorMemoryAllocation\n \\notefnerr\n \\note_init_rt\n \\note_callback
///
/// \\sa ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
/// ::cudaMalloc3D, ::cudaMalloc3DArray,\n \\ref ::cudaMallocHost(void**, size_t) \"cudaMallocHost (C API)\",
/// ::cudaFreeHost, ::cudaHostAlloc,\n ::cuMemAlloc"]
pub fn cudaMalloc(
    client: &Mutex<service::CudaClient>,
    devPtr: *mut *mut c_void,
    size: usize,
) -> cudaError_t {
    let (remotePtr, res) =
        block_on(client.lock().unwrap().cudaMalloc(context::current(), size)).unwrap();

    if res != 0 {
        return res;
    }

    unsafe {
        *devPtr = remotePtr as *mut c_void;
    }

    res
}

pub fn cudaFree(
    client: &Mutex<service::CudaClient>,
    devPtr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    block_on(
        client
            .lock()
            .unwrap()
            .cudaFree(context::current(), devPtr as usize),
    )
    .unwrap()
}

pub fn cudaMemcpy(
    client: &Mutex<service::CudaClient>,
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let c = client.lock().unwrap();

    match kind {
        cudaMemcpyKind_cudaMemcpyHostToHost => {
            memcpy(dst as usize, src as usize, count)
        },

        cudaMemcpyKind_cudaMemcpyHostToDevice => {
            let data = unsafe { Vec::<u8>::from_raw_parts(src as *mut u8, count, count) };
            block_on(c.cudaMemcpyHtoD(context::current(), dst as usize, data, count))
                .unwrap()
        }

        cudaMemcpyKind_cudaMemcpyDeviceToHost => {
            let (data, res) =
                block_on(c.cudaMemcpyDtoH(context::current(), dst as usize, src as usize, count))
                    .unwrap();
            if res != 0 {
                return res;
            }

            unsafe {
                memcpy(dst as usize, data.as_ptr() as usize, count);
            }

            res
        }

        cudaMemcpyKind_cudaMemcpyDeviceToDevice => {
            block_on(c.cudaMemcpyDtoD(context::current(), dst as usize, src as usize, count))
                .unwrap()
        }

        cudaMemcpyKind_cudaMemcpyDefault => {
            // "< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing"]
            // TODO(asalkeld) not sure how to figure out which direction this is..
            cudaError_enum_CUDA_ERROR_STUB_LIBRARY
        }
    }
}

pub fn cudaMemset(
    client: &Mutex<service::CudaClient>,
    devPtr: *mut ::std::os::raw::c_void,
    value: ::std::os::raw::c_int,
    count: usize,
) -> cudaError_t {
    block_on(
        client
            .lock()
            .unwrap()
            .cudaMemset(context::current(), devPtr as usize, value, count),
    )
    .unwrap()
}

pub fn cudaGetLastError(client: &Mutex<service::CudaClient>) -> cudaError_t {
    block_on(client.lock().unwrap().cudaGetLastError(context::current())).unwrap()
}

pub fn cudaPeekAtLastError(client: &Mutex<service::CudaClient>) -> cudaError_t {
    block_on(
        client
            .lock()
            .unwrap()
            .cudaPeekAtLastError(context::current()),
    )
    .unwrap()
}

pub fn cudaGetErrorName(
    client: &Mutex<service::CudaClient>,
    error: cudaError_t,
) -> *const ::std::os::raw::c_char {
    let c_str = block_on(
        client
            .lock()
            .unwrap()
            .cudaGetErrorName(context::current(), error),
    )
    .unwrap();

    CString::new(c_str).unwrap().into_raw()
}

pub fn cudaGetErrorString(
    client: &Mutex<service::CudaClient>,
    error: cudaError_t,
) -> *const ::std::os::raw::c_char {
    let c_str = block_on(
        client
            .lock()
            .unwrap()
            .cudaGetErrorString(context::current(), error),
    )
    .unwrap();

    CString::new(c_str).unwrap().into_raw()
}

pub fn cudaGetDeviceCount(
    client: &Mutex<service::CudaClient>,
    count: *mut ::std::os::raw::c_int,
) -> cudaError_t {
    let (cnt, res) = block_on(
        client
            .lock()
            .unwrap()
            .cudaGetDeviceCount(context::current()),
    )
    .unwrap();

    if res != 0 {
        return res;
    }

    unsafe {
        *count = cnt;
    }

    res
}

pub fn cudaSetDevice(
    client: &Mutex<service::CudaClient>,
    device: ::std::os::raw::c_int,
) -> cudaError_t {
    block_on(
        client
            .lock()
            .unwrap()
            .cudaSetDevice(context::current(), device),
    )
    .unwrap()
}

pub fn cudaDeviceGetAttribute(
    client: &Mutex<service::CudaClient>,
    value: *mut ::std::os::raw::c_int,
    attr: cudaDeviceAttr,
    device: ::std::os::raw::c_int,
) -> cudaError_t {
    let (v, res) = block_on(client.lock().unwrap().cudaDeviceGetAttribute(
        context::current(),
        attr,
        device,
    ))
    .unwrap();

    if res != 0 {
        return res;
    }

    unsafe {
        *value = v;
    }

    res
}

pub fn cuModuleLoad(
    client: &Mutex<service::CudaClient>,
    module: *mut CUmodule,
    fname: *const ::std::os::raw::c_char,
) -> cudaError_t {
    let contents = fs::read_to_string(fname).expect("Should have been able to read the file");

    let (remotePtr, res) = block_on(
        client
            .lock()
            .unwrap()
            .cuModuleLoadData(context::current(), contents.into_bytes()),
    )
    .unwrap();

    if res != 0 {
        return res;
    }

    unsafe {
        *module = remotePtr as *mut cuModule_t;
    }

    res
}

/// Load a module's data
/// Takes a pointer \\p image and loads the corresponding module  into the current context.
/// The pointer may be obtained by mapping a cubin or PTX or fatbin file,
/// passing a cubin or PTX or fatbin file as a NULL-terminated text string
/// or incorporating a cubin or fatbin object into the executable resources and 
/// using operating system calls such as Windows FindResource() to obtain the pointer.
/// 
/// module - Returned module
/// image  - Module data to load
pub fn cuModuleLoadData(
    client: &Mutex<service::CudaClient>,
    module: *mut CUmodule,
    image: *const ::std::os::raw::c_void,
) -> cudaError_t {
    let str = unsafe{CString::from_raw(image as *mut i8)};

    let (remotePtr, res) = block_on(
        client
            .lock()
            .unwrap()
            .cuModuleLoadData(context::current(), str.into_bytes()),
    )
    .unwrap();

    if res != 0 {
        return res;
    }

    unsafe {
        *module = remotePtr as CUmodule;
    }

    res
}

pub fn cuModuleLoadFatBinary(
    client: &Mutex<service::CudaClient>,
    module: *mut CUmodule,
    fatCubin: *const ::std::os::raw::c_void,
) -> cudaError_t {
    let str = unsafe{CString::from_raw(fatCubin as *mut i8)};

    let (remotePtr, res) = block_on(
        client
            .lock()
            .unwrap()
            .cuModuleLoadFatBinary(context::current(), str.into_bytes()),
    )
    .unwrap();

    if res != 0 {
        return res;
    }

    unsafe {
        *module = remotePtr as CUmodule;
    }

    res
}

pub fn cuModuleUnload(client: &Mutex<service::CudaClient>, module: CUmodule) -> cudaError_t {
    block_on(
        client
            .lock()
            .unwrap()
            .cuModuleUnload(context::current(), module as usize),
    )
    .unwrap()
}
