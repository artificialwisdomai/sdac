#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use cuda::cuda_client::CudaClient;
use futures::executor::block_on;
use std::borrow::BorrowMut;
use std::ffi::CString;
use std::sync::{Mutex, Once};
use tonic::transport::Channel;

pub type CUresult = ::std::os::raw::c_uint;
pub type cuuint32_t = u32;
pub type cuuint64_t = u64;
pub type CUdeviceptr_v2 = ::std::os::raw::c_ulonglong;
pub type CUdeviceptr = CUdeviceptr_v2;
pub type CUdevice_v1 = ::std::os::raw::c_int;
pub type CUdevice = CUdevice_v1;

pub mod cuda {
    tonic::include_proto!("cuda");
}

static mut CUDA_CLIENT: Option<Mutex<CudaClient<Channel>>> = None;
static INIT: Once = Once::new();

fn cudaClient<'a>() -> &'a Mutex<CudaClient<Channel>> {
    INIT.call_once(|| {
        // Since this access is inside a call_once, before any other accesses, it is safe
        unsafe {
            let client = block_on(CudaClient::connect("http://[::1]:50051")).unwrap();

            *CUDA_CLIENT.borrow_mut() = Some(Mutex::new(client));
        }
    });

    // As long as this function is the only place with access to the static variable,
    // giving out a read-only borrow here is safe because it is guaranteed no more mutable
    // references will exist at this point or in the future.
    unsafe { CUDA_CLIENT.as_ref().unwrap() }
}

// The CUDA C library function cuInit is replaced at runtime with this function.
// This function sends a request to the server, using the main tokio async thread,
// which is not currently implemented. After receiving the server response, a
// CUresult is returned to the C caller which is contained within the response.
#[no_mangle]
unsafe extern "C" fn cuInit(flags: ::std::os::raw::c_uint) -> CUresult {
    println!("cuInit({})", flags);

    let request = tonic::Request::new(cuda::CuInitRequest {
        flags: flags.into(),
    });

    let response = block_on(cudaClient().lock().unwrap().cu_init(request)).unwrap();

    println!("RESPONSE={:?}", response);

    response.into_inner().result
}

// The CUDA C library function cuGetDevice is replaced at runtime with this function.
// This function sends a request to the server, using the main tokio async thread,
// which is not currently implemented. After receiving the server response, a
// CUresult is returned to the C caller which is contained within the response.
// Additionally, the CUDA application will receive the device as a return.

#[no_mangle]
unsafe extern "C" fn cuDeviceGetName(
    name: *mut ::std::os::raw::c_char,
    len: ::std::os::raw::c_int,
    dev: CUdevice,
) -> CUresult {
    let request = tonic::Request::new(cuda::CuDeviceGetNameRequest {
        device: dev,
        max_len: len,
    });

    let response = block_on(cudaClient().lock().unwrap().cu_device_get_name(request)).unwrap();

    let resp = response.into_inner();
    println!("RESPONSE={:?}", resp);

    let cs = CString::new(resp.name).unwrap();
    libc::strcpy(name, cs.as_ptr());

    resp.result
}

#[no_mangle]
unsafe extern "C" fn cuDeviceGetCount(count: *mut ::std::os::raw::c_int) -> CUresult {
    let request = tonic::Request::new(cuda::CuDeviceGetCountRequest {});

    let response = block_on(cudaClient().lock().unwrap().cu_device_get_count(request)).unwrap();

    let resp = response.into_inner();
    println!("RESPONSE={:?}", resp);

    *count = resp.count;

    resp.result
}

#[no_mangle]
unsafe extern "C" fn cuDeviceGet(
    device: *mut CUdevice,
    ordinal: ::std::os::raw::c_int,
) -> CUresult {
    let request = tonic::Request::new(cuda::CuDeviceGetRequest { ordinal: ordinal });

    let response = block_on(cudaClient().lock().unwrap().cu_device_get(request)).unwrap();

    let resp = response.into_inner();
    println!("RESPONSE={:?}", resp);

    *device = resp.device;

    resp.result
}
