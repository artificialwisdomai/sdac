#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/cuda_driver_bindings.rs"));

use cuda::cuda_server::{Cuda, CudaServer};
use std::ffi::CString;
use tonic::{transport::Server, Request, Response, Status};

pub mod cuda {
    tonic::include_proto!("cuda");
}

#[derive(Debug, Default)]
pub struct MyCuda {}

#[tonic::async_trait]
impl Cuda for MyCuda {
    async fn cu_init(
        &self,
        request: Request<cuda::CuInitRequest>,
    ) -> Result<Response<cuda::CuInitResponse>, Status> {
        println!("Got a request: {:?}", request);

        // This does not look right. cuInit should be called once per process.
        // I suspect what we need to do is:
        // 1. new connection
        // 2. fork
        // 3. forked server handles all requests for that connection...
        unsafe {
            let res = cuInit(request.into_inner().flags);

            let reply = cuda::CuInitResponse { result: res };

            Ok(tonic::Response::new(reply))
        }
    }
    async fn cu_device_get(
        &self,
        request: Request<cuda::CuDeviceGetRequest>,
    ) -> Result<Response<cuda::CuDeviceGetResponse>, Status> {
        println!("Got a request: {:?}", request);

        let mut dev: CUdevice = 0;
        let devPtr: *mut CUdevice = &mut dev;

        unsafe {
            let res = cuDeviceGet(devPtr, request.into_inner().ordinal);
            let reply = cuda::CuDeviceGetResponse {
                result: res.into(),
                device: *devPtr,
            };
            Ok(Response::new(reply))
        }
    }
    async fn cu_device_get_count(
        &self,
        request: Request<cuda::CuDeviceGetCountRequest>,
    ) -> Result<Response<cuda::CuDeviceGetCountResponse>, Status> {
        println!("Got a request: {:?}", request);

        let mut count: i32 = 0;
        let countPtr: *mut i32 = &mut count;

        unsafe {
            let res = cuDeviceGetCount(countPtr);
            let reply = cuda::CuDeviceGetCountResponse {
                result: res.into(),
                count: *countPtr,
            };

            Ok(Response::new(reply))
        }
    }
    async fn cu_device_get_name(
        &self,
        request: Request<cuda::CuDeviceGetNameRequest>,
    ) -> Result<Response<cuda::CuDeviceGetNameResponse>, Status> {
        println!("Got a request: {:?}", request);

        let req = request.into_inner();

        let mut name = [0; 1024];
        let mut len = req.max_len;
        if len > 1024 {
            len = 1024
        }

        unsafe {
            let res = cuDeviceGetName(name.as_mut_ptr(), len, req.device);

            let reply = cuda::CuDeviceGetNameResponse {
                result: res.into(),
                name: CString::from_raw(name.as_mut_ptr())
                    .into_string()
                    .expect("failed to convert name"),
            };

            Ok(Response::new(reply))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let c = MyCuda::default();

    Server::builder()
        .add_service(CudaServer::new(c))
        .serve(addr)
        .await?;

    Ok(())
}
