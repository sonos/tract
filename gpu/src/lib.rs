use std::borrow::Cow;
use std::fmt::Debug;
use tract_core::prelude::{natural_strides, DatumType, Tensor};
use tract_data::TVec;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, Buffer, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, Queue,
    ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

pub struct GPUTensor {
    dt: DatumType,
    shape: TVec<usize>,
    strides: TVec<isize>,
    len: usize,
    buffer: Buffer,
}

#[derive(Debug)]
pub struct GpuAccel {
    device: Device,
    queue: Queue,
    sigmoid_shader: ShaderModule,
    tanh_shader: ShaderModule,
}

impl GpuAccel {
    pub async fn default() -> Option<GpuAccel> {
        let instance = Instance::new(wgpu::Backends::all());
        let adapter = match wgpu::util::initialize_adapter_from_env_or_default(
            &instance,
            wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all),
            None,
        )
        .await
        {
            Some(a) => a,
            None => return None,
        };

        let (device, queue) =
            match adapter.request_device(&DeviceDescriptor::default(), None).await.ok() {
                Some((d, q)) => (d, q),
                None => return None,
            };

        println!("{:#?}", adapter.get_info());

        let sigmoid_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/sigmoid.wgsl"))),
        });

        let tanh_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/tanh.wgsl"))),
        });

        Some(GpuAccel { device, queue, sigmoid_shader, tanh_shader })
    }

    pub fn alloc_in_buffer<T: bytemuck::NoUninit>(&self, label: String, bytes: &Vec<T>) -> Buffer {
        self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&label),
            contents: bytemuck::cast_slice(bytes),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        })
    }

    pub fn import_tensor(&self, label: String, t: &Tensor) -> GPUTensor {
        GPUTensor {
            dt: t.datum_type(),
            shape: t.shape().into(),
            strides: t.strides().into(),
            len: t.len(),
            // TODO: proper type
            buffer: self.alloc_in_buffer(label, &Vec::from(t.as_slice::<f32>().unwrap())),
        }
    }

    pub fn alloc_storage_buffer(&self, label: String, size: u64, output: bool) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&label),
            size,
            usage: if output {
                BufferUsages::STORAGE | BufferUsages::MAP_READ | BufferUsages::COPY_SRC
            } else {
                BufferUsages::STORAGE
            },
            mapped_at_creation: false,
        })
    }

    fn create_generic_storage_tensor(
        &self,
        label: String,
        dt: DatumType,
        shape: TVec<usize>,
        output: bool,
    ) -> GPUTensor {
        let strides = natural_strides(&shape);
        let len = if shape.len() == 0 {
            1
        } else {
            *strides.get(0).unwrap() as usize * shape.get(0).unwrap()
        };

        GPUTensor {
            dt,
            shape,
            strides,
            len,
            buffer: self.alloc_storage_buffer(label, len as u64 * dt.size_of() as u64, output),
        }
    }

    pub fn create_storage_tensor(
        &self,
        label: String,
        dt: DatumType,
        shape: TVec<usize>,
    ) -> GPUTensor {
        self.create_generic_storage_tensor(label, dt, shape, false)
    }

    pub fn create_out_tensor(&self, label: String, dt: DatumType, shape: TVec<usize>) -> GPUTensor {
        self.create_generic_storage_tensor(label, dt, shape, true)
    }

    pub async fn buffer_move_out<T: bytemuck::Pod>(&self, buf: Buffer) -> Vec<T> {
        let buffer_slice = buf.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future.await {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            buf.unmap();

            println!("Moved buffer {:?} out from GPU memory", buf);

            return result;
        } else {
            panic!("Failed to move buffer {:?} out from GPU memory", buf);
        }
    }

    pub async fn tensor_move_out(&self, t: GPUTensor) -> Tensor {
        unsafe {
            Tensor::from_raw_dt(t.dt, &t.shape, &self.buffer_move_out::<u8>(t.buffer).await)
                .unwrap()
        }
    }

    pub fn sigmoid(&self, in_tensor: &GPUTensor, out_tensor: &GPUTensor) {
        let bind_group = vec![
            BindGroupEntry { binding: 0, resource: in_tensor.buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out_tensor.buffer.as_entire_binding() },
        ];

        self.execute_shader(
            &"sigmoid".to_string(),
            &self.sigmoid_shader,
            bind_group,
            out_tensor.len as u32,
            1,
            1,
        );
    }

    pub fn tanh(&self, in_tensor: &GPUTensor, out_tensor: &GPUTensor) {
        let bind_group = vec![
            BindGroupEntry { binding: 0, resource: in_tensor.buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: out_tensor.buffer.as_entire_binding() },
        ];

        self.execute_shader(
            &"tanh".to_string(),
            &self.tanh_shader,
            bind_group,
            out_tensor.len as u32,
            1,
            1,
        );
    }

    pub fn execute_shader(
        &self,
        label: &String,
        shader: &ShaderModule,
        bind_group_entries: Vec<BindGroupEntry<'_>>,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) {
        let compute_pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(&(label.clone() + "_pipeline")),
            layout: None,
            module: shader,
            entry_point: "main",
        });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&(label.clone() + "_bind_group")),
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&(label.clone() + "_encoder")),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&(label.clone() + "_compute_pass")),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Number of cells to run, the (x,y,z) size of item being processed
            cpass.dispatch(wg_x, wg_y, wg_z);
        }

        self.queue.submit(Some(encoder.finish()));

        self.device.poll(wgpu::Maintain::Wait);
    }
}
