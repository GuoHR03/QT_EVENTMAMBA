#include <cuda_runtime.h>

#include <cstdint>
#include <new>
#include <string>

#include "onnxruntime_c_api.h"

namespace {

constexpr int kInputCount = 8;
constexpr int kMaxStateSize = 32;

struct SelectiveScanKernel {
  const OrtApi* api;
};

__device__ __forceinline__ float stable_softplus(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
}

__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__global__ void selective_scan_forward_kernel(
    const float* u,
    const float* delta,
    const float* a_term,
    const float* b_term,
    const float* c_term,
    const float* d_skip,
    const float* z,
    const float* delta_bias,
    float* output,
    int batch,
    int channels,
    int length,
    int state_size) {
  const int batch_channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_channel >= batch * channels) {
    return;
  }

  const int batch_index = batch_channel / channels;
  const int channel = batch_channel % channels;
  const int sequence_offset = batch_channel * length;
  const int state_sequence_offset = batch_index * state_size * length;
  const int a_offset = channel * state_size;

  float state[kMaxStateSize];
#pragma unroll
  for (int index = 0; index < kMaxStateSize; ++index) {
    state[index] = 0.0f;
  }

  for (int position = 0; position < length; ++position) {
    const int input_index = sequence_offset + position;
    const float input = u[input_index];
    const float dt = stable_softplus(delta[input_index] + delta_bias[channel]);
    float value = 0.0f;

    for (int state_index = 0; state_index < state_size; ++state_index) {
      const int state_sequence_index =
          state_sequence_offset + state_index * length + position;
      state[state_index] =
          expf(dt * a_term[a_offset + state_index]) * state[state_index] +
          dt * b_term[state_sequence_index] * input;
      value += state[state_index] * c_term[state_sequence_index];
    }

    value += input * d_skip[channel];
    output[input_index] = value * silu(z[input_index]);
  }
}

__global__ void selective_scan_core_kernel(
    const float* delta_a,
    const float* delta_b_u,
    const float* c_term,
    float* output,
    int batch,
    int channels,
    int length,
    int state_size) {
  const int batch_channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_channel >= batch * channels) {
    return;
  }

  const int batch_index = batch_channel / channels;
  const int sequence_offset = batch_channel * length;
  const int state_sequence_offset = batch_index * state_size * length;
  float state[kMaxStateSize];
#pragma unroll
  for (int index = 0; index < kMaxStateSize; ++index) {
    state[index] = 0.0f;
  }

  for (int position = 0; position < length; ++position) {
    float value = 0.0f;
    const int recurrence_offset =
        (batch_channel * length + position) * state_size;
    for (int state_index = 0; state_index < state_size; ++state_index) {
      state[state_index] =
          delta_a[recurrence_offset + state_index] * state[state_index] +
          delta_b_u[recurrence_offset + state_index];
      const int c_index =
          state_sequence_offset + state_index * length + position;
      value += state[state_index] * c_term[c_index];
    }
    output[sequence_offset + position] = value;
  }
}

OrtStatus* make_error(const OrtApi* api, const std::string& message) {
  return api->CreateStatus(ORT_FAIL, message.c_str());
}

OrtStatus* read_shape(
    const OrtApi* api,
    const OrtValue* value,
    int64_t* dimensions,
    size_t expected_rank) {
  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  OrtStatus* status = api->GetTensorTypeAndShape(value, &shape_info);
  if (status != nullptr) {
    return status;
  }

  size_t rank = 0;
  status = api->GetDimensionsCount(shape_info, &rank);
  if (status == nullptr && rank != expected_rank) {
    status = make_error(api, "SelectiveScan received an input with an invalid rank");
  }
  if (status == nullptr) {
    status = api->GetDimensions(shape_info, dimensions, expected_rank);
  }
  api->ReleaseTensorTypeAndShapeInfo(shape_info);
  return status;
}

OrtStatus* validate_vector_shape(
    const OrtApi* api,
    const OrtValue* value,
    int64_t expected_size,
    const char* name) {
  int64_t dimensions[1] = {};
  OrtStatus* status = read_shape(api, value, dimensions, 1);
  if (status != nullptr) {
    return status;
  }
  if (dimensions[0] != expected_size) {
    return make_error(api, std::string("SelectiveScan shape mismatch for ") + name);
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL create_kernel_v2(
    const OrtCustomOp*,
    const OrtApi* api,
    const OrtKernelInfo*,
    void** kernel) {
  auto* instance = new (std::nothrow) SelectiveScanKernel{api};
  if (instance == nullptr) {
    return api->CreateStatus(ORT_FAIL, "Unable to allocate SelectiveScan kernel");
  }
  *kernel = instance;
  return nullptr;
}

void ORT_API_CALL destroy_kernel(void* kernel) {
  delete static_cast<SelectiveScanKernel*>(kernel);
}

OrtStatus* ORT_API_CALL compute_v2(void* kernel, OrtKernelContext* context) {
  const OrtApi* api = static_cast<SelectiveScanKernel*>(kernel)->api;
  const OrtValue* inputs[kInputCount] = {};
  for (size_t index = 0; index < kInputCount; ++index) {
    OrtStatus* status = api->KernelContext_GetInput(context, index, &inputs[index]);
    if (status != nullptr) {
      return status;
    }
  }

  int64_t u_shape[3] = {};
  OrtStatus* status = read_shape(api, inputs[0], u_shape, 3);
  if (status != nullptr) {
    return status;
  }
  const int64_t batch = u_shape[0];
  const int64_t channels = u_shape[1];
  const int64_t length = u_shape[2];
  if (batch <= 0 || channels <= 0 || length <= 0) {
    return make_error(api, "SelectiveScan requires positive input dimensions");
  }

  int64_t a_shape[2] = {};
  status = read_shape(api, inputs[2], a_shape, 2);
  if (status != nullptr) {
    return status;
  }
  const int64_t state_size = a_shape[1];
  if (a_shape[0] != channels || state_size <= 0 || state_size > kMaxStateSize) {
    return make_error(api, "SelectiveScan A must have shape [channels, state<=32]");
  }

  for (size_t index : {size_t{1}, size_t{6}}) {
    int64_t shape[3] = {};
    status = read_shape(api, inputs[index], shape, 3);
    if (status != nullptr) {
      return status;
    }
    if (shape[0] != batch || shape[1] != channels || shape[2] != length) {
      return make_error(api, "SelectiveScan u, delta and z shapes must match");
    }
  }

  for (size_t index : {size_t{3}, size_t{4}}) {
    int64_t shape[3] = {};
    status = read_shape(api, inputs[index], shape, 3);
    if (status != nullptr) {
      return status;
    }
    if (shape[0] != batch || shape[1] != state_size || shape[2] != length) {
      return make_error(api, "SelectiveScan B/C shape mismatch");
    }
  }

  status = validate_vector_shape(api, inputs[5], channels, "D");
  if (status != nullptr) {
    return status;
  }
  status = validate_vector_shape(api, inputs[7], channels, "delta_bias");
  if (status != nullptr) {
    return status;
  }

  const float* device_inputs[kInputCount] = {};
  for (size_t index = 0; index < kInputCount; ++index) {
    const void* data = nullptr;
    status = api->GetTensorData(inputs[index], &data);
    if (status != nullptr) {
      return status;
    }
    device_inputs[index] = static_cast<const float*>(data);
  }

  OrtValue* output_value = nullptr;
  status = api->KernelContext_GetOutput(context, 0, u_shape, 3, &output_value);
  if (status != nullptr) {
    return status;
  }
  void* output_data = nullptr;
  status = api->GetTensorMutableData(output_value, &output_data);
  if (status != nullptr) {
    return status;
  }

  void* stream_pointer = nullptr;
  status = api->KernelContext_GetGPUComputeStream(context, &stream_pointer);
  if (status != nullptr) {
    return status;
  }
  if (stream_pointer == nullptr) {
    return make_error(api, "SelectiveScan could not obtain the CUDA compute stream");
  }

  constexpr int threads = 128;
  const int work_items = static_cast<int>(batch * channels);
  const int blocks = (work_items + threads - 1) / threads;
  selective_scan_forward_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream_pointer)>>>(
      device_inputs[0],
      device_inputs[1],
      device_inputs[2],
      device_inputs[3],
      device_inputs[4],
      device_inputs[5],
      device_inputs[6],
      device_inputs[7],
      static_cast<float*>(output_data),
      static_cast<int>(batch),
      static_cast<int>(channels),
      static_cast<int>(length),
      static_cast<int>(state_size));

  const cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return make_error(
        api, std::string("SelectiveScan CUDA launch failed: ") + cudaGetErrorString(cuda_status));
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL compute_core_v2(void* kernel, OrtKernelContext* context) {
  const OrtApi* api = static_cast<SelectiveScanKernel*>(kernel)->api;
  const OrtValue* inputs[3] = {};
  for (size_t index = 0; index < 3; ++index) {
    OrtStatus* status = api->KernelContext_GetInput(context, index, &inputs[index]);
    if (status != nullptr) {
      return status;
    }
  }

  int64_t delta_shape[4] = {};
  OrtStatus* status = read_shape(api, inputs[0], delta_shape, 4);
  if (status != nullptr) {
    return status;
  }
  const int64_t batch = delta_shape[0];
  const int64_t channels = delta_shape[1];
  const int64_t length = delta_shape[2];
  const int64_t state_size = delta_shape[3];
  if (batch <= 0 || channels <= 0 || length <= 0 || state_size <= 0 ||
      state_size > kMaxStateSize) {
    return make_error(api, "SelectiveScanCore received invalid dimensions");
  }

  int64_t delta_b_u_shape[4] = {};
  status = read_shape(api, inputs[1], delta_b_u_shape, 4);
  if (status != nullptr) {
    return status;
  }
  for (int index = 0; index < 4; ++index) {
    if (delta_b_u_shape[index] != delta_shape[index]) {
      return make_error(api, "SelectiveScanCore recurrence tensor shapes must match");
    }
  }

  int64_t c_shape[3] = {};
  status = read_shape(api, inputs[2], c_shape, 3);
  if (status != nullptr) {
    return status;
  }
  if (c_shape[0] != batch || c_shape[1] != state_size || c_shape[2] != length) {
    return make_error(api, "SelectiveScanCore C shape mismatch");
  }

  const float* device_inputs[3] = {};
  for (size_t index = 0; index < 3; ++index) {
    const void* data = nullptr;
    status = api->GetTensorData(inputs[index], &data);
    if (status != nullptr) {
      return status;
    }
    device_inputs[index] = static_cast<const float*>(data);
  }

  const int64_t output_shape[3] = {batch, channels, length};
  OrtValue* output_value = nullptr;
  status = api->KernelContext_GetOutput(context, 0, output_shape, 3, &output_value);
  if (status != nullptr) {
    return status;
  }
  void* output_data = nullptr;
  status = api->GetTensorMutableData(output_value, &output_data);
  if (status != nullptr) {
    return status;
  }

  void* stream_pointer = nullptr;
  status = api->KernelContext_GetGPUComputeStream(context, &stream_pointer);
  if (status != nullptr) {
    return status;
  }
  if (stream_pointer == nullptr) {
    return make_error(api, "SelectiveScanCore could not obtain the CUDA compute stream");
  }

  constexpr int threads = 128;
  const int work_items = static_cast<int>(batch * channels);
  const int blocks = (work_items + threads - 1) / threads;
  selective_scan_core_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream_pointer)>>>(
      device_inputs[0],
      device_inputs[1],
      device_inputs[2],
      static_cast<float*>(output_data),
      static_cast<int>(batch),
      static_cast<int>(channels),
      static_cast<int>(length),
      static_cast<int>(state_size));
  const cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return make_error(
        api, std::string("SelectiveScanCore CUDA launch failed: ") +
                 cudaGetErrorString(cuda_status));
  }
  return nullptr;
}

const char* ORT_API_CALL get_name(const OrtCustomOp*) {
  return "SelectiveScan";
}

const char* ORT_API_CALL get_core_name(const OrtCustomOp*) {
  return "SelectiveScanCore";
}

const char* ORT_API_CALL get_execution_provider(const OrtCustomOp*) {
  return "CUDAExecutionProvider";
}

ONNXTensorElementDataType ORT_API_CALL get_input_type(const OrtCustomOp*, size_t) {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL get_input_count(const OrtCustomOp*) {
  return kInputCount;
}

size_t ORT_API_CALL get_core_input_count(const OrtCustomOp*) {
  return 3;
}

ONNXTensorElementDataType ORT_API_CALL get_output_type(const OrtCustomOp*, size_t) {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ORT_API_CALL get_output_count(const OrtCustomOp*) {
  return 1;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL get_characteristic(
    const OrtCustomOp*, size_t) {
  return INPUT_OUTPUT_REQUIRED;
}

OrtMemType ORT_API_CALL get_input_memory_type(const OrtCustomOp*, size_t) {
  return OrtMemTypeDefault;
}

int ORT_API_CALL get_start_version(const OrtCustomOp*) {
  return 1;
}

int ORT_API_CALL get_end_version(const OrtCustomOp*) {
  return 1;
}

OrtCustomOp make_custom_op() {
  OrtCustomOp op{};
  op.version = ORT_API_VERSION;
  op.GetName = get_name;
  op.GetExecutionProviderType = get_execution_provider;
  op.GetInputType = get_input_type;
  op.GetInputTypeCount = get_input_count;
  op.GetOutputType = get_output_type;
  op.GetOutputTypeCount = get_output_count;
  op.KernelDestroy = destroy_kernel;
  op.GetInputCharacteristic = get_characteristic;
  op.GetOutputCharacteristic = get_characteristic;
  op.GetInputMemoryType = get_input_memory_type;
  op.CreateKernelV2 = create_kernel_v2;
  op.KernelComputeV2 = compute_v2;
  op.GetStartVersion = get_start_version;
  op.GetEndVersion = get_end_version;
  return op;
}

OrtCustomOp make_core_custom_op() {
  OrtCustomOp op = make_custom_op();
  op.GetName = get_core_name;
  op.GetInputTypeCount = get_core_input_count;
  op.KernelComputeV2 = compute_core_v2;
  return op;
}

OrtCustomOp g_custom_op = make_custom_op();
OrtCustomOp g_core_custom_op = make_core_custom_op();
OrtCustomOpDomain* g_domain = nullptr;

}  // namespace

extern "C" __declspec(dllexport) OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) {
    return nullptr;
  }
  if (g_domain == nullptr) {
    OrtStatus* status = api->CreateCustomOpDomain("com.eventmamba", &g_domain);
    if (status != nullptr) {
      return status;
    }
    status = api->CustomOpDomain_Add(g_domain, &g_custom_op);
    if (status != nullptr) {
      return status;
    }
    status = api->CustomOpDomain_Add(g_domain, &g_core_custom_op);
    if (status != nullptr) {
      return status;
    }
  }
  return api->AddCustomOpDomain(options, g_domain);
}
