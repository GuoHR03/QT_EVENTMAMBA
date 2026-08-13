#include "hierarchical_fps_custom_op.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace {

constexpr int64_t kCoordinateCount = 3;
constexpr int64_t kInputPointCount = 1024;
constexpr int64_t kFps0Count = 512;
constexpr int64_t kFps1Count = 256;
constexpr int64_t kFps2Count = 128;
constexpr float kInitialDistance = 1.0e10F;

struct HierarchicalFpsKernel {
  const OrtApi* api;
};

OrtStatus* make_invalid_argument(const OrtApi* api, const char* message) {
  return api->CreateStatus(ORT_INVALID_ARGUMENT, message);
}

OrtStatus* read_shape(
    const OrtApi* api,
    const OrtValue* value,
    int64_t* dimensions,
    size_t expected_rank,
    const char* rank_error) {
  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  OrtStatus* status = api->GetTensorTypeAndShape(value, &shape_info);
  if (status != nullptr) {
    return status;
  }

  size_t rank = 0;
  status = api->GetDimensionsCount(shape_info, &rank);
  if (status == nullptr && rank != expected_rank) {
    status = make_invalid_argument(api, rank_error);
  }
  if (status == nullptr) {
    status = api->GetDimensions(shape_info, dimensions, expected_rank);
  }
  api->ReleaseTensorTypeAndShapeInfo(shape_info);
  return status;
}

template <size_t PointCount, size_t SampleCount>
void farthest_point_sample(
    const float* points,
    int64_t start,
    int64_t* output) {
  std::array<float, PointCount> minimum_distances{};
  minimum_distances.fill(kInitialDistance);
  int64_t farthest = start;

  for (size_t sample = 0; sample < SampleCount; ++sample) {
    output[sample] = farthest;

    const float center_x = points[farthest];
    const float center_y = points[PointCount + farthest];
    const float center_z = points[2 * PointCount + farthest];

    farthest = 0;
    float greatest_distance = -std::numeric_limits<float>::infinity();
    for (size_t point = 0; point < PointCount; ++point) {
      const int64_t point_index = static_cast<int64_t>(point);
      const float delta_x = points[point_index] - center_x;
      const float delta_y = points[PointCount + point_index] - center_y;
      const float delta_z = points[2 * PointCount + point_index] - center_z;
      float squared_distance = delta_x * delta_x;
      squared_distance = squared_distance + delta_y * delta_y;
      squared_distance = squared_distance + delta_z * delta_z;
      const float candidate =
          squared_distance < minimum_distances[point]
              ? squared_distance
              : minimum_distances[point];
      minimum_distances[point] = candidate;
      if (candidate > greatest_distance) {
        farthest = point_index;
        greatest_distance = candidate;
      }
    }
  }
}

OrtStatus* ORT_API_CALL create_fps_kernel(
    const OrtCustomOp*,
    const OrtApi* api,
    const OrtKernelInfo*,
    void** kernel) {
  auto* instance = new (std::nothrow) HierarchicalFpsKernel{api};
  if (instance == nullptr) {
    return api->CreateStatus(
        ORT_FAIL, "Unable to allocate HierarchicalFarthestPointSampling kernel");
  }
  *kernel = instance;
  return nullptr;
}

void ORT_API_CALL destroy_fps_kernel(void* kernel) {
  delete static_cast<HierarchicalFpsKernel*>(kernel);
}

OrtStatus* ORT_API_CALL compute_fps(void* kernel, OrtKernelContext* context) {
  const OrtApi* api = static_cast<HierarchicalFpsKernel*>(kernel)->api;
  const OrtValue* events_value = nullptr;
  OrtStatus* status = api->KernelContext_GetInput(context, 0, &events_value);
  if (status != nullptr) {
    return status;
  }
  const OrtValue* starts_value = nullptr;
  status = api->KernelContext_GetInput(context, 1, &starts_value);
  if (status != nullptr) {
    return status;
  }

  int64_t events_shape[3] = {};
  status = read_shape(
      api,
      events_value,
      events_shape,
      3,
      "HierarchicalFarthestPointSampling events must have rank 3");
  if (status != nullptr) {
    return status;
  }
  const int64_t batch = events_shape[0];
  if (batch <= 0 || events_shape[1] != kCoordinateCount ||
      events_shape[2] != kInputPointCount) {
    return make_invalid_argument(
        api,
        "HierarchicalFarthestPointSampling events must have shape [B,3,1024] with B > 0");
  }

  int64_t starts_shape[2] = {};
  status = read_shape(
      api,
      starts_value,
      starts_shape,
      2,
      "HierarchicalFarthestPointSampling fps_starts must have rank 2");
  if (status != nullptr) {
    return status;
  }
  if (starts_shape[0] != batch || starts_shape[1] != 3) {
    return make_invalid_argument(
        api,
        "HierarchicalFarthestPointSampling fps_starts must have shape [B,3]");
  }

  const void* events_data_raw = nullptr;
  status = api->GetTensorData(events_value, &events_data_raw);
  if (status != nullptr) {
    return status;
  }
  const void* starts_data_raw = nullptr;
  status = api->GetTensorData(starts_value, &starts_data_raw);
  if (status != nullptr) {
    return status;
  }
  const auto* events = static_cast<const float*>(events_data_raw);
  const auto* starts = static_cast<const int64_t*>(starts_data_raw);

  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    const float* batch_events =
        events + batch_index * kCoordinateCount * kInputPointCount;
    for (int64_t value_index = 0;
         value_index < kCoordinateCount * kInputPointCount;
         ++value_index) {
      if (!std::isfinite(batch_events[value_index])) {
        return make_invalid_argument(
            api,
            "HierarchicalFarthestPointSampling events values must be finite");
      }
    }
    const int64_t* batch_starts = starts + batch_index * 3;
    if (batch_starts[0] < 0 || batch_starts[0] >= kInputPointCount ||
        batch_starts[1] < 0 || batch_starts[1] >= kFps0Count ||
        batch_starts[2] < 0 || batch_starts[2] >= kFps1Count) {
      return make_invalid_argument(
          api,
          "HierarchicalFarthestPointSampling fps_starts values are out of range");
    }
  }

  const int64_t fps0_shape[2] = {batch, kFps0Count};
  const int64_t fps1_shape[2] = {batch, kFps1Count};
  const int64_t fps2_shape[2] = {batch, kFps2Count};
  OrtValue* fps0_value = nullptr;
  status = api->KernelContext_GetOutput(context, 0, fps0_shape, 2, &fps0_value);
  if (status != nullptr) {
    return status;
  }
  OrtValue* fps1_value = nullptr;
  status = api->KernelContext_GetOutput(context, 1, fps1_shape, 2, &fps1_value);
  if (status != nullptr) {
    return status;
  }
  OrtValue* fps2_value = nullptr;
  status = api->KernelContext_GetOutput(context, 2, fps2_shape, 2, &fps2_value);
  if (status != nullptr) {
    return status;
  }

  void* fps0_data_raw = nullptr;
  status = api->GetTensorMutableData(fps0_value, &fps0_data_raw);
  if (status != nullptr) {
    return status;
  }
  void* fps1_data_raw = nullptr;
  status = api->GetTensorMutableData(fps1_value, &fps1_data_raw);
  if (status != nullptr) {
    return status;
  }
  void* fps2_data_raw = nullptr;
  status = api->GetTensorMutableData(fps2_value, &fps2_data_raw);
  if (status != nullptr) {
    return status;
  }
  auto* fps0 = static_cast<int64_t*>(fps0_data_raw);
  auto* fps1 = static_cast<int64_t*>(fps1_data_raw);
  auto* fps2 = static_cast<int64_t*>(fps2_data_raw);

  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    const float* batch_events =
        events + batch_index * kCoordinateCount * kInputPointCount;
    const int64_t* batch_starts = starts + batch_index * 3;
    int64_t* batch_fps0 = fps0 + batch_index * kFps0Count;
    int64_t* batch_fps1 = fps1 + batch_index * kFps1Count;
    int64_t* batch_fps2 = fps2 + batch_index * kFps2Count;

    farthest_point_sample<kInputPointCount, kFps0Count>(
        batch_events, batch_starts[0], batch_fps0);

    std::array<int64_t, kFps0Count> fps1_point_map{};
    std::copy_n(batch_fps0, kFps0Count, fps1_point_map.begin());
    std::sort(fps1_point_map.begin(), fps1_point_map.end());
    std::array<float, kCoordinateCount * kFps0Count> fps1_points{};
    for (int64_t coordinate = 0; coordinate < kCoordinateCount; ++coordinate) {
      for (int64_t point = 0; point < kFps0Count; ++point) {
        fps1_points[coordinate * kFps0Count + point] =
            batch_events[coordinate * kInputPointCount + fps1_point_map[point]];
      }
    }
    farthest_point_sample<kFps0Count, kFps1Count>(
        fps1_points.data(), batch_starts[1], batch_fps1);

    std::array<int64_t, kFps1Count> fps2_point_map{};
    std::copy_n(batch_fps1, kFps1Count, fps2_point_map.begin());
    std::sort(fps2_point_map.begin(), fps2_point_map.end());
    std::array<float, kCoordinateCount * kFps1Count> fps2_points{};
    for (int64_t coordinate = 0; coordinate < kCoordinateCount; ++coordinate) {
      for (int64_t point = 0; point < kFps1Count; ++point) {
        fps2_points[coordinate * kFps1Count + point] =
            fps1_points[coordinate * kFps0Count + fps2_point_map[point]];
      }
    }
    farthest_point_sample<kFps1Count, kFps2Count>(
        fps2_points.data(), batch_starts[2], batch_fps2);
  }
  return nullptr;
}

const char* ORT_API_CALL get_fps_name(const OrtCustomOp*) {
  return "HierarchicalFarthestPointSampling";
}

const char* ORT_API_CALL get_fps_execution_provider(const OrtCustomOp*) {
  return nullptr;
}

ONNXTensorElementDataType ORT_API_CALL get_fps_input_type(
    const OrtCustomOp*, size_t index) {
  return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

size_t ORT_API_CALL get_fps_input_count(const OrtCustomOp*) {
  return 2;
}

ONNXTensorElementDataType ORT_API_CALL get_fps_output_type(
    const OrtCustomOp*, size_t) {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

size_t ORT_API_CALL get_fps_output_count(const OrtCustomOp*) {
  return 3;
}

OrtCustomOpInputOutputCharacteristic ORT_API_CALL get_fps_characteristic(
    const OrtCustomOp*, size_t) {
  return INPUT_OUTPUT_REQUIRED;
}

OrtMemType ORT_API_CALL get_fps_input_memory_type(const OrtCustomOp*, size_t) {
  return OrtMemTypeDefault;
}

int ORT_API_CALL get_fps_start_version(const OrtCustomOp*) {
  return 1;
}

int ORT_API_CALL get_fps_end_version(const OrtCustomOp*) {
  return 1;
}

OrtCustomOp make_fps_custom_op() {
  OrtCustomOp op{};
  op.version = ORT_API_VERSION;
  op.GetName = get_fps_name;
  op.GetExecutionProviderType = get_fps_execution_provider;
  op.GetInputType = get_fps_input_type;
  op.GetInputTypeCount = get_fps_input_count;
  op.GetOutputType = get_fps_output_type;
  op.GetOutputTypeCount = get_fps_output_count;
  op.KernelDestroy = destroy_fps_kernel;
  op.GetInputCharacteristic = get_fps_characteristic;
  op.GetOutputCharacteristic = get_fps_characteristic;
  op.GetInputMemoryType = get_fps_input_memory_type;
  op.CreateKernelV2 = create_fps_kernel;
  op.KernelComputeV2 = compute_fps;
  op.GetStartVersion = get_fps_start_version;
  op.GetEndVersion = get_fps_end_version;
  return op;
}

OrtCustomOp g_fps_custom_op = make_fps_custom_op();

}  // namespace

namespace eventmamba {

const OrtCustomOp* GetHierarchicalFpsCustomOp() {
  return &g_fps_custom_op;
}

}  // namespace eventmamba
