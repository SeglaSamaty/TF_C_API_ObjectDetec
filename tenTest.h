
#ifdef __cplusplus
extern "C" {
#endif

struct OD_Result {
  float* boxes;
  float* scores;
  float* label_ids;
  float* num_detections;
};

TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) { free(data); }

TF_Tensor* CreateTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, const void* data, std::size_t len);

#ifdef __cplusplus
}
#endif