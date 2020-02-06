#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector> 


void deallocator(void* ptr, size_t len, void* arg) { free((void*)ptr); }

#include "tenTest.h"

extern "C" {

int main(int argc, char const* argv[]) {

  //####################### Var
  
  float confidence_score_threshold;
  int max_detections;


  TF_Buffer* graph_def;
  TF_Graph* graph;
  TF_Status* status;
  TF_ImportGraphDefOptions* opts;
  TF_SessionOptions* opt;
  TF_Session* sess;
  //cv::Mat frame;
  cv::VideoCapture cap;

  //####################### fin var


  // recharge le graphe
  // ================================================================================
  
  graph_def = read_file("humanDetect.pb");
  
  graph = TF_NewGraph();
  status = TF_NewStatus();
  opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "graphe non construit %s\n", TF_Message(status));
    return 1;
  }
  fprintf(stdout, "Graphe construit avec success\n");

  // creer une session
  // ================================================================================
  opt = TF_NewSessionOptions();
  sess = TF_NewSession(graph, opt, status);
  TF_DeleteSessionOptions(opt);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "Creation de session echec %s\n", TF_Message(status));
    return 1;
  }
  fprintf(stdout, "Ok pour session\n");
  fprintf(stdout, "-0-----\n");
  // tensor d'entre: pour l'image
  // ================================================================================
  size_t pos = 0;
  TF_Operation* oper;
  while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
    if ((int)pos < 10)
    {
      printf("%s\n", TF_OperationName(oper));
      printf("%d\n", (int)pos);
    }
     
  }

    ///*
  
    // ouverture de la camera.
    if(!cap.open(0))
        return 0;
    while(true)
    {
          TF_Operation* input_op;
          TF_Output input_opout;
          std::vector<TF_Output> input_ops;
          std::vector<TF_Tensor*> input_values;
          TF_Operation* boxes_op;
          TF_Operation* scores_op;
          TF_Operation* classes_op;
          TF_Operation* num_detections_op;
          TF_Output boxes_opout, scores_opout, classes_opout, num_detections_opout;
          std::vector<TF_Output> output_ops;
          std::vector<TF_Tensor*> output_values;


          cv::Mat frame;


          //cap >> frame;
          //if (! cap.read(frame)) continue;
          //printf(" AAA\n");
          cap.read(frame);
          //frame = cv::imread("img.jpg", 1);
          
          resize(frame, frame, cv::Size(400, 400));
          //cvtColor(frame, frame, 4);
          

          std::vector<std::int64_t> input_dims = {1, frame.cols, frame.rows, frame.channels()};
          int image_size_by_dims = frame.cols*frame.rows*frame.channels();

          int image_tensor_size = std::min(image_size_by_dims, image_size_by_dims); // a bit of dum


          TF_Tensor* input_value = CreateTensor(TF_UINT8, input_dims.data(), input_dims.size(), frame.data, image_tensor_size);

 
          input_values.emplace_back(input_value);
  // Create output variable
          max_detections = 100;
          std::vector<std::int64_t> box_dims = {1, max_detections, 4};
          std::vector<std::int64_t> scores_dims = {1, max_detections};
          std::vector<std::int64_t> classes_dims = {1, max_detections};
          std::vector<std::int64_t> num_detections_dims = {1, 1};

          TF_Tensor* boxes_value = TF_AllocateTensor(TF_FLOAT, box_dims.data(), box_dims.size(), sizeof(float) * 4 * max_detections);
          TF_Tensor* scores_value = TF_AllocateTensor(TF_FLOAT, scores_dims.data(), scores_dims.size(), sizeof(float) * max_detections);
          TF_Tensor* classes_value = TF_AllocateTensor(TF_FLOAT, classes_dims.data(), classes_dims.size(), sizeof(float) * max_detections);
          TF_Tensor* num_detections_value = TF_AllocateTensor(TF_FLOAT, num_detections_dims.data(), num_detections_dims.size(), sizeof(float));
 
  //std::vector<TF_Tensor*> output_values;
          output_values.emplace_back(boxes_value);
          output_values.emplace_back(scores_value);
          output_values.emplace_back(classes_value);
          output_values.emplace_back(num_detections_value);


          input_op = TF_GraphOperationByName(graph, "image_tensor");
          input_opout = {input_op, 0};
          input_ops.push_back(input_opout);
  // Set up output ops
          boxes_op = TF_GraphOperationByName(graph, "detection_boxes");
          scores_op = TF_GraphOperationByName(graph, "detection_scores");
          classes_op = TF_GraphOperationByName(graph, "detection_classes");
          num_detections_op = TF_GraphOperationByName(graph, "num_detections");

          boxes_opout = {boxes_op, 0};
          scores_opout = {scores_op, 0};
          classes_opout = {classes_op, 0};
          num_detections_opout = {num_detections_op, 0};

          output_ops.push_back(boxes_opout);
          output_ops.push_back(scores_opout);
          output_ops.push_back(classes_opout);
          output_ops.push_back(num_detections_opout);

          TF_Output* inputs_ptr = input_ops.empty() ? nullptr : &input_ops[0];
          TF_Tensor** input_values_ptr = input_values.empty() ? nullptr : &input_values[0];
          TF_Output* outputs_ptr = output_ops.empty() ? nullptr : &output_ops[0];
          TF_Tensor** output_values_ptr = output_values.empty() ? nullptr : &output_values[0];


          TF_SessionRun(sess, nullptr, inputs_ptr, input_values_ptr, input_ops.size(), outputs_ptr, output_values_ptr, output_ops.size(), nullptr, 0, nullptr, status);
          
          OD_Result od_result; 
          od_result.boxes = (float*)TF_TensorData(output_values[0]);
          od_result.scores = (float*)TF_TensorData(output_values[1]);
          od_result.label_ids = (float*)TF_TensorData(output_values[2]);
          od_result.num_detections = (float*)TF_TensorData(output_values[3]);

          int img_width = frame.cols;
          int img_height = frame.rows;
          int img_channel = frame.channels();  
          int num_detections = (int)od_result.num_detections[0];
          int box_cnt = 0;

          
          int nombre_persone=0;


          for (int i=0; i<num_detections; i++) {
              if (od_result.label_ids[i] == 1.0 && od_result.scores[i] >= 0.6) {
                  printf("persone: %d ", i);
                  printf(", score: %f \n", od_result.scores[i]);
                  int xmin = (int)(od_result.boxes[i*4+1] * img_width);
                  int ymin = (int)(od_result.boxes[i*4+0] * img_height);
                  int xmax = (int)(od_result.boxes[i*4+3] * img_width);
                  int ymax = (int)(od_result.boxes[i*4+2] * img_height);

                  nombre_persone++;
      
                  cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), CV_RGB(0, 255, 255), 2);
      
                  box_cnt++;
              }
          }

          printf("Nombre de persone: %d ", nombre_persone);
          
          cv::imshow("Video :", frame);

          if(cv::waitKey(10) == 27) break;
    }

 
 
  TF_CloseSession(sess, status);
  TF_DeleteSession(sess, status);
  TF_DeleteStatus(status);
  TF_DeleteBuffer(graph_def);

  TF_DeleteGraph(graph);

  return 0;
}



TF_Tensor* CreateTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, const void* data, std::size_t len) {
  if (dims == nullptr || data == nullptr) {
    return nullptr;
  }
  TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
  if (tensor == nullptr) {
    return nullptr;
  }
  std::memcpy(TF_TensorData(tensor), data, std::min(len, TF_TensorByteSize(tensor)));
  return tensor;
}

TF_Buffer* read_file(const char* file) {
  FILE* f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  // same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

}
