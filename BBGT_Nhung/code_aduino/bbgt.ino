#include <Arduino.h>
#include "esp_camera.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Camera settings
const int cameraWidth = 320;
const int cameraHeight = 240;

// TensorFlow Lite model settings
const int inputTensorWidth = 32;
const int inputTensorHeight = 32;
const int outputTensorSize = 3;  // Number of output classes in your model
const float detectionThreshold = 0.9;  // Probability threshold for detection

// TensorFlow Lite model file
#include "bbgt_model.h"

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;

camera_fb_t* captureImage();
void releaseCapturedImage(camera_fb_t* fb);
void preprocessImage(camera_fb_t* fb);
void runInference();
void printResult();

void setup() {
  Serial.begin(115200);

  // Camera initialization
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;  // XCLK 20MHz
  config.pixel_format = PIXFORMAT_GRAYSCALE;  // Grayscale format
  config.frame_size = FRAMESIZE_QVGA;  // QVGA resolution
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // TensorFlow Lite model initialization
  error_reporter = new tflite::MicroErrorReporter();
  model = tflite::GetModel(bbgt_model);
  interpreter = new tflite::MicroInterpreter(model, *error_reporter, inputTensor, outputTensorSize);
  interpreter->AllocateTensors();
  inputTensor = interpreter->input(0);
}

void loop() {
  camera_fb_t* fb = captureImage();
  preprocessImage(fb);
  runInference();
  printResult();
  releaseCapturedImage(fb);
}

camera_fb_t* captureImage() {
  camera_fb_t* fb = esp_camera_fb_get();
  return fb;
}

void releaseCapturedImage(camera_fb_t* fb) {
  esp_camera_fb_return(fb);
}

void preprocessImage(camera_fb_t* fb) {
  uint8_t* imgData = inputTensor->data.uint8;
  size_t imgSize = inputTensor->bytes;

  // Resize the captured image to match the input size of the model
  cv2.resize(fb->buf, imgData, (inputTensorWidth, inputTensorHeight));

  // Preprocess the image (if needed)
  // You may need to adjust this part based on how you preprocessed the images during training
  // For example: convert to grayscale, normalize pixel values, etc.
  // ...

  // Convert pixel values to float32 and scale to the range [0, 1]
  for (size_t i = 0; i < imgSize; ++i) {
    imgData[i] = (float)imgData[i] / 255.0;
  }
}

void runInference() {
  interpreter->Invoke();
}

const char* classNames[] = {
  "Gioi han toc do (30km/h)",
  "Giao nhau voi duong uu tien",
  "Cong trinh dang thi cong"
};

void printResult() {
  // Get the output tensor values
  TfLiteTensor* outputTensor = interpreter->output(0);
  int predictedClass = -1;
  float maxProbability = 0.0;

  // Find the predicted class with the highest probability
  for (int i = 0; i < outputTensorSize; ++i) {
    float probability = outputTensor->data.f[i];
    if (probability > maxProbability) {
      maxProbability = probability;
      predictedClass = i;
    }
  }

  // Print the result if the probability is above the threshold
  if (maxProbability > detectionThreshold) {
    Serial.print("Predicted Class: ");
    Serial.println(predictedClass);
    Serial.print("Probability: ");
    Serial.println(maxProbability);
    // You can add your class labels here based on your model's classes
    // For example: Serial.println(getClassName(predictedClass));
  }

//   // Print the result if the probability is above the threshold
//   if (maxProbability > detectionThreshold) {
//     Serial.print("Predicted Class: ");
//     Serial.println(predictedClass);
//     Serial.print("Class Name: ");
//     Serial.println(classNames[predictedClass]);
//     Serial.print("Probability: ");
//     Serial.println(maxProbability);
//   }
}
