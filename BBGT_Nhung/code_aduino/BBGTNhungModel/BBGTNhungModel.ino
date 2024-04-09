#include <Arduino.h>
#include "esp_camera.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define CAMERA_MODEL_AI_THINKER // Has PSRAM
#include "camera_pins.h"

const int cameraWidth = 320;
const int cameraHeight = 240;

const int inputTensorWidth = 32;
const int inputTensorHeight = 32;
const int outputTensorSize = 3;
const float detectionThreshold = 0.9;

#include "bbgt_model.h"

TfLiteTensor* inputTensor = nullptr;
camera_fb_t* fb = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

void setup() {
  Serial.begin(115200);

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
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  error_reporter = new tflite::MicroErrorReporter();
  model = tflite::GetModel(bbgt_model);
  interpreter = new tflite::MicroInterpreter(model, *error_reporter);
  interpreter->AllocateTensors();
  inputTensor = interpreter->input(0);
}

void loop() {
  fb = captureImage();
  if (fb) {
    // preprocessImage(fb);
    runInference();
    printResult();
    releaseCapturedImage();
  }
}

camera_fb_t* captureImage() {
  return esp_camera_fb_get();
}

void releaseCapturedImage() {
  esp_camera_fb_return(fb);
}

// void preprocessImage(camera_fb_t* fb) {
//   size_t imgSize = inputTensor->bytes;
//   uint8_t* imgData = inputTensor->data.uint8;

//   // // Resize the captured image to match the input size of the model
//   // resize(fb->buf, imgData, Size(inputTensorWidth, inputTensorHeight));

//   // Convert pixel values to float32 and scale to the range [0, 1]
//   for (size_t i = 0; i < imgSize; ++i) {
//     imgData[i] = (float)imgData[i] / 255.0;
//   }
// }

// void resize(uint8_t* input, uint8_t* output, Size size) {
//   // Function to resize the image from input buffer to output buffer
//   // Implement according to your requirements
// }

void runInference() {
  interpreter->Invoke();
}

void printResult() {
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
  }
}
