#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_camera.h"

#define ASCII_CHARS " .:-=+*#%@"

#include "bbgt_model.h"

const float detectionThreshold = 0.8;

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// 4 for flash led or 33 for normal led
#define LED_GPIO_NUM       4

// Constants for image processing
constexpr int kImageWidth = 32;
constexpr int kImageHeight = 32;
constexpr int kImageChannels = 1; // Grayscale
uint8_t image_data[kImageWidth * kImageHeight * kImageChannels];


// TensorFlow Lite variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
constexpr int kTensorArenaSize = 32*1024;
uint8_t tensor_arena[kTensorArenaSize];


// Function to resize image to 32x32
void resize_image_to_32x32(uint8_t* input, uint8_t* output, int inputWidth, int inputHeight) {
  float scaleWidth = inputWidth / (float)kImageWidth;
  float scaleHeight = inputHeight / (float)kImageHeight;

  for (int y = 0; y < kImageHeight; y++) {
    for (int x = 0; x < kImageWidth; x++) {
      int srcX = (int)(x * scaleWidth);
      int srcY = (int)(y * scaleHeight);
      srcX = min(srcX, inputWidth - 1);
      srcY = min(srcY, inputHeight - 1);
      int inputIndex = (srcY * inputWidth) + srcX;
      int outputIndex = (y * kImageWidth) + x;
      output[outputIndex] = input[inputIndex];
    }
  }
}

// void preprocess_image(uint8_t* input, float* output) {
//   // Convert image to grayscale and normalize pixel values
//   for (int i = 0; i < kImageWidth * kImageHeight; i++) {
//     output[i] = input[i] / 255.0f;
//   }
// }


void grayscale(uint8_t* input, uint8_t* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            output[index] = (input[index * 3] + input[index * 3 + 1] + input[index * 3 + 2]) / 3; // Trung bình cộng của các kênh màu RGB
        }
    }
}

// Cân bằng histogram
void equalize(uint8_t* input, uint8_t* output, int width, int height) {
    int hist[256] = {0};

    // Tính toán histogram
    for (int i = 0; i < width * height; i++) {
        hist[input[i]]++;
    }

    // Tính toán phân bố tích lũy
    int cumulative = 0;
    int minCumulative = width * height;
    for (int i = 0; i < 256; i++) {
        cumulative += hist[i];
        if (cumulative > 0 && cumulative < minCumulative) {
            minCumulative = cumulative;
        }
    }

    // Chuẩn hóa histogram
    float scale = 255.0 / (width * height - minCumulative);
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += hist[i];
        hist[i] = round((sum - minCumulative) * scale);
    }

    // Ánh xạ lại các giá trị pixel
    for (int i = 0; i < width * height; i++) {
        output[i] = hist[input[i]];
    }
}

// Tiền xử lý ảnh
void preprocess_image(uint8_t* input, float* output, int width, int height) {
    uint8_t grayscale_img[width * height];
    uint8_t equalized_img[width * height];
    float normalized_img[width * height];

    // Chuyển đổi sang ảnh xám
    grayscale(input, grayscale_img, width, height);

    // Cân bằng histogram
    equalize(grayscale_img, equalized_img, width, height);

    // Chuẩn hóa ảnh
    for (int i = 0; i < width * height; i++) {
        output[i] = equalized_img[i] / 255.0f;
    }
}

// Function to convert pixel value to ASCII character
char pixelToAscii(uint8_t pixelValue) {
  // ASCII characters arranged from darkest to lightest
  static const char asciiChars[] = " .:-=+*#";
  // Determine ASCII index based on pixel value
  int asciiIndex = map(pixelValue, 0, 255, 0, strlen(asciiChars) - 1);
  // Return corresponding ASCII character
  return asciiChars[asciiIndex];
}


// Function to display image as ASCII art with smaller size
void displayAsciiArt(camera_fb_t *fb) {
  // Display image as ASCII art in Serial Monitor
  for (int y = 0; y < fb->height; y += 8) {
    for (int x = 0; x < fb->width; x += 4) {
      // Calculate average pixel value in 8x8 block
      int sum = 0;
      for (int dy = 0; dy < 8; dy++) {
        for (int dx = 0; dx < 4; dx++) {
          int index = (y + dy) * fb->width + (x + dx);
          sum += fb->buf[index];
        }
      }
      int avg = sum / 32;

      // Convert average pixel value to ASCII character
      char asciiChar = pixelToAscii(avg);
      
      // Print ASCII character
      Serial.write(asciiChar);
    }
    Serial.println(); // Newline after each row
  }
}

const char* classNames[] = {
  "Gioi han toc do (30km/h)",
  "Giao nhau voi duong uu tien",
  "Cong trinh dang thi cong"
};

// Function to print prediction result
void printResult() {
  // Get output tensor
  TfLiteTensor* outputTensor = interpreter->output(0);
  
  // Find the predicted class with the highest probability
  int predictedClass = -1;
  float maxProbability = 0.0;
  for (int i = 0; i < outputTensor->dims->data[1]; ++i) {
    float probability = outputTensor->data.f[i];
    if (probability > maxProbability) {
      maxProbability = probability;
      predictedClass = i;
    }
  }

  // // Print the result if the probability is above the threshold
  // if (maxProbability > detectionThreshold) {
  //   Serial.print("PredictedClass: ");
  //   Serial.println(predictedClass);
  //   Serial.print("Probability: ");
  //   Serial.println(maxProbability);
  // }

    // Print the result if the probability is above the threshold
  if (maxProbability > detectionThreshold) {
    Serial.print("Predicted Class: ");
    Serial.println(predictedClass);
    Serial.print("Class Name: ");
    Serial.println(classNames[predictedClass]);
    Serial.print("Probability: ");
    Serial.println(maxProbability);
  }
}

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
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Initialize TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(bbgt_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
}

void loop() {
  // Capture image
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  // Resize image to 32x32
  resize_image_to_32x32(fb->buf, image_data, fb->width, fb->height);


   // Preprocess image data
  preprocess_image(image_data, model_input->data.f, kImageWidth, kImageHeight);


  // Invoke interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on x_val");
    return;
  }

  // Display image as ASCII art
  displayAsciiArt(fb);

  // Print the results directly
    Serial.print("Prediction for class 1(ghtd 30k/h): ");
    Serial.println(model_output->data.f[0]);

    Serial.print("Prediction for class 2(giao duong uu tien): ");
    Serial.println(model_output->data.f[1]);

    Serial.print("Prediction for class 3(ct dang thi cong): ");
    Serial.println(model_output->data.f[2]);


  // Print prediction result
  printResult();
  
  Serial.println(". ");
  Serial.println(".. ");
  Serial.println();
  Serial.println();

  // Cleanup
  esp_camera_fb_return(fb);

  delay(3000);

}
