#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//load model
#include "cnn_sine_model.h"

// Some settings
constexpr int led_pin = 2;
constexpr float pi = 3.14159265;                  // Some pi
constexpr float freq = 2;                       // Frequency (Hz) of sinewave
constexpr float period = (1 / freq) * (1000);  // Period (mili seconds)

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
 TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;


// Define the size of the input and output tensors
// constexpr int kInputTensorSize = 2;
// constexpr int kOutputTensorSize = 1;

// Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 64* 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace
// // Define the input and output tensors
// TfLiteTensor* input_tensor;
// TfLiteTensor* output_tensor;

void setup() {
  // Let's make an LED vary in brightness
  pinMode(led_pin, OUTPUT);
  Serial.begin(115200);

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(cnn_sine_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
}

void loop() {
  // put your main code here, to run repeatedly:
 // Get current timestamp and modulo with period
  unsigned long timestamp = millis();
  timestamp = timestamp % (unsigned long)period;

  // Calculate x value to feed to the model
  float x_val = ((float)timestamp * 2 * pi) / period;

  // Copy value to input buffer (tensor)
  model_input->data.f[0] = sin(x_val);
  model_input->data.f[1] = sin(x_val + pi/4);
  model_input->data.f[2] = sin(x_val + 2*pi/4);
  model_input->data.f[3] = sin(x_val + 3*pi/4);
  model_input->data.f[4] = sin(x_val + 4*pi/4);
    // model_input->data.f[0] = x_val;
    // model_input->data.f[1] = x_val;
    // model_input->data.f[2] = x_val;
    // model_input->data.f[3] = x_val;
    // model_input->data.f[4] = x_val;
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input: %f\n", x_val);
  }

  // Read predicted y value from output buffer (tensor)
  float y_val = model_output->data.f[0];
  delay(9);
  // Translate to a PWM LED brightness
  int brightness = (int)(255 * y_val);
  analogWrite(led_pin, brightness);

  // Print value
  Serial.println(y_val);
  
}
