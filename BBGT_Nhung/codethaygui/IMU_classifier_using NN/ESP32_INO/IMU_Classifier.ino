/*
  IMU Classifier

  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.

  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry

  This example code is in the public domain.
*/



#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include "gesture_model.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "punch",
  "flex"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))
float input_data[714] = {
    0.456, 0.355, 0.639, 0.504, 0.498, 0.503, 0.446, 0.348,
    0.630, 0.504, 0.499, 0.503, 0.437, 0.342, 0.620, 0.504,
    0.500, 0.504, 0.427, 0.334, 0.611, 0.502, 0.501, 0.506,
    0.420, 0.326, 0.604, 0.501, 0.501, 0.510, 0.413, 0.324,
    0.596, 0.499, 0.500, 0.514, 0.411, 0.327, 0.585, 0.497,
    0.499, 0.519, 0.409, 0.338, 0.571, 0.495, 0.498, 0.526,
    0.411, 0.357, 0.556, 0.492, 0.497, 0.533, 0.413, 0.384,
    0.541, 0.487, 0.495, 0.541, 0.414, 0.420, 0.525, 0.481,
    0.493, 0.548, 0.427, 0.473, 0.511, 0.476, 0.491, 0.553,
    0.440, 0.527, 0.508, 0.472, 0.487, 0.557, 0.455, 0.584,
    0.515, 0.470, 0.482, 0.559, 0.476, 0.643, 0.526, 0.471,
    0.477, 0.557, 0.492, 0.689, 0.540, 0.473, 0.473, 0.552,
    0.495, 0.723, 0.549, 0.476, 0.474, 0.544, 0.489, 0.746,
    0.551, 0.478, 0.480, 0.535, 0.484, 0.766, 0.548, 0.477,
    0.488, 0.527, 0.494, 0.781, 0.550, 0.473, 0.496, 0.522,
    0.511, 0.789, 0.562, 0.469, 0.501, 0.522, 0.530, 0.801,
    0.586, 0.465, 0.504, 0.525, 0.548, 0.816, 0.626, 0.462,
    0.500, 0.526, 0.569, 0.829, 0.676, 0.462, 0.494, 0.527,
    0.595, 0.830, 0.719, 0.466, 0.491, 0.526, 0.622, 0.818,
    0.737, 0.473, 0.496, 0.522, 0.639, 0.796, 0.725, 0.483,
    0.507, 0.514, 0.637, 0.765, 0.695, 0.496, 0.515, 0.504,
    0.619, 0.725, 0.662, 0.508, 0.515, 0.494, 0.599, 0.689,
    0.630, 0.518, 0.511, 0.488, 0.584, 0.661, 0.604, 0.524,
    0.507, 0.485, 0.573, 0.637, 0.586, 0.527, 0.503, 0.484,
    0.560, 0.621, 0.578, 0.526, 0.500, 0.482, 0.546, 0.611,
    0.576, 0.523, 0.498, 0.480, 0.536, 0.605, 0.577, 0.520,
    0.498, 0.479, 0.532, 0.607, 0.580, 0.517, 0.499, 0.479,
    0.532, 0.615, 0.584, 0.515, 0.501, 0.480, 0.534, 0.627,
    0.588, 0.513, 0.501, 0.480, 0.531, 0.636, 0.594, 0.513,
    0.501, 0.479, 0.526, 0.641, 0.599, 0.514, 0.501, 0.478,
    0.525, 0.637, 0.602, 0.515, 0.501, 0.478, 0.528, 0.625,
    0.600, 0.517, 0.503, 0.477, 0.530, 0.609, 0.593, 0.520,
    0.504, 0.476, 0.528, 0.592, 0.584, 0.521, 0.504, 0.475,
    0.524, 0.575, 0.578, 0.522, 0.503, 0.473, 0.517, 0.560,
    0.575, 0.521, 0.501, 0.473, 0.509, 0.548, 0.576, 0.521,
    0.499, 0.473, 0.500, 0.534, 0.583, 0.519, 0.497, 0.474,
    0.493, 0.521, 0.592, 0.518, 0.496, 0.475, 0.488, 0.512,
    0.594, 0.517, 0.496, 0.476, 0.488, 0.504, 0.595, 0.515,
    0.496, 0.478, 0.492, 0.498, 0.596, 0.514, 0.496, 0.479,
    0.495, 0.490, 0.601, 0.513, 0.496, 0.481, 0.495, 0.482,
    0.607, 0.511, 0.495, 0.482, 0.493, 0.478, 0.614, 0.510,
    0.493, 0.483, 0.491, 0.471, 0.623, 0.510, 0.492, 0.484,
    0.485, 0.467, 0.628, 0.509, 0.492, 0.485, 0.479, 0.466,
    0.632, 0.509, 0.492, 0.485, 0.474, 0.460, 0.636, 0.509,
    0.493, 0.485, 0.468, 0.456, 0.638, 0.509, 0.495, 0.486,
    0.463, 0.454, 0.639, 0.510, 0.496, 0.486, 0.463, 0.449,
    0.640, 0.510, 0.497, 0.487, 0.463, 0.446, 0.639, 0.511,
    0.500, 0.488, 0.468, 0.444, 0.636, 0.511, 0.503, 0.488,
    0.470, 0.440, 0.634, 0.511, 0.506, 0.489, 0.471, 0.437,
    0.629, 0.511, 0.510, 0.489, 0.477, 0.432, 0.629, 0.510,
    0.512, 0.488, 0.484, 0.418, 0.633, 0.509, 0.514, 0.488,
    0.497, 0.399, 0.629, 0.510, 0.516, 0.489, 0.520, 0.381,
    0.618, 0.510, 0.517, 0.490, 0.542, 0.369, 0.606, 0.509,
    0.515, 0.490, 0.550, 0.369, 0.597, 0.506, 0.512, 0.489,
    0.537, 0.382, 0.593, 0.503, 0.506, 0.485, 0.501, 0.414,
    0.597, 0.499, 0.497, 0.479, 0.464, 0.427, 0.612, 0.497,
    0.489, 0.475, 0.436, 0.430, 0.629, 0.498, 0.486, 0.476,
    0.429, 0.431, 0.639, 0.501, 0.487, 0.480, 0.434, 0.428,
    0.638, 0.507, 0.488, 0.485, 0.436, 0.427, 0.628, 0.514,
    0.488, 0.488, 0.441, 0.427, 0.617, 0.518, 0.489, 0.490,
    0.446, 0.424, 0.611, 0.518, 0.493, 0.492, 0.454, 0.425,
    0.604, 0.514, 0.499, 0.493, 0.464, 0.426, 0.596, 0.511,
    0.504, 0.496, 0.476, 0.428, 0.590, 0.508, 0.508, 0.499,
    0.488, 0.433, 0.586, 0.505, 0.510, 0.502, 0.499, 0.438,
    0.585, 0.503, 0.510, 0.506, 0.507, 0.444, 0.585, 0.503,
    0.508, 0.508, 0.515, 0.449, 0.589, 0.502, 0.506, 0.509,
    0.523, 0.451, 0.593, 0.500, 0.505, 0.509, 0.529, 0.453,
    0.596, 0.498, 0.505, 0.508, 0.531, 0.456, 0.599, 0.496,
    0.504, 0.507, 0.531, 0.459, 0.605, 0.494, 0.503, 0.507,
    0.530, 0.461, 0.615, 0.493, 0.501, 0.506, 0.527, 0.466,
    0.623, 0.493, 0.498, 0.506, 0.524, 0.470, 0.629, 0.493,
    0.496, 0.506, 0.522, 0.474, 0.636, 0.494, 0.496, 0.506,
    0.521, 0.476, 0.639, 0.495, 0.496, 0.505, 0.521, 0.476,
    0.641, 0.496, 0.497, 0.504, 0.522, 0.475, 0.640, 0.497,
    0.498, 0.503, 0.524, 0.474, 0.636, 0.498, 0.499, 0.503,
    0.526, 0.473, 0.631, 0.500, 0.499, 0.502, 0.527, 0.472,
    0.625, 0.500, 0.500, 0.502, 0.529, 0.473, 0.619, 0.501,
    0.501, 0.502, 0.530, 0.473, 0.615, 0.500, 0.502, 0.502,
    0.531, 0.472, 0.611, 0.500, 0.503, 0.502, 0.532, 0.473,
    0.607, 0.500, 0.503, 0.502, 0.532, 0.474, 0.605, 0.500,
    0.502, 0.501, 0.529, 0.475, 0.606, 0.499, 0.502, 0.501,
    0.527, 0.476, 0.607, 0.499, 0.501, 0.501, 0.524, 0.475,
    0.612, 0.499, 0.500, 0.500, 0.522, 0.475, 0.617, 0.499,
    0.500, 0.500, 0.520, 0.475, 0.618, 0.499, 0.501, 0.500,
    0.518, 0.475, 0.618, 0.499, 0.500, 0.499, 0.517, 0.474,
    0.619, 0.499, 0.500, 0.499, 0.515, 0.473, 0.620, 0.499,
    0.500, 0.499, 0.516, 0.473, 0.619, 0.499, 0.500, 0.500,
    0.519, 0.475, 0.617, 0.499, 0.500, 0.500, 0.520, 0.475,
    0.617, 0.499, 0.499, 0.500, 0.521, 0.476, 0.616, 0.499,
    0.500, 0.501,
};
float input_data1[714] = {
    0.513, 0.471, 0.788, 0.524, 0.512, 0.510, 0.520, 0.469,
    0.799, 0.527, 0.511, 0.509, 0.525, 0.473, 0.798, 0.530,
    0.509, 0.508, 0.532, 0.475, 0.786, 0.533, 0.506, 0.506,
    0.537, 0.480, 0.771, 0.537, 0.502, 0.503, 0.535, 0.483,
    0.761, 0.543, 0.497, 0.500, 0.530, 0.501, 0.741, 0.547,
    0.495, 0.499, 0.526, 0.518, 0.722, 0.552, 0.494, 0.497,
    0.526, 0.517, 0.707, 0.555, 0.490, 0.497, 0.523, 0.513,
    0.690, 0.555, 0.485, 0.496, 0.524, 0.508, 0.674, 0.555,
    0.481, 0.497, 0.524, 0.497, 0.664, 0.556, 0.479, 0.496,
    0.528, 0.483, 0.646, 0.557, 0.481, 0.495, 0.530, 0.500,
    0.643, 0.558, 0.485, 0.494, 0.527, 0.505, 0.640, 0.558,
    0.487, 0.493, 0.525, 0.510, 0.630, 0.556, 0.488, 0.493,
    0.520, 0.498, 0.630, 0.556, 0.486, 0.494, 0.527, 0.480,
    0.610, 0.556, 0.483, 0.494, 0.533, 0.480, 0.591, 0.555,
    0.481, 0.493, 0.528, 0.489, 0.590, 0.555, 0.478, 0.491,
    0.515, 0.500, 0.580, 0.556, 0.478, 0.489, 0.503, 0.510,
    0.570, 0.558, 0.480, 0.488, 0.499, 0.502, 0.561, 0.560,
    0.483, 0.490, 0.502, 0.490, 0.534, 0.559, 0.486, 0.490,
    0.498, 0.481, 0.525, 0.556, 0.485, 0.489, 0.492, 0.477,
    0.518, 0.554, 0.483, 0.488, 0.479, 0.475, 0.510, 0.553,
    0.482, 0.487, 0.470, 0.470, 0.491, 0.554, 0.483, 0.487,
    0.465, 0.463, 0.465, 0.555, 0.484, 0.486, 0.457, 0.454,
    0.450, 0.554, 0.485, 0.486, 0.450, 0.458, 0.427, 0.552,
    0.486, 0.486, 0.442, 0.452, 0.410, 0.550, 0.485, 0.486,
    0.437, 0.451, 0.381, 0.549, 0.484, 0.487, 0.426, 0.457,
    0.359, 0.546, 0.483, 0.488, 0.410, 0.467, 0.331, 0.543,
    0.483, 0.489, 0.401, 0.469, 0.299, 0.540, 0.482, 0.490,
    0.397, 0.468, 0.269, 0.536, 0.478, 0.490, 0.400, 0.465,
    0.233, 0.532, 0.473, 0.491, 0.409, 0.476, 0.217, 0.525,
    0.466, 0.492, 0.407, 0.485, 0.199, 0.516, 0.459, 0.494,
    0.397, 0.468, 0.197, 0.505, 0.455, 0.494, 0.381, 0.454,
    0.222, 0.496, 0.457, 0.496, 0.386, 0.438, 0.256, 0.488,
    0.467, 0.499, 0.403, 0.424, 0.293, 0.484, 0.482, 0.503,
    0.429, 0.413, 0.322, 0.482, 0.499, 0.506, 0.446, 0.405,
    0.359, 0.480, 0.513, 0.507, 0.460, 0.401, 0.392, 0.477,
    0.522, 0.507, 0.470, 0.398, 0.420, 0.475, 0.526, 0.507,
    0.482, 0.402, 0.437, 0.474, 0.527, 0.506, 0.479, 0.403,
    0.455, 0.474, 0.524, 0.504, 0.479, 0.406, 0.461, 0.474,
    0.520, 0.503, 0.477, 0.409, 0.459, 0.474, 0.515, 0.502,
    0.475, 0.411, 0.459, 0.473, 0.510, 0.501, 0.471, 0.415,
    0.457, 0.471, 0.507, 0.500, 0.470, 0.419, 0.456, 0.470,
    0.504, 0.499, 0.470, 0.418, 0.455, 0.469, 0.504, 0.497,
    0.470, 0.413, 0.455, 0.467, 0.505, 0.495, 0.473, 0.411,
    0.458, 0.465, 0.508, 0.493, 0.475, 0.409, 0.461, 0.464,
    0.512, 0.490, 0.477, 0.413, 0.467, 0.464, 0.517, 0.487,
    0.476, 0.427, 0.476, 0.464, 0.522, 0.483, 0.471, 0.457,
    0.489, 0.464, 0.527, 0.481, 0.448, 0.481, 0.500, 0.465,
    0.529, 0.481, 0.427, 0.493, 0.505, 0.465, 0.530, 0.484,
    0.418, 0.506, 0.513, 0.462, 0.528, 0.492, 0.413, 0.506,
    0.532, 0.457, 0.525, 0.500, 0.421, 0.488, 0.548, 0.451,
    0.523, 0.507, 0.438, 0.471, 0.557, 0.448, 0.523, 0.510,
    0.451, 0.451, 0.570, 0.448, 0.523, 0.511, 0.463, 0.435,
    0.578, 0.449, 0.524, 0.509, 0.475, 0.423, 0.582, 0.451,
    0.526, 0.505, 0.479, 0.424, 0.584, 0.454, 0.529, 0.501,
    0.474, 0.443, 0.587, 0.459, 0.531, 0.499, 0.477, 0.468,
    0.591, 0.463, 0.532, 0.499, 0.469, 0.503, 0.601, 0.468,
    0.531, 0.501, 0.455, 0.529, 0.623, 0.473, 0.528, 0.507,
    0.459, 0.525, 0.650, 0.472, 0.523, 0.514, 0.475, 0.508,
    0.669, 0.464, 0.520, 0.519, 0.486, 0.492, 0.689, 0.458,
    0.517, 0.523, 0.504, 0.471, 0.707, 0.454, 0.509, 0.524,
    0.516, 0.446, 0.730, 0.453, 0.502, 0.521, 0.522, 0.431,
    0.736, 0.456, 0.497, 0.515, 0.523, 0.427, 0.724, 0.464,
    0.495, 0.509, 0.523, 0.424, 0.702, 0.473, 0.497, 0.503,
    0.520, 0.425, 0.677, 0.481, 0.502, 0.498, 0.516, 0.435,
    0.652, 0.488, 0.509, 0.495, 0.511, 0.456, 0.628, 0.493,
    0.516, 0.493, 0.501, 0.477, 0.616, 0.496, 0.521, 0.491,
    0.489, 0.503, 0.613, 0.497, 0.524, 0.491, 0.475, 0.521,
    0.617, 0.496, 0.524, 0.493, 0.467, 0.508, 0.637, 0.495,
    0.521, 0.498, 0.481, 0.480, 0.648, 0.491, 0.520, 0.503,
    0.504, 0.470, 0.641, 0.486, 0.520, 0.508, 0.514, 0.458,
    0.639, 0.480, 0.518, 0.508, 0.516, 0.431, 0.639, 0.478,
    0.513, 0.505, 0.516, 0.435, 0.652, 0.478, 0.506, 0.503,
    0.522, 0.446, 0.591, 0.479, 0.503, 0.501, 0.507, 0.430,
    0.641, 0.481, 0.488, 0.496, 0.483, 0.437, 0.675, 0.486,
    0.486, 0.495, 0.487, 0.459, 0.655, 0.487, 0.498, 0.495,
    0.494, 0.471, 0.633, 0.488, 0.507, 0.493, 0.474, 0.481,
    0.662, 0.488, 0.507, 0.494, 0.468, 0.498, 0.648, 0.485,
    0.507, 0.498, 0.379, 0.409, 0.811, 0.483, 0.501, 0.501,
    0.466, 0.375, 0.974, 0.493, 0.505, 0.511, 0.579, 0.484,
    0.803, 0.501, 0.527, 0.517, 0.556, 0.489, 0.691, 0.499,
    0.526, 0.509, 0.558, 0.443, 0.604, 0.494, 0.514, 0.496,
    0.537, 0.415, 0.607, 0.499, 0.494, 0.492, 0.537, 0.457,
    0.582, 0.503, 0.486, 0.497, 0.516, 0.468, 0.565, 0.501,
    0.481, 0.497, 0.504, 0.467, 0.549, 0.501, 0.478, 0.495,
    0.494, 0.459, 0.537, 0.501, 0.478, 0.494, 0.485, 0.457,
    0.535, 0.500, 0.480, 0.496, 0.467, 0.465, 0.562, 0.498,
    0.485, 0.497, 0.454, 0.468, 0.610, 0.497, 0.494, 0.498,
    0.449, 0.465, 0.652, 0.496, 0.505, 0.498, 0.454, 0.470,
    0.662, 0.496, 0.516, 0.499, 0.468, 0.470, 0.670, 0.496,
    0.519, 0.500,
};
float aX[119] = {
    -0.218, -0.076, -0.196, -0.112, 0.696, 0.884, 0.278, 0.228,
    0.491, 0.342, 0.197, 0.167, 0.176, 0.139, 0.058, -0.022,
    -0.044, -0.056, -0.077, 0.002, 0.064, 0.083, 0.046, -0.024,
    -0.071, -0.071, -0.070, -0.171, -0.313, -0.361, -0.393, -0.384,
    -0.397, -0.430, -0.422, -0.413, -0.559, -0.561, -0.420, -0.390,
    -0.490, -0.520, -0.617, -0.688, -0.737, -0.833, -1.102, -1.103,
    -1.015, -1.078, -0.959, -0.697, -0.709, -0.749, -0.671, -0.525,
    -0.413, -0.343, -0.317, -0.290, -0.303, -0.317, -0.345, -0.421,
    -0.538, -0.589, -0.675, -0.664, -0.582, -0.501, -0.401, -0.299,
    -0.274, -0.354, -0.312, -0.324, -0.505, -0.646, -0.626, -0.666,
    -0.582, -0.221, -0.057, -0.030, -0.223, -0.573, -0.341, -0.093,
    -0.377, -0.382, -0.118, 0.140, 0.269, 0.314, 0.281, 0.229,
    0.276, 0.271, 0.199, 0.069, -0.038, 0.026, 0.140, 0.237,
    0.321, 0.388, 0.346, 0.278, 0.119, 0.157, 0.128, 0.085,
    -0.002, -0.054, -0.180, -0.274, -0.305, -0.292, -0.286,
};

float aY[119] = {
    -0.370, -0.057, 0.003, -0.212, -0.683, -0.715, -0.373, -0.334,
    -0.369, -0.357, -0.332, -0.425, -0.519, -0.511, -0.334, -0.205,
    -0.167, -0.047, -0.123, -0.234, -0.333, -0.375, -0.335, -0.463,
    -0.542, -0.499, -0.563, -0.485, -0.318, -0.473, -0.545, -0.518,
    -0.499, -0.475, -0.525, -0.546, -0.542, -0.684, -0.738, -0.658,
    -0.592, -0.644, -0.657, -0.587, -0.586, -0.536, -0.510, -0.666,
    -0.486, -0.504, -0.485, -0.303, -0.475, -0.461, -0.567, -0.716,
    -0.784, -0.736, -0.754, -0.783, -0.826, -0.898, -0.912, -0.825,
    -0.757, -0.737, -0.755, -0.747, -0.716, -0.761, -0.760, -0.726,
    -0.718, -0.753, -0.417, -0.127, -0.218, -0.201, -0.136, -0.069,
    -0.144, -0.305, -0.285, -0.235, -0.250, -0.280, -0.007, 0.261,
    0.262, 0.136, 0.186, 0.229, 0.107, -0.064, -0.117, -0.262,
    -0.274, -0.203, -0.164, 0.062, 0.195, 0.272, 0.110, -0.073,
    -0.357, -0.302, -0.398, -0.262, -0.318, -0.320, -0.328, -0.348,
    -0.303, -0.253, -0.226, -0.290, -0.389, -0.380, -0.390,
};

float aZ[119] = {
    2.201, 2.112, 1.959, 1.565, 0.453, 0.795, 1.635, 1.596,
    1.303, 1.404, 1.433, 1.278, 1.049, 0.892, 0.805, 0.713,
    0.645, 0.707, 0.730, 0.744, 0.818, 0.889, 0.972, 0.937,
    0.888, 0.859, 0.710, 0.539, 0.417, 0.263, 0.255, 0.336,
    0.314, 0.345, 0.358, 0.370, 0.422, 0.342, 0.187, 0.224,
    0.164, 0.027, -0.200, -0.291, -0.485, -0.716, -0.573, -1.043,
    -1.350, -1.312, -1.352, -1.385, -1.260, -1.149, -0.951, -0.790,
    -0.623, -0.430, -0.211, -0.182, -0.168, -0.166, -0.248, -0.299,
    -0.357, -0.413, -0.375, -0.424, -0.464, -0.457, -0.453, -0.409,
    -0.204, -0.033, -0.005, 0.052, 0.021, -0.027, 0.027, 0.350,
    0.767, 0.461, 0.347, 0.551, 0.834, 1.101, 0.976, 0.996,
    0.946, 1.195, 1.361, 1.643, 1.902, 2.119, 2.162, 2.047,
    1.810, 1.495, 1.370, 1.189, 0.865, 1.018, 1.227, 1.312,
    1.611, 2.123, 1.983, 1.947, 1.867, 1.674, 1.392, 1.269,
    1.116, 0.983, 0.821, 0.619, 0.771, 0.820, 0.890,
};

float gX[119] = {
    126.343, 60.791, -2.563, -70.496, -98.999, -6.958, 111.084, 169.861,
    189.392, 184.998, 182.861, 192.505, 197.571, 190.674, 180.481, 171.387,
    159.973, 150.024, 143.921, 138.000, 125.122, 107.422, 97.534, 96.741,
    98.999, 106.262, 121.887, 137.329, 142.822, 137.878, 126.221, 114.868,
    120.178, 132.874, 139.893, 141.541, 147.339, 158.936, 155.884, 144.409,
    137.329, 134.277, 135.071, 131.592, 132.996, 137.146, 150.818, 199.646,
    191.956, 133.972, 91.919, 40.771, 7.996, -9.583, -23.315, -36.987,
    -54.016, -73.181, -78.491, -71.594, -64.819, -58.228, -55.054, -58.167,
    -68.726, -87.891, -106.750, -118.408, -129.272, -145.569, -160.522, -173.950,
    -183.411, -178.101, -164.978, -158.752, -167.664, -176.331, -192.322, -213.745,
    -216.980, -215.393, -217.834, -221.191, -215.515, -203.369, -197.998, -193.909,
    -199.707, -218.018, -230.774, -252.625, -263.550, -245.117, -188.110, -119.385,
    -61.401, -23.376, -7.996, -6.897, -31.189, -73.792, -107.422, -123.474,
    -132.446, -120.178, -87.219, -65.491, -32.227, 7.751, 29.419, 36.865,
    40.649, 38.391, 25.940, 14.587, 2.747, -10.681, -16.663,
};

float gY[119] = {
    126.343, 60.791, -2.563, -70.496, -98.999, -6.958, 111.084, 169.861,
    189.392, 184.998, 182.861, 192.505, 197.571, 190.674, 180.481, 171.387,
    159.973, 150.024, 143.921, 138.000, 125.122, 107.422, 97.534, 96.741,
    98.999, 106.262, 121.887, 137.329, 142.822, 137.878, 126.221, 114.868,
    120.178, 132.874, 139.893, 141.541, 147.339, 158.936, 155.884, 144.409,
    137.329, 134.277, 135.071, 131.592, 132.996, 137.146, 150.818, 199.646,
    191.956, 133.972, 91.919, 40.771, 7.996, -9.583, -23.315, -36.987,
    -54.016, -73.181, -78.491, -71.594, -64.819, -58.228, -55.054, -58.167,
    -68.726, -87.891, -106.750, -118.408, -129.272, -145.569, -160.522, -173.950,
    -183.411, -178.101, -164.978, -158.752, -167.664, -176.331, -192.322, -213.745,
    -216.980, -215.393, -217.834, -221.191, -215.515, -203.369, -197.998, -193.909,
    -199.707, -218.018, -230.774, -252.625, -263.550, -245.117, -188.110, -119.385,
    -61.401, -23.376, -7.996, -6.897, -31.189, -73.792, -107.422, -123.474,
    -132.446, -120.178, -87.219, -65.491, -32.227, 7.751, 29.419, 36.865,
    40.649, 38.391, 25.940, 14.587, 2.747, -10.681, -16.663,
};

float gZ[119] = {
    126.343, 60.791, -2.563, -70.496, -98.999, -6.958, 111.084, 169.861,
    189.392, 184.998, 182.861, 192.505, 197.571, 190.674, 180.481, 171.387,
    159.973, 150.024, 143.921, 138.000, 125.122, 107.422, 97.534, 96.741,
    98.999, 106.262, 121.887, 137.329, 142.822, 137.878, 126.221, 114.868,
    120.178, 132.874, 139.893, 141.541, 147.339, 158.936, 155.884, 144.409,
    137.329, 134.277, 135.071, 131.592, 132.996, 137.146, 150.818, 199.646,
    191.956, 133.972, 91.919, 40.771, 7.996, -9.583, -23.315, -36.987,
    -54.016, -73.181, -78.491, -71.594, -64.819, -58.228, -55.054, -58.167,
    -68.726, -87.891, -106.750, -118.408, -129.272, -145.569, -160.522, -173.950,
    -183.411, -178.101, -164.978, -158.752, -167.664, -176.331, -192.322, -213.745,
    -216.980, -215.393, -217.834, -221.191, -215.515, -203.369, -197.998, -193.909,
    -199.707, -218.018, -230.774, -252.625, -263.550, -245.117, -188.110, -119.385,
    -61.401, -23.376, -7.996, -6.897, -31.189, -73.792, -107.422, -123.474,
    -132.446, -120.178, -87.219, -65.491, -32.227, 7.751, 29.419, 36.865,
    40.649, 38.391, 25.940, 14.587, 2.747, -10.681, -16.663,
};

void setup() {
  Serial.begin(115200);

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(gesture_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();


  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  
    // input tensor
    for (int i = 0; i < 714; i++){
        tflInputTensor->data.f[i] = input_data[i];
    }
// Run inferencing
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }
    // Loop through the output tensor values from the model
    for (int i = 0; i < 2; i++) {
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(tflOutputTensor->data.f[i], 6);
    }
    Serial.println(); 

    delay(2000);
        // input tensor
    for (int i = 0; i < 6*numSamples; i++){
        tflInputTensor->data.f[i] = input_data1[i];
    }
    // Run inferencing
    invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }
    // Loop through the output tensor values from the model
    for (int i = 0; i < 2; i++) {
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(tflOutputTensor->data.f[i], 6);
    }
    delay(2000);
}

