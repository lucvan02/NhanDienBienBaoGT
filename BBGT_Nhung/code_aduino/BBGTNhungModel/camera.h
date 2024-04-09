#ifndef CAMERA_H
#define CAMERA_H

#include <esp_camera.h>
#include "camera_pins.h"

#define CAMERA_MODEL_AI_THINKER // Define the camera model

// Định nghĩa chất lượng ảnh và độ phân giải
#define CAMERA_FRAME_WIDTH 640
#define CAMERA_FRAME_HEIGHT 480
#define CAMERA_FRAME_QUALITY 10

// Hàm khởi tạo camera
void camera_init() {
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
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = CAMERA_FRAME_WIDTH * CAMERA_FRAME_HEIGHT;
  config.jpeg_quality = CAMERA_FRAME_QUALITY;
  config.fb_count = 2;

  // Khởi tạo camera với cấu hình đã cài đặt
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

#endif // CAMERA_H
