#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>  //Download This Library: https://github.com/yoursunny/esp32cam

const char* WIFI_SSID = "WINDOWS-11 6880";             //Set Your Wifi SSID/Name
const char* WIFI_PASS = "vanluc22";  //Set Your Wifi Password

WebServer server(80);


// static auto loRes = esp32cam::Resolution::find(320, 320);
// static auto midRes = esp32cam::Resolution::find(320, 320);
// static auto hiRes = esp32cam::Resolution::find(320, 320);

static auto hiRes = esp32cam::Resolution::find(480, 320);

void serveJpg() {
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                static_cast<int>(frame->size()));

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

// void handleJpgLo() {
//   if (!esp32cam::Camera.changeResolution(loRes)) {
//     Serial.println("SET-LO-RES FAIL");
//   }
//   serveJpg();
// }

void handleJpgHi() {
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }
  serveJpg();
}

// void handleJpgMid() {
//   if (!esp32cam::Camera.changeResolution(midRes)) {
//     Serial.println("SET-MID-RES FAIL");
//   }
//   serveJpg();
// }


void setup() {
  Serial.begin(115200);
  Serial.println();
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(2);
    cfg.setJpeg(80);

    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.print("http://");
  Serial.println(WiFi.localIP());
  // Serial.println("  /cam-lo.jpg");
  Serial.println("  /cam.jpg");
  // Serial.println("  /cam-mid.jpg");

  // server.on("/cam-lo.jpg", handleJpgLo);
  server.on("/cam.jpg", handleJpgHi);
  // server.on("/cam-mid.jpg", handleJpgMid);

  server.begin();
}

void loop() {
  server.handleClient();
}