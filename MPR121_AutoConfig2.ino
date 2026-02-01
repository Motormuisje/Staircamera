/*********************************************************
MPR121 Autoconfig + Touch/Release (ESP32) + WiFi UDP events
- behoudt autoconfig flow zoals jouw werkende sketch
- SDA=32, SCL=33
- I2C addr = 0x5B
**********************************************************/
#include <Wire.h>
#include <Adafruit_MPR121.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#ifndef _BV
#define _BV(bit) (1 << (bit))
#endif

// ====== ESP32 I2C ======
static const int I2C_SDA = 32;
static const int I2C_SCL = 33;
static const uint8_t MPR121_ADDR = 0x5B;

// ====== WiFi/UDP (wireless) ======
const char* WIFI_SSID = "iPhone";
const char* WIFI_PASS = "gunwifiaub";

// IP van PC/Jetson die UDP ontvangt:
IPAddress PC_IP(172, 20, 10, 4);
const uint16_t UDP_PORT = 4213;

WiFiUDP udp;

// ====== MPR121 ======
Adafruit_MPR121 cap = Adafruit_MPR121();
uint16_t lasttouched = 0;
uint16_t currtouched = 0;

void dump_regs() {
  Serial.println("========================================");
  Serial.println("CHAN 00 01 02 03 04 05 06 07 08 09 10 11");
  Serial.println("     -- -- -- -- -- -- -- -- -- -- -- --");
  // CDC
  Serial.print("CDC: ");
  for (int chan = 0; chan < 12; chan++) {
    uint8_t reg = cap.readRegister8(0x5F + chan);
    if (reg < 10) Serial.print(" ");
    Serial.print(reg);
    Serial.print(" ");
  }
  Serial.println();
  // CDT
  Serial.print("CDT: ");
  for (int chan = 0; chan < 6; chan++) {
    uint8_t reg = cap.readRegister8(0x6C + chan);
    uint8_t cdtx = reg & 0b111;
    uint8_t cdty = (reg >> 4) & 0b111;
    if (cdtx < 10) Serial.print(" ");
    Serial.print(cdtx);
    Serial.print(" ");
    if (cdty < 10) Serial.print(" ");
    Serial.print(cdty);
    Serial.print(" ");
  }
  Serial.println();
  Serial.println("========================================");
}

void send_udp(const String& msg) {
  // Stuur alleen als WiFi up is
  if (WiFi.status() != WL_CONNECTED) return;
  udp.beginPacket(PC_IP, UDP_PORT);
  udp.write((const uint8_t*)msg.c_str(), msg.length());
  udp.endPacket();
}

void wifi_connect_non_blocking(uint32_t maxMs) {
  if (WiFi.status() == WL_CONNECTED) return;

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint32_t t0 = millis();
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED && (millis() - t0) < maxMs) {
    delay(200);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi OK, IP=");
    Serial.println(WiFi.localIP());
    udp.begin(UDP_PORT); // lokale bind (mag)
  } else {
    Serial.println("WiFi not connected (continuing offline, will retry later)");
  }
}

uint32_t lastWifiRetry = 0;

void setup() {
  delay(1000);
  Serial.begin(115200);
  delay(200);

  Serial.println("MPR121 Autoconfig + Touch/Release + WiFi UDP (ESP32)");
  Serial.println("startup `Wire`");

  // --- CRUCIAAL OP ESP32: pins meegeven ---
  Wire.begin(I2C_SDA, I2C_SCL);

  Serial.println("startup `Wire` done.");
  delay(100);

  Serial.println("cap.begin..");
  if (!cap.begin(MPR121_ADDR, &Wire)) {
    Serial.println("MPR121 not found, check wiring/address?");
    while (1) delay(10);
  }
  Serial.println("MPR121 found!");
  delay(100);

  // --- EXACT zoals jouw werkende autoconfig flow ---
  Serial.println("Initial CDC/CDT values:");
  dump_regs();

  cap.setAutoconfig(true);

  Serial.println("After autoconfig CDC/CDT values:");
  dump_regs();

  // WiFi proberen (maar niet eindeloos blokkeren)
  wifi_connect_non_blocking(6000);

  Serial.println("READY");
}

void loop() {
  // WiFi af en toe opnieuw proberen zonder je touch loop te “killen”
  if (WiFi.status() != WL_CONNECTED && (millis() - lastWifiRetry) > 5000) {
    lastWifiRetry = millis();
    wifi_connect_non_blocking(1500);
  }

  // --- JOUW BEWEZEN TOUCH/RELEASE LOGICA ---
  currtouched = cap.touched();

  for (uint8_t i = 0; i < 12; i++) {
    if ((currtouched & _BV(i)) && !(lasttouched & _BV(i))) {
      // touched
      Serial.print(i); Serial.println(" touched");

      // UDP event (PC kan hierop beepen/loggen)
      // voorbeeld: "TOUCH 3 123456"
      String msg = "TOUCH " + String(i) + " " + String(millis());
      send_udp(msg);
    }

    if (!(currtouched & _BV(i)) && (lasttouched & _BV(i))) {
      // released
      Serial.print(i); Serial.println(" released");

      String msg = "RELEASE " + String(i) + " " + String(millis());
      send_udp(msg);
    }
  }

  lasttouched = currtouched;
  delay(10);
}
