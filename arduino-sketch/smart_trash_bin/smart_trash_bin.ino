/**
 * Smart Trash Bin – Servo Controller
 * ====================================
 * Controls 4 servo motors distributed across 2 lids.
 *
 *   Lid 1  →  Servo 0 (pin 9)  &  Servo 1 (pin 10)   [opposite sides]
 *   Lid 2  →  Servo 2 (pin 11) &  Servo 3 (pin 3)    [opposite sides]
 *
 * Because each pair is mounted on opposite sides of the lid, opening
 * requires the two servos to rotate in mirror directions:
 *   Servo 0 / 2  →  DEFAULT 45°  |  OPEN 90°
 *   Servo 1 / 3  →  DEFAULT 45°  |  OPEN  0°
 *
 * Serial Protocol (USB, 9600 baud)
 * ---------------------------------
 * Send one of the following commands (case-insensitive), terminated with '\n':
 *
 *   OPEN 1    → opens lid 1
 *   OPEN 2    → opens lid 2
 *   OPEN ALL  → opens both lids
 *   CLOSE 1   → closes lid 1
 *   CLOSE 2   → closes lid 2
 *   CLOSE ALL → closes both lids
 *
 * Every command receives a single-line response ending with '\n':
 *   OK <message>   – success
 *   ERR <message>  – unknown command
 *
 * Wiring
 * -------
 *   Servo 0 signal → Pin  9   (Lid 1, side A)
 *   Servo 1 signal → Pin 10   (Lid 1, side B – mirror)
 *   Servo 2 signal → Pin 11   (Lid 2, side A)
 *   Servo 3 signal → Pin  3   (Lid 2, side B – mirror)
 *   All servo GND  → Arduino GND
 *   All servo VCC  → External 5 V supply (do NOT use the Arduino 5 V rail)
 */

#include <Servo.h>

// ── Pin assignment ─────────────────────────────────────────────────────────
const uint8_t SERVO_PIN[4] = {9, 10, 11, 3};

// ── Servo positions ────────────────────────────────────────────────────────
//   Index 0 / 2  = "side A"  servos  →  mirrors of side B
//   Index 1 / 3  = "side B"  servos
const int POS_DEFAULT_A = 45;   // closed position for side-A servos
const int POS_DEFAULT_B = 45;   // closed position for side-B servos
const int POS_OPEN_A    = 90;   // open position  for side-A servos
const int POS_OPEN_B    =  0;   // open position  for side-B servos (mirror)

// ── Servo objects ──────────────────────────────────────────────────────────
Servo    servos[4];
String   inputBuffer = "";

// ══════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);

  for (uint8_t i = 0; i < 4; i++) {
    servos[i].attach(SERVO_PIN[i]);
  }

  closeLid(1);
  closeLid(2);
  delay(500);

  Serial.println("READY");
}

// ══════════════════════════════════════════════════════════════════════════
void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        handleCommand(inputBuffer);
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════
void handleCommand(const String& cmd) {
  String upper = cmd;
  upper.toUpperCase();

  if (upper.startsWith("OPEN")) {
    String target = upper.substring(4);
    target.trim();

    if (target == "1") {
      openLid(1);
      Serial.println("OK LID_1 OPEN");
    } else if (target == "2") {
      openLid(2);
      Serial.println("OK LID_2 OPEN");
    } else if (target == "ALL") {
      openLid(1);
      openLid(2);
      Serial.println("OK ALL OPEN");
    } else {
      Serial.println("ERR Usage: OPEN 1 | OPEN 2 | OPEN ALL");
    }

  } else if (upper.startsWith("CLOSE")) {
    String target = upper.substring(5);
    target.trim();

    if (target == "1") {
      closeLid(1);
      Serial.println("OK LID_1 CLOSED");
    } else if (target == "2") {
      closeLid(2);
      Serial.println("OK LID_2 CLOSED");
    } else if (target == "ALL") {
      closeLid(1);
      closeLid(2);
      Serial.println("OK ALL CLOSED");
    } else {
      Serial.println("ERR Usage: CLOSE 1 | CLOSE 2 | CLOSE ALL");
    }

  } else {
    Serial.println("ERR Unknown command. Use OPEN or CLOSE.");
  }
}

// ══════════════════════════════════════════════════════════════════════════
// lid: 1 or 2
void openLid(uint8_t lid) {
  uint8_t a = (lid - 1) * 2;       // side-A servo index  (0 or 2)
  uint8_t b = a + 1;                // side-B servo index  (1 or 3)
  servos[a].write(POS_OPEN_A);
  servos[b].write(POS_OPEN_B);
}

void closeLid(uint8_t lid) {
  uint8_t a = (lid - 1) * 2;
  uint8_t b = a + 1;
  servos[a].write(POS_DEFAULT_A);
  servos[b].write(POS_DEFAULT_B);
}
