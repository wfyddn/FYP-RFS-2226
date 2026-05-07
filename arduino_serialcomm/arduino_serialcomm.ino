// ================================================================
// REAL-TIME TRAFFIC LIGHT SYSTEM - ARDUINO UNO
// Pedestrian + Vehicle Traffic Lights Only
// ================================================================

// ── Pin Configuration ────────────────────────────────────────────
const int PED_RED_PIN    = 2;
const int PED_GREEN_PIN  = 3;
const int VEH_RED_PIN    = 4;
const int VEH_YELLOW_PIN = 5;
const int VEH_GREEN_PIN  = 6;

// ── Timing Configuration ─────────────────────────────────────────
const unsigned long YELLOW_DURATION = 5000;    // 5s yellow / buffer
const unsigned long SIGNAL_TIMEOUT  = 10000;   // 10s watchdog

// ── State Machine ─────────────────────────────────────────────────
enum TrafficState {
  STATE_VEHICLE_GREEN,       // Normal: vehicles moving
  STATE_VEHICLE_YELLOW,      // Signal '1' → yellow before red
  STATE_VEHICLE_RED,         // Pedestrian crossing (GREEN)
  STATE_PEDESTRIAN_BUFFER    // Signal '0' → 5s buffer before vehicle green
};

TrafficState  currentState   = STATE_VEHICLE_GREEN;
unsigned long stateStartTime = 0;
unsigned long lastSignalTime = 0;
char          lastSignal     = '0';

// ================================================================
// HELPER FUNCTIONS
// ================================================================

void setPedestrianLight(const char* color) {
  if (strcmp(color, "GREEN") == 0) {
    digitalWrite(PED_RED_PIN,   LOW);
    digitalWrite(PED_GREEN_PIN, HIGH);
    Serial.println("[Pedestrian] GREEN");
  } else {
    digitalWrite(PED_RED_PIN,   HIGH);
    digitalWrite(PED_GREEN_PIN, LOW);
    Serial.println("[Pedestrian] RED");
  }
}

void setVehicleLight(const char* color) {
  digitalWrite(VEH_RED_PIN,    LOW);
  digitalWrite(VEH_YELLOW_PIN, LOW);
  digitalWrite(VEH_GREEN_PIN,  LOW);

  if (strcmp(color, "GREEN") == 0) {
    digitalWrite(VEH_GREEN_PIN, HIGH);
    Serial.println("[Vehicle] GREEN");
  } else if (strcmp(color, "YELLOW") == 0) {
    digitalWrite(VEH_YELLOW_PIN, HIGH);
    Serial.println("[Vehicle] YELLOW");
  } else {
    digitalWrite(VEH_RED_PIN, HIGH);
    Serial.println("[Vehicle] RED");
  }
}

void enterState(TrafficState newState) {
  currentState   = newState;
  stateStartTime = millis();

  switch (newState) {

    case STATE_VEHICLE_GREEN:
      // ── Normal flow: vehicles moving ─────────────────────────
      Serial.println("\n[State] VEHICLE GREEN");
      setPedestrianLight("RED");
      setVehicleLight("GREEN");
      break;

    case STATE_VEHICLE_YELLOW:
      // ── Signal '1': warn vehicles before pedestrian crossing ─
      Serial.println("\n[State] VEHICLE YELLOW (5s)");
      setPedestrianLight("RED");     // Pedestrian stays RED during yellow
      setVehicleLight("YELLOW");
      break;

    case STATE_VEHICLE_RED:
      // ── Yellow done: pedestrian can cross now ─────────────────
      Serial.println("\n[State] VEHICLE RED - PEDESTRIAN CROSSING");
      setPedestrianLight("GREEN");   // Pedestrian GREEN only after yellow
      setVehicleLight("RED");
      break;

    case STATE_PEDESTRIAN_BUFFER:
      // ── Signal '0': crossing done, buffer before vehicle green
      Serial.println("\n[State] PEDESTRIAN BUFFER (5s)");
      setPedestrianLight("RED");     // Pedestrian RED immediately
      setVehicleLight("RED");        // Vehicle stays RED during buffer
      break;
  }
}

// ================================================================
// SETUP
// ================================================================

void setup() {
  pinMode(PED_RED_PIN,    OUTPUT);
  pinMode(PED_GREEN_PIN,  OUTPUT);
  pinMode(VEH_RED_PIN,    OUTPUT);
  pinMode(VEH_YELLOW_PIN, OUTPUT);
  pinMode(VEH_GREEN_PIN,  OUTPUT);

  Serial.begin(9600);

  enterState(STATE_VEHICLE_GREEN);
  lastSignalTime = millis();

  Serial.println("================================================");
  Serial.println("  TRAFFIC LIGHT SYSTEM - ARDUINO READY");
  Serial.println("================================================");
  Serial.println("  '1' = Pedestrians → Yellow(5s) → Veh RED + Ped GREEN");
  Serial.println("  '0' = Crossing done → Ped RED → Buffer(5s) → Veh GREEN");
  Serial.println("================================================\n");
}

// ================================================================
// MAIN LOOP
// ================================================================

void loop() {
  unsigned long now = millis();

  // ── Read Serial Signal from Jetson ─────────────────────────────
  if (Serial.available() > 0) {
    char signal = Serial.read();

    if (signal == '1' || signal == '0') {
      lastSignal     = signal;
      lastSignalTime = now;

      Serial.print("[Serial] Received: ");
      Serial.println(signal);

      if (signal == '1') {
        // Only trigger if currently in vehicle green
        if (currentState == STATE_VEHICLE_GREEN) {
          enterState(STATE_VEHICLE_YELLOW);
        }
      }

      else if (signal == '0') {
        // Only trigger if currently in pedestrian crossing
        if (currentState == STATE_VEHICLE_RED) {
          enterState(STATE_PEDESTRIAN_BUFFER);
        }
      }
    }
  }

  // ── State Machine Transitions ───────────────────────────────────
  switch (currentState) {

    case STATE_VEHICLE_GREEN:
      // Waiting for '1' from Jetson
      break;

    case STATE_VEHICLE_YELLOW:
      // After 5s → vehicle red, pedestrian green
      if (now - stateStartTime >= YELLOW_DURATION) {
        enterState(STATE_VEHICLE_RED);
      }
      break;

    case STATE_VEHICLE_RED:
      // Waiting for '0' from Jetson (Python countdown finished)
      break;

    case STATE_PEDESTRIAN_BUFFER:
      // After 5s buffer → back to vehicle green
      if (now - stateStartTime >= YELLOW_DURATION) {
        enterState(STATE_VEHICLE_GREEN);
      }
      break;
  }

  // ── Watchdog: safety fallback if Jetson stops sending ──────────
  if (currentState != STATE_VEHICLE_GREEN &&
      (now - lastSignalTime > SIGNAL_TIMEOUT)) {
    Serial.println("\n[Watchdog] Timeout! Forcing safe state.");
    lastSignal = '0';
    enterState(STATE_PEDESTRIAN_BUFFER);
  }
}