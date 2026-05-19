#include <Arduino.h>

const int PIN_CH_A = 32; // 環境に合わせてピン番号を調整
const int PIN_CH_B = 33;

const unsigned long INTERVAL_MICROS = 5000; // 5000us = 200Hz
unsigned long next_sample_time = 0;
unsigned long sample_count = 0;

void setup() {
    Serial.begin(115200);
    pinMode(PIN_CH_A, INPUT);
    pinMode(PIN_CH_B, INPUT);
    next_sample_time = micros();
}

void loop() {
    // 厳密な時間管理によるビジーウェイト（周期の揺らぎを排除）
    if (micros() >= next_sample_time) {
        int val_A = analogRead(PIN_CH_A);
        int val_B = analogRead(PIN_CH_B);

        // CSVストリーミング形式で出力
        Serial.print(sample_count);
        Serial.print(",");
        Serial.print(val_A);
        Serial.print(",");
        Serial.println(val_B);

        sample_count++;
        next_sample_time += INTERVAL_MICROS;
    }
}
