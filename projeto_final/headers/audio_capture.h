// audio_capture.h

#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "i2s_mic.pio.h"

void audio_capture_init();
void audio_capture_start();
void audio_capture_stop();
void gpio_irq_handler(uint gpio, uint32_t events);
void audio_capture_loop();

#endif // AUDIO_CAPTURE_H
