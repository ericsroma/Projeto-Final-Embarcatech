#ifndef DISPLAY_H
#define DISPLAY_H

#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "inc/ssd1306.h"

void display_info(float freq, const char *cond, struct render_area *frame_area);
void display_menu(struct render_area *frame_area);
void display_wifi(struct render_area *frame_area);
void display_audio(struct render_area *frame_area);
void setup_i2c();

#endif // DISPLAY_H

//display_info(72.5, "Normal", &frame_area);
