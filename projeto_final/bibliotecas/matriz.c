#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "ws2818b.pio.h"
#include "headers/matriz.h"  

#define LED_COUNT 25 // Número de LEDs
#define LED_PIN 7 // Pino onde os LEDs estão conectados

// Estrutura para representar um pixel com componentes RGB
struct pixel_t {
    uint8_t G, R, B;
};
typedef struct pixel_t pixel_t;
typedef pixel_t npLED_t;

npLED_t leds[LED_COUNT]; // Array de LEDs
PIO np_pio;
uint sm;

// Inicializa os LEDs
void npInit(uint pin) {
    uint offset = pio_add_program(pio1, &ws2818b_program);
    np_pio = pio1;
    sm = pio_claim_unused_sm(np_pio, true);

    ws2818b_program_init(np_pio, sm, offset, pin, 800000.f);

    for (uint i = 0; i < LED_COUNT; ++i) {
        leds[i].R = 0;
        leds[i].G = 0;
        leds[i].B = 0;
    }
}

// Define a cor de um LED específico
void npSetLED(const uint index, const uint8_t r, const uint8_t g, const uint8_t b) {
    leds[index].R = r;
    leds[index].G = g;
    leds[index].B = b;
}

// Limpa todos os LEDs (desliga)
void npClear() {
    for (uint i = 0; i < LED_COUNT; ++i)
        npSetLED(i, 0, 0, 0);
}

// Escreve os valores dos LEDs para o hardware
void npWrite() {
    for (uint i = 0; i < LED_COUNT; ++i) {
        pio_sm_put_blocking(np_pio, sm, leds[i].G);
        pio_sm_put_blocking(np_pio, sm, leds[i].R);
        pio_sm_put_blocking(np_pio, sm, leds[i].B);
    }
}

// Calcula o índice de um LED baseado em suas coordenadas x e y
int getIndex(int x, int y) {
    if (y % 2 == 0) {
        return y * 5 + x;
    } else {
        return y * 5 + (4 - x);
    }
}

// Acende um LED específico com a cor fornecida
void acende_leds(int x, int y, int r, int g, int b) {
    int index = getIndex(x, y);
    if (index >= 0 && index < LED_COUNT) {
        npSetLED(index, (r * 255) / 100, (g * 255) / 100, (b * 255) / 100);
    }
}

// Desenha um coração na matriz de LEDs
void drawHeart(int red, int green, int blue) {
    int heart[10][2] = {
        {2, 0}, {1, 1}, {3, 1}, {0, 2}, {4, 2},
        {0, 3}, {4, 3}, {2, 3}, {1, 4}, {3, 4}
    };
    for (int i = 0; i < 10; i++) {
        acende_leds(heart[i][0], heart[i][1], red, green, blue);
    }
    npWrite();
}
