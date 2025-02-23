#ifndef MATRIZ_H
#define MATRIZ_H

#define LED_PIN 7  // Agora LED_PIN é visível em main.c

void drawHeart(int red, int green, int blue);
void npInit(uint pin);
void npClear();
void npWrite();

#endif