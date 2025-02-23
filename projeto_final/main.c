/*
 * Autor: Eric Senne Roma
 * Data de Criação: 23/02/2025
 * Descrição: Este arquivo contém o código principal para o projeto final no Emabractech (2025).
 *            Trata-se de um sistema que utiliza a Placa BitDogLab para capturar os sons do coração
 *           de um usuário com um microfone INM441, processar esses dados em um computador e exibir informações
 *           visuais em uma matriz de LEDs, um lED RGB e um display OLED.
 */

#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/gpio.h"
#include "headers/display.h"
#include "headers/matriz.h"
#include "headers/led_bpm.h"
#include "headers/data_server.h"
#include "headers/audio_capture.h"

// Definição dos pinos GPIO para os botões e LEDs
#define BUTTON_A 5  // GP5
#define BUTTON_B 6  // GP6
#define LED_R 13
#define LED_G 11
#define LED_B 12

// Variáveis para rastrear o estado dos botões
volatile bool button_b_pressed = false;
volatile bool button_a_pressed = false;

// Definição da área de renderização para o display
struct render_area frame_area = {
    .start_column = 0,
    .end_column = ssd1306_width - 1,
    .start_page = 0,
    .end_page = ssd1306_n_pages - 1
};

// Função de callback para interrupções dos botões
void button_callback(uint gpio, uint32_t events) {
    if (gpio == BUTTON_B) {
        button_b_pressed = true;
    } else if (gpio == BUTTON_A) {
        button_a_pressed = true;
    }
}

// Configuração dos pinos GPIO para os botões e LEDs
void setup_gpio() {
    gpio_init(BUTTON_B);
    gpio_set_dir(BUTTON_B, GPIO_IN);
    gpio_pull_up(BUTTON_B);
    gpio_set_irq_enabled_with_callback(BUTTON_B, GPIO_IRQ_EDGE_FALL, true, &button_callback);

    gpio_init(BUTTON_A);
    gpio_set_dir(BUTTON_A, GPIO_IN);
    gpio_pull_up(BUTTON_A);
    gpio_set_irq_enabled_with_callback(BUTTON_A, GPIO_IRQ_EDGE_FALL, true, &button_callback);

    gpio_init(LED_R);
    gpio_init(LED_G);
    gpio_init(LED_B);
    gpio_set_dir(LED_R, GPIO_OUT);
    gpio_set_dir(LED_G, GPIO_OUT);
    gpio_set_dir(LED_B, GPIO_OUT);
}

// Define a cor do LED RGB
void set_led_color(bool r, bool g, bool b) {
    gpio_put(LED_R, r);
    gpio_put(LED_G, g);
    gpio_put(LED_B, b);
}

// Reseta o estado do sistema
void reset_state() {
    button_b_pressed = false;
    button_a_pressed = false;
    npInit(7);
    npClear();  // Define todas as cores dos LEDs como (0, 0, 0)
    npWrite();  // Envia as alterações para a matriz de LEDs
    parar_pwm_e_temporizador(); // Para o PWM e o temporizador
    set_led_color(0, 0, 0);  // Apaga o LED ao iniciar
    ssd1306_clear();         // Limpa o display
}

// Exibe o menu e aguarda o pressionamento de um botão
void display_menu_and_wait() {
    reset_state();
    set_led_color(1, 1, 0);  // Liga o LED amarelo (Vermelho + Verde)
    display_menu(&frame_area);

    // Aguarda o pressionamento do botão A ou B
    while (!button_b_pressed && !button_a_pressed) {
        display_menu(&frame_area);
        sleep_ms(100);
    }
}

// Processa os dados JSON recebidos do servidor
void process_json_data() {
    set_led_color(0, 0, 0);  // Apaga o LED ao iniciar
    display_wifi(&frame_area);

    double frequencia = 0.0;
    char condicao[256] = {0};

    if (receive_json_data(&frequencia, condicao, sizeof(condicao))) {
        // Limpa a matriz de LEDs
        npClear();

        // Exibe um coração vermelho para condição "Normal", azul caso contrário
        if (strcmp(condicao, "Normal") == 0) {
            drawHeart(100, 0, 0);  // Vermelho
        } else {
            drawHeart(0, 0, 100);  // Azul
        }

        // Configura o PWM e o temporizador com a frequência recebida
        configurar_pwm_e_temporizador(frequencia);

        // Exibe continuamente as informações
        while (true) {
            display_info(frequencia, condicao, &frame_area);
            sleep_ms(2000);
        }
    } else {
        // Trata erro ao receber dados JSON
    }
}

// Função principal
int main() 
{
    stdio_init_all();  // Inicializa a entrada/saída padrão
    setup_i2c();       // Configura a comunicação I2C
    ssd1306_init();    // Inicializa o display
    calculate_render_area_buffer_length(&frame_area);  // Calcula o comprimento do buffer para a área de renderização
    setup_gpio();      // Configura os pinos GPIO

    while (true) {
        // Exibe o menu e aguarda o pressionamento de um botão
        display_menu_and_wait();

        // Verifica qual botão foi pressionado
        if (button_a_pressed) 
        {
            // Botão A pressionado: inicia a captura de áudio
            button_a_pressed = false; // Reseta o estado do botão A
            display_audio(&frame_area);
            audio_capture_init();
            while(true)
            {
                audio_capture_loop();  // Entra no loop de captura de áudio
            }
        } 
        else if (button_b_pressed) 
        {
            // Botão B pressionado: inicia o processamento de dados JSON
            button_b_pressed = false; // Reseta o estado do botão B
            process_json_data();      // Entra na função de processamento de dados JSON
        }
    }

    return 0;
}