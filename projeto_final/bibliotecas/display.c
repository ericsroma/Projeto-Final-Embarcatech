#include "headers/display.h"
#include <stdio.h>
#include <string.h>
#include "hardware/i2c.h"
#include "inc/ssd1306.h"

// Definição dos pinos I2C
const uint I2C_SDA = 14;
const uint I2C_SCL = 15;

// Função para exibir informações de frequência cardíaca a a condição do coração no display
void display_info(float freq, const char *cond, struct render_area *frame_area)
{
    char buffer[20];
    uint8_t ssd[ssd1306_buffer_length];
    memset(ssd, 0, ssd1306_buffer_length); // Limpa o buffer do display

    ssd1306_draw_string(ssd, 5, 0, "Modo Visualiza"); // Desenha o título

    snprintf(buffer, sizeof(buffer), "Freq: %.1f bpm", freq); // Formata a frequência
    ssd1306_draw_string(ssd, 5, 16, buffer); // Desenha a frequência no display

    snprintf(buffer, sizeof(buffer), "Cond: %s", cond); // Formata a condição
    ssd1306_draw_string(ssd, 5, 32, buffer); // Desenha a condição no display

    render_on_display(ssd, frame_area); // Renderiza no display
}

// Função para exibir o menu no display
void display_menu(struct render_area *frame_area)
{
    uint8_t ssd[ssd1306_buffer_length];
    memset(ssd, 0, ssd1306_buffer_length); // Limpa o buffer do display

    ssd1306_draw_string(ssd, 52, 0, "Menu"); // Desenha o título do menu

    ssd1306_draw_string(ssd, 5, 16, "Botao A"); // Desenha a opção do botão A
    ssd1306_draw_string(ssd, 5, 24, "Captura Som"); // Desenha a descrição da opção do botão A

    ssd1306_draw_string(ssd, 5, 40, "Botao B"); // Desenha a opção do botão B
    ssd1306_draw_string(ssd, 5, 48, "Visualizacao"); // Desenha a descrição da opção do botão B

    render_on_display(ssd, frame_area); // Renderiza no display
}

// Função para exibir o status de aguardar a conexão WiFi e o recebimento do JSON no display
void display_wifi(struct render_area *frame_area)
{
    uint8_t ssd[ssd1306_buffer_length];
    memset(ssd, 0, ssd1306_buffer_length); // Limpa o buffer do display

    ssd1306_draw_string(ssd, 5, 0, "Aguardando"); // Desenha a mensagem de aguardando

    ssd1306_draw_string(ssd, 5, 16, "Conexao"); // Desenha a mensagem de conexão

    render_on_display(ssd, frame_area); // Renderiza no display
}

// Função para indicar o modo de captura de áudio no display
void display_audio(struct render_area *frame_area)
{
    uint8_t ssd[ssd1306_buffer_length];
    memset(ssd, 0, ssd1306_buffer_length); // Limpa o buffer do display

    ssd1306_draw_string(ssd, 5, 0, "Captura"); // Desenha a mensagem de captura

    ssd1306_draw_string(ssd, 5, 16, "de audio..."); // Desenha a mensagem de áudio

    render_on_display(ssd, frame_area); // Renderiza no display
}

// Função para configurar o I2C
void setup_i2c()
{
    i2c_init(i2c1, ssd1306_i2c_clock * 1000); // Inicializa o I2C com a frequência especificada
    gpio_set_function(I2C_SDA, GPIO_FUNC_I2C); // Configura o pino SDA
    gpio_set_function(I2C_SCL, GPIO_FUNC_I2C); // Configura o pino SCL
    gpio_pull_up(I2C_SDA); // Ativa o pull-up no pino SDA
    gpio_pull_up(I2C_SCL); // Ativa o pull-up no pino SCL
}