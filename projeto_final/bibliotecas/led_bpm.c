#include "pico/stdlib.h"
#include "hardware/pwm.h"
#include "hardware/timer.h"
#include <math.h>
#include "headers/led_bpm.h"

#define LED_PIN 13  // Pino do LED (PWM)

typedef enum { DIASTOLE, SISTOLE } Estado;
Estado estado = DIASTOLE;
volatile int brilho = 0;  // Brilho do LED (0-255)
const float DIVIDER_PWM = 16.0;
float bpm = 120.0;  // Exemplo de BPM (batimentos por minuto)

struct repeating_timer timer;  // Definir o temporizador como uma variável global

bool atualiza_led(struct repeating_timer *t) 
{
    static int passo = 0;
    
    // Cálculo do ciclo total baseado no BPM
    int cicloTotal = (60.0 / bpm) * 1000;  

    // Tempo das fases do ciclo cardíaco
    int tempoDiastole = cicloTotal * 0.65;  // 65% do ciclo
    int tempoSistole = cicloTotal * 0.35;   // 35% do ciclo

    int passos;
    // Ajusta a quantidade de passos com base no BPM

    if (bpm <= 50) 
    {
        passos = 50;  // Para BPM baixos, mais passos
    } 
    else if (bpm >= 150) 
    {
        passos = 20;  // Para BPM altos, menos passos
    } 
    else 
    {
        passos = 50 - (bpm - 50) / 2;  // Reduz progressivamente os passos
    }

    // Intervalo para a fase atual
    int intervalo;

    if (estado == DIASTOLE) 
    {
        intervalo = tempoDiastole / passos;
    } 
    else 
    {
        intervalo = tempoSistole / passos;
    }

    // Ajuste do brilho do LED baseado no estado
    if (passo < passos) 
    {
        float progresso = (float)passo / passos;

        float intensidade;

        if (estado == DIASTOLE) 
        {
            // Curva de decaimento exponencial para a diástole
            intensidade = exp(-progresso * 4.0);  // Fator aumentado para decaimento mais suave
        } 
        else 
        {
            // Curva de crescimento exponencial para a sístole
            intensidade = 1.0 - exp(-progresso * 4.0);  // Fator aumentado para crescimento mais suave

            // Adiciona uma pequena oscilação no meio da sístole para simular a ejeção do sangue
            if (progresso >= 0.3 && progresso <= 0.7) 
            {
                intensidade += 0.1 * sin(2 * M_PI * progresso);  // Oscilação suave
            }
        }

        brilho = (int)(intensidade * 255);

        // Garantir que o brilho não ultrapasse os limites
        if (brilho < 0) 
        {
            brilho = 0;
        } 
        else if (brilho > 255) 
        {
            brilho = 255;
        }

        // Atualizar o nível de PWM do LED
        pwm_set_gpio_level(LED_PIN, brilho);  // Ajuste suave do PWM

        passo++;
    } 
    else 
    {
        passo = 0;

        // Trocar o estado entre diástole e sístole
        if (estado == DIASTOLE) 
        {
            estado = SISTOLE;
        } 
        else 
        {
            estado = DIASTOLE;
        }
    }
    
    return true; // Continua repetindo o temporizador
}

// Função para configurar o PWM e o temporizador com a frequência em BPM
void configurar_pwm_e_temporizador(float nova_frequencia) 
{
    bpm = nova_frequencia;  // Atualiza a frequência global

    // Configuração do PWM
    gpio_set_function(LED_PIN, GPIO_FUNC_PWM);
    uint slice_num = pwm_gpio_to_slice_num(LED_PIN);
    pwm_set_wrap(slice_num, 255);  // Resolução do PWM
    pwm_set_clkdiv(slice_num, DIVIDER_PWM);  // Ajuste de frequência PWM
    pwm_set_gpio_level(LED_PIN, 0); // Inicializa com duty cycle baixo
    pwm_set_enabled(slice_num, true);

    // Configuração do temporizador
    add_repeating_timer_ms(10, atualiza_led, NULL, &timer);  // Chama a função atualiza_led a cada 10ms
}

void parar_pwm_e_temporizador() 
{
    // Desabilita o PWM
    uint slice_num = pwm_gpio_to_slice_num(LED_PIN);
    pwm_set_enabled(slice_num, false);  // Desativa o PWM no slice correspondente

    // Cancela o temporizador repetitivo
    cancel_repeating_timer(&timer);
}
