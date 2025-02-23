/*Este código é uma adaptação do  código no repositório: xdsopl. (2023). i2s_mic. GitHub repository. Disponível em: https://github.com/xdsopl/i2s_mic */

#include "headers/audio_capture.h"

// Definição dos pinos e variáveis globais
static const uint button_pin = 5;  
static const uint led_green_pin = 11;
static const uint led_red_pin = 13;
static volatile bool button_pressed = false;
static volatile bool is_sending_audio = false;

static volatile bool button_state = false; 
static volatile bool last_button_state = false; 
static volatile absolute_time_t last_debounce_time = 0;
static const uint debounce_delay = 200 * 1000; 

static PIO pio = pio0;           
static uint offset;
static uint sm;

// Função de tratamento de interrupção do botão
void gpio_irq_handler(uint gpio, uint32_t events) {
    if (gpio == button_pin) {
        // Verifica se o tempo de debounce passou
        if (absolute_time_diff_us(last_debounce_time, get_absolute_time()) > debounce_delay) {
            button_state = !button_state; // Alterna o estado do botão
            button_pressed = true; // Marca que o botão foi pressionado
            last_debounce_time = get_absolute_time(); // Atualiza o tempo de debounce
        }
    }
}

// Função de inicialização da captura de áudio
void audio_capture_init() {
    uint sd_pin = 2;          
    uint sck_ws_pins = 3;     

    stdio_init_all(); // Inicializa todas as saídas padrão

    // Configuração dos LEDs
    gpio_init(led_green_pin);
    gpio_init(led_red_pin);
    gpio_set_dir(led_green_pin, GPIO_OUT);
    gpio_set_dir(led_red_pin, GPIO_OUT);
    gpio_put(led_green_pin, 1); // Liga o LED verde
    gpio_put(led_red_pin, 0); // Desliga o LED vermelho

    // Configuração do botão
    gpio_init(button_pin);
    gpio_set_dir(button_pin, GPIO_IN);
    gpio_pull_up(button_pin); // Ativa o pull-up interno do botão
    gpio_set_irq_enabled_with_callback(button_pin, GPIO_IRQ_EDGE_FALL, true, &gpio_irq_handler); // Configura a interrupção do botão

    // Configuração do PIO
    offset = pio_add_program(pio, &i2s_mic_program); // Adiciona o programa PIO
    sm = pio_claim_unused_sm(pio, true); // Reivindica uma máquina de estado não utilizada

    pio_gpio_init(pio, sd_pin);
    pio_gpio_init(pio, sck_ws_pins); 
    pio_gpio_init(pio, sck_ws_pins + 1); 

    pio_sm_set_consecutive_pindirs(pio, sm, sd_pin, 1, false); // Configura a direção dos pinos
    pio_sm_set_consecutive_pindirs(pio, sm, sck_ws_pins, 2, true);

    pio_sm_config conf = i2s_mic_program_get_default_config(offset);
    sm_config_set_in_pins(&conf, sd_pin);    
    sm_config_set_sideset_pins(&conf, sck_ws_pins);
    sm_config_set_in_shift(&conf, false, true, 32); 
    sm_config_set_fifo_join(&conf, PIO_FIFO_JOIN_RX); 
    sm_config_set_clkdiv_int_frac(&conf, 122, 16);

    pio_sm_init(pio, sm, offset, &conf); // Inicializa a máquina de estado com a configuração
}

// Função para iniciar a captura de áudio
void audio_capture_start() {
    gpio_put(led_green_pin, 0); // Desliga o LED verde
    gpio_put(led_red_pin, 1);   // Liga o LED vermelho
    pio_sm_set_enabled(pio, sm, true); // Habilita a máquina de estado
    is_sending_audio = true; // Marca que está enviando áudio
}

// Função para parar a captura de áudio
void audio_capture_stop() {
    gpio_put(led_green_pin, 1); // Liga o LED verde
    gpio_put(led_red_pin, 0);   // Desliga o LED vermelho
    pio_sm_set_enabled(pio, sm, false); // Desabilita a máquina de estado
    is_sending_audio = false; // Marca que não está enviando áudio
}

// Função principal de loop da captura de áudio
void audio_capture_loop() {
    if (button_pressed) {
        button_pressed = false; // Reseta o estado do botão

        if (is_sending_audio) {
            audio_capture_stop(); // Para a captura de áudio se estiver enviando
        } else {
            audio_capture_start(); // Inicia a captura de áudio se não estiver enviando
        }
    }

    if (is_sending_audio) {
        uint32_t left = pio_sm_get_blocking(pio, sm); // Obtém dados de áudio
        putchar_raw((left >> 0) & 255); // Envia o byte menos significativo
        putchar_raw((left >> 8) & 255); // Envia o segundo byte
        putchar_raw((left >> 16) & 255); // Envia o terceiro byte
        putchar_raw((left >> 24) & 255); // Envia o byte mais significativo
    }
}