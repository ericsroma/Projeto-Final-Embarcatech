// data_server.c
#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"
#include "lwip/tcp.h"
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include "headers/cJSON.h"
#include "headers/data_server.h"

// Parâmetros de Wi-Fi e servidor
#define WIFI_SSID "RomaNet1"
#define WIFI_PASS "romanet35"
#define DATA_PORT 5000
#define DATA_BUFFER_SIZE 1024

// Variáveis globais para controle de dados
static volatile bool data_complete = false;
static char data_buffer[DATA_BUFFER_SIZE];

// ============================================================================
// Função para processar o JSON e extrair os dados desejados
// ----------------------------------------------------------------------------
static void process_json_data(const char *json_data, double *frequencia_cardiaca, char *condicao, size_t condicao_size) {
    cJSON *root = cJSON_Parse(json_data);
    if (root == NULL) {
        printf("Erro ao parsear JSON\n");
        return;
    }

    cJSON *freq_json = cJSON_GetObjectItem(root, "Frequencia cardiaca");
    cJSON *cond_json = cJSON_GetObjectItem(root, "Condicao");

    if (freq_json && cond_json) {
        *frequencia_cardiaca = freq_json->valuedouble;
        // Garante que não ocorra estouro do buffer
        snprintf(condicao, condicao_size, "%s", cond_json->valuestring);
    } else {
        printf("Dados incompletos no JSON\n");
    }

    cJSON_Delete(root);
}

// ============================================================================
// Callback para processar os dados recebidos via TCP
// ----------------------------------------------------------------------------
static err_t data_callback(void *arg, struct tcp_pcb *tpcb, struct pbuf *p, err_t err) {
    if (p == NULL) {
        // Conexão encerrada: sinaliza que os dados foram recebidos completamente
        data_complete = true;
        tcp_close(tpcb);
        return ERR_OK;
    }

    if (p->tot_len > 0) {
        if (p->tot_len <= DATA_BUFFER_SIZE) {
            pbuf_copy_partial(p, data_buffer, p->tot_len, 0);
            data_buffer[p->tot_len] = '\0'; // Garante que a string seja terminada
        } else {
            printf("Erro: buffer de dados cheio\n");
        }
    }

    pbuf_free(p);
    return ERR_OK;
}

// ============================================================================
// Callback de conexão: associa o callback de dados à conexão
// ----------------------------------------------------------------------------
static err_t connection_callback(void *arg, struct tcp_pcb *newpcb, err_t err) {
    tcp_recv(newpcb, data_callback);
    return ERR_OK;
}

// ============================================================================
// Configura e inicia o servidor TCP para recebimento dos dados JSON
// ----------------------------------------------------------------------------
static void start_server(void) {
    struct tcp_pcb *pcb = tcp_new();
    if (!pcb) {
        printf("Erro ao criar PCB\n");
        return;
    }

    if (tcp_bind(pcb, IP_ADDR_ANY, DATA_PORT) != ERR_OK) {
        printf("Erro ao ligar o servidor na porta %d\n", DATA_PORT);
        return;
    }

    pcb = tcp_listen(pcb);
    tcp_accept(pcb, connection_callback);
    printf("Servidor de dados rodando na porta %d...\n", DATA_PORT);
}

// ============================================================================
// Inicializa o Wi-Fi, conecta e aguarda o recebimento dos dados JSON
// ----------------------------------------------------------------------------
static bool wait_for_data(void) {
    // Inicializa o Wi-Fi
    if (cyw43_arch_init()) {
        printf("Erro ao inicializar o Wi-Fi\n");
        return false;
    }
    cyw43_arch_enable_sta_mode();
    printf("Conectando ao Wi-Fi...\n");

    if (cyw43_arch_wifi_connect_timeout_ms(WIFI_SSID, WIFI_PASS, CYW43_AUTH_WPA2_AES_PSK, 10000)) {
        printf("Falha ao conectar ao Wi-Fi\n");
        return false;
    } else {
        printf("Conectado.\n");
        uint8_t *ip_address = (uint8_t*)&(cyw43_state.netif[0].ip_addr.addr);
        printf("Endereço IP %d.%d.%d.%d\n", ip_address[0], ip_address[1], ip_address[2], ip_address[3]);
    }

    // Inicia o servidor de dados
    start_server();

    // Loop para manter o Wi-Fi ativo e aguardar o JSON
    while (!data_complete) {
        cyw43_arch_poll();
        sleep_ms(100);
    }

    return true;
}

// ============================================================================
// Função pública que aguarda e processa os dados JSON, retornando os valores via parâmetros
// ----------------------------------------------------------------------------
bool receive_json_data(double *frequencia_cardiaca, char *condicao, size_t condicao_size) {
    // Reinicia o flag e buffer para uma nova operação
    data_complete = false;
    memset(data_buffer, 0, DATA_BUFFER_SIZE);

    if (!wait_for_data()) {
        return false;
    }

    process_json_data(data_buffer, frequencia_cardiaca, condicao, condicao_size);
    return true;
}
