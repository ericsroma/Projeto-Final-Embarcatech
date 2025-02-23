// data_server.h
#ifndef DATA_SERVER_H
#define DATA_SERVER_H

#include <stdbool.h>
#include <stddef.h>

// Função que aguarda o recebimento do JSON via Wi-Fi e preenche os parâmetros
// frequencia_cardiaca: variável que receberá o valor da frequência cardíaca
// condicao: buffer que receberá a string da condição
// condicao_size: tamanho do buffer da condição
// Retorna true se os dados foram recebidos e processados com sucesso, ou false em caso de erro.
bool receive_json_data(double *frequencia_cardiaca, char *condicao, size_t condicao_size);

#endif // DATA_SERVER_H
