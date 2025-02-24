import serial  # Biblioteca para comunicação serial
import numpy as np  # Biblioteca para manipulação de arrays numéricos
import sounddevice as sd  # Biblioteca para reprodução e captura de áudio em tempo real
import soundfile as sf  # Biblioteca para salvar arquivos de áudio
import keyboard  # Biblioteca para capturar eventos do teclado

# Configuração da porta serial
porta_serial = "COM3"  # Define a porta serial (ajustar conforme necessário)
taxa_amostragem = 8000  # Define a taxa de amostragem do áudio (8 kHz)
formato_dados = np.int32  # Formato dos dados recebidos: inteiro de 32 bits

# Abrir a conexão serial
ser = serial.Serial(porta_serial, baudrate=115200, timeout=1)  # Configura a porta com baudrate de 115200 e timeout de 1 segundo
ser.reset_input_buffer()  # Limpa o buffer de entrada para descartar dados antigos

# Variáveis para controle da gravação
gravando = False  # Flag para indicar se a gravação está ativa
audio_gravado = []  # Buffer para armazenar o áudio gravado

# Função de callback para processar e reproduzir o áudio
def audio_callback(outdata, frames, time, status):
    global gravando, audio_gravado

    if status:  # Verifica se há algum erro ou aviso no stream
        print(f"Status: {status}")

    # Calcula o número de bytes necessários para preencher o buffer
    bytes_necessarios = frames * 4  # 4 bytes por amostra (32 bits)

    # Verifica se há dados suficientes no buffer serial
    if ser.in_waiting >= bytes_necessarios:
        data = ser.read(bytes_necessarios)  # Lê os dados da porta serial

        # Verifica se o pacote está alinhado (múltiplo de 4 bytes)
        if len(data) % 4 == 0:
            # Converte os bytes para amostras de áudio (int32) e normaliza para float32
            samples = np.frombuffer(data, dtype=formato_dados).astype(np.float32)
            samples /= (2**31 - 1)  # Normaliza para a faixa -1 a 1
        else:
            print("Pacote de áudio desalinhado, descartando...")
            samples = np.zeros(frames, dtype=np.float32)  # Zera o áudio para evitar ruído

        outdata[:, 0] = samples  # Reproduz o áudio processado

        if gravando:
            audio_gravado.extend(samples)  # Armazena os samples para gravação
    else:
        # Se não houver dados suficientes, preenche com silêncio
        outdata[:, 0] = np.zeros(frames, dtype=np.float32)

try:
    # Iniciar a reprodução de áudio com o stream de saída
    with sd.OutputStream(samplerate=taxa_amostragem, channels=1, callback=audio_callback, dtype='float32'):
        print("Reproduzindo áudio... Pressione 'G' para gravar e 'S' para parar a gravação.")
        
        while True:
            # Inicia a gravação ao pressionar a tecla 'G'
            if keyboard.is_pressed('g') and not gravando:
                print("Iniciando gravação...")
                gravando = True
                audio_gravado = []  # Limpa o buffer de gravação para começar do zero
                
            # Para a gravação ao pressionar a tecla 'S'
            if keyboard.is_pressed('s') and gravando:
                print("Parando gravação e salvando arquivo...")
                gravando = False
                # Salva o áudio gravado em um arquivo WAV
                sf.write('audio_gravado.wav', np.array(audio_gravado), taxa_amostragem)
                print(f"Áudio salvo como 'audio_gravado.wav'")

except KeyboardInterrupt:
    # Trata a interrupção por teclado (Ctrl + C)
    print("\nParando reprodução de áudio.")
finally:
    ser.close()  # Fecha a conexão serial ao encerrar o programa
