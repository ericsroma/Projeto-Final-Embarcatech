import joblib
import numpy as np
import librosa
import json
import socket
from collections import Counter
from preproc_functions import butterworth, signal_windowing
from scipy.signal import find_peaks

# Carregar o modelo SVM treinado para classificação de condições cardíacas
model = joblib.load('./modelo_svm.pkl')


def preprocess_audio(filepath, config):
    """
    Pré-processa o áudio aplicando filtragem Butterworth e segmentação por janelas.
    Extrai características MFCC para serem usadas na inferência.
    """
    y, sr = librosa.load(filepath, sr=config['sampling_rate'])
    
    # Filtrar o sinal para remover ruídos fora da faixa de interesse
    filtered_signal = butterworth(data=y, lowcut=config['filter_lowcut'], highcut=config['filter_highcut'], fs=config['sampling_rate'], order=config['filter_order'])
    
    # Segmentar o sinal em janelas
    windows, _ = signal_windowing(signal=filtered_signal, window_size=config['signal_window_size'])
    
    features = []
    for window in windows:
        # Extrair coeficientes MFCC para cada janela
        mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=config['n_mfcc'], 
                                    n_fft=int(config['mel_frame_lenght'] * sr), 
                                    hop_length=int(config['mel_frame_overlap'] * sr), 
                                    n_mels=config['n_mels'])
        
        mfcc = mfcc.T
        feature_vector = mfcc.reshape(1, -1)
        features.append(feature_vector.flatten())
    
    return np.array(features)


def infer(filepath, config):
    """
    Realiza a inferência com o modelo SVM para classificar a condição cardíaca.
    Retorna a condição mais comum encontrada nas janelas do sinal.
    """
    features = preprocess_audio(filepath, config)
    
    # Ajustar o número de features para corresponder ao modelo treinado
    features = features[:, :model.n_features_in_]
    predictions = model.predict(features)
    
    # Mapeamento de rótulos do modelo para condições cardíacas
    label_mapping = {0: 'AS', 1: 'MR', 2: 'MS', 3: 'MVP', 4: 'N'}
    predicted_labels = [label_mapping[pred] for pred in predictions]
    most_common_prediction = Counter(predicted_labels).most_common(1)[0][0]

    # Retornar a condição mais frequente detectada
    if most_common_prediction == 'AS':
        return 'AS'
    elif most_common_prediction == 'MR':
        return 'MR'
    elif most_common_prediction == 'MS':
        return 'MS'
    elif most_common_prediction == 'MVP':
        return 'MVP'
    elif most_common_prediction == 'N':
        return 'Normal'
    
    return most_common_prediction


def calcular_frequencia_cardiaca(audio_file):
    """
    Calcula a frequência cardíaca com base nos picos do sinal de áudio.
    Retorna a frequência em BPM ou None se não forem detectados batimentos.
    """
    y, sr = librosa.load(audio_file, sr=44000)

    # Normalizar o sinal para evitar variações bruscas de amplitude
    y = y / np.max(np.abs(y))

    # Detectar picos que representam batimentos cardíacos
    peaks, _ = find_peaks(y, distance=sr//2, height=0.1)

    if len(peaks) > 1:
        # Calcular a média dos intervalos entre picos para estimar o BPM
        intervals = np.diff(peaks) / sr
        bpm = 60 / np.mean(intervals)
        return bpm
    else:
        print("Nenhum batimento detectado. Verifique a qualidade do áudio.")
        return None


def enviar_json(frequencia_cardiaca, condicao, ip, porta):
    """
    Envia os dados de frequência cardíaca e condição detectada via socket TCP.
    """
    dados = {"Frequencia cardiaca": frequencia_cardiaca, "Condicao": condicao}
    json_data = json.dumps(dados)
    
    cliente_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        cliente_socket.connect((ip, porta))
        cliente_socket.sendall(json_data.encode('utf-8'))
    except Exception as e:
        print(f"Erro ao enviar dados: {e}")
    finally:
        cliente_socket.close()


if __name__ == "__main__":
    # Configuração dos parâmetros de pré-processamento e extração de características
    config = {
        'filter_lowcut': 25,
        'filter_highcut': 600,
        'filter_order': 2,
        'n_mfcc': 19,
        'mel_frame_lenght': 0.01,
        'mel_frame_overlap': 0.001,
        'n_mels': 24,
        'signal_window_size': 6000,
        'sampling_rate': 8000
    }
    
    # Arquivo de áudio de entrada
    arquivo_audio = 'amostra_2.wav'
    # amostra_2.wav - "Normal" - Exemplo de som cardíaco capturado e processado
    # New_MR_001.wav - "MR" - Amostra anormal da Base de dados
    # som_coracao_amp - "Normal" - Exemplo de som cardíaco capturado sem ser processado
    
    # Realizar inferência e calcular frequência cardíaca
    condicao = infer(arquivo_audio, config)
    frequencia_cardiaca = calcular_frequencia_cardiaca(arquivo_audio)
    
    # Enviar os dados processados para o servidor
    ip_destino = '192.168.0.38'
    porta_destino = 5000
    
    enviar_json(frequencia_cardiaca, condicao, ip_destino, porta_destino)