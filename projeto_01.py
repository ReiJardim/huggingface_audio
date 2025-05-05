# Nesse exmplo os dados do datsets são salvos na maquina local, devido  a isso demora  um pouco para executar na primeira vez.

from datasets import Audio
from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path

import IPython
import soundfile
import IPython.display
import numpy as np
import librosa
import librosa.display


dataset = load_dataset('ashraq/esc50')


dados = dataset['train']
primeiras_linhas = dados.select(range(10))
for linha in primeiras_linhas:
    print(linha)
    print('-----')


idx_dados = 1
linha = dados[idx_dados]

plt.subplots(figsize=(30, 4))
plt.plot(linha['audio']['array'])
plt.suptitle(linha['category'])

print(f"Quantidade de dados: {linha['audio']['array'].shape}")

plt.show()

pasta_saida = Path('audios') / 'objetos'
pasta_saida.mkdir(exist_ok=True, parents=True)

for i, linha in enumerate(primeiras_linhas):
    objeto = linha['category']
    dados_som = linha['audio']['array']
    taxa_amostragem = linha['audio']['sampling_rate']
    caminho_saida = pasta_saida / f'{i:03d}_{objeto}.wav'
    # Salvando em um arquivo de áudio
    soundfile.write(file=caminho_saida, data=dados_som,
                    samplerate=taxa_amostragem)
    # Exibindo no Jupyter Notebook
    # display(IPython.display.Audio(data=dados_som, rate=taxa_amostragem))

    # Plota o gráfico de forma de onda
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(dados_som, sr=taxa_amostragem)
    plt.title(f'Waveform - Áudio {i+1}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    # plt.show()


dados = dados.cast_column('audio', Audio(sampling_rate=16000))
