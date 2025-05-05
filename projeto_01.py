# Nesse exmplo os dados do datsets são salvos na maquina local, devido  a isso demora  um pouco para executar na primeira vez.

# Importa a classe Audio do HuggingFace Datasets para manipulação de áudio
from datasets import Audio
# Importa função para carregar datasets do HuggingFace
from datasets import load_dataset
# Importa matplotlib para visualização de gráficos
import matplotlib.pyplot as plt
# Importa Path para manipulação de caminhos de arquivos
from pathlib import Path

# Importa IPython para exibir áudio no notebook
import IPython
# Importa soundfile para salvar arquivos de áudio
import soundfile
# Importa display do IPython para exibir áudio
import IPython.display
# Importa numpy para manipulação de arrays
import numpy as np
# Importa librosa para processamento e visualização de áudio
import librosa
import librosa.display

# Carrega o dataset ESC-50 de sons ambientais
# O download pode demorar na primeira execução

# Carrega o dataset
dataset = load_dataset('ashraq/esc50')

# Seleciona a divisão de treino do dataset
dados = dataset['train']
# Seleciona as 10 primeiras linhas para inspeção
primeiras_linhas = dados.select(range(10))
# Exibe as primeiras linhas do dataset
for linha in primeiras_linhas:
    print(linha)
    print('-----')

# Seleciona um índice específico para análise
idx_dados = 1
linha = dados[idx_dados]

# Plota a forma de onda do áudio selecionado
plt.subplots(figsize=(30, 4))
plt.plot(linha['audio']['array'])
plt.suptitle(linha['category'])

# Exibe a quantidade de amostras do áudio
print(f"Quantidade de dados: {linha['audio']['array'].shape}")

plt.show()

# Cria a pasta de saída para salvar os áudios
pasta_saida = Path('audios') / 'objetos'
pasta_saida.mkdir(exist_ok=True, parents=True)

# Salva e plota as formas de onda dos 10 primeiros áudios
for i, linha in enumerate(primeiras_linhas):
    objeto = linha['category']  # Categoria do som
    dados_som = linha['audio']['array']  # Array de áudio
    taxa_amostragem = linha['audio']['sampling_rate']  # Taxa de amostragem
    caminho_saida = pasta_saida / f'{i:03d}_{objeto}.wav'
    # Salvando em um arquivo de áudio
    soundfile.write(file=caminho_saida, data=dados_som,
                    samplerate=taxa_amostragem)
    # Exibindo no Jupyter Notebook (comentado)
    # display(IPython.display.Audio(data=dados_som, rate=taxa_amostragem))

    # Plota o gráfico de forma de onda
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(dados_som, sr=taxa_amostragem)
    plt.title(f'Waveform - Áudio {i+1}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    # plt.show()

# Altera a taxa de amostragem da coluna de áudio para 16kHz
dados = dados.cast_column('audio', Audio(sampling_rate=16000))
