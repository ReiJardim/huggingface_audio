# Projeto real
# Importa Path para manipulação de caminhos de arquivos
from pathlib import Path
# Importa função para carregar datasets do HuggingFace
from datasets import load_dataset
# Importa IPython para exibir áudio no notebook
import IPython
# Importa soundfile para salvar arquivos de áudio
import soundfile
# Importa pipeline da HuggingFace Transformers para tarefas de áudio
from transformers import pipeline

# Importa sounddevice para gravação de áudio pelo microfone
import sounddevice as sd

# Define a duração da gravação em segundos
duracao = 20
# Define a taxa de amostragem do áudio
taxa_amostragem = 16000
# Calcula o tamanho do vetor de áudio
tamanho_vetor = int(duracao * taxa_amostragem)

# Grava o áudio do microfone
gravacao = sd.rec(tamanho_vetor, samplerate=taxa_amostragem, channels=1)
sd.wait()

gravacao

# Ajusta o formato do vetor de áudio para 1D
gravacao = gravacao.reshape(-1)
gravacao.shape

# Exibe o áudio gravado no notebook
IPython.display.Audio(data=gravacao, rate=taxa_amostragem)

# Define o modelo de classificação de áudio
modelo = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'
# Cria o pipeline de classificação de áudio
classificador = pipeline('audio-classification', model=modelo)
# Classifica o áudio gravado
classificador(gravacao)
