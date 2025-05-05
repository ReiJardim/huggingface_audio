# Reconhecimento de voz

# Importa Path para manipulação de caminhos de arquivos
from pathlib import Path
# Importa função para carregar datasets do HuggingFace
from datasets import load_dataset
# Importa IPython para exibir áudio no notebook
import IPython

# Importa pipeline da HuggingFace Transformers para tarefas de áudio
from transformers import pipeline


# Base de dados : https://huggingface.co/datasets/PolyAI/minds14

# Define o nome do dataset e o idioma
nome_dataset = "PolyAI/minds14"
lingua_dataset = "pt-PT"
# Carrega as 10 primeiras amostras do dataset
dados = load_dataset(nome_dataset, name=lingua_dataset, split='train[:10]')

# Exibe a primeira linha do dataset para inspeção
dados[0]

# Percorre as amostras do dataset e exibe o áudio (comentado)
for linha in dados:
    dados_som = linha['audio']['array']  # Array de áudio
    taxa_amostragem = linha['audio']['sampling_rate']  # Taxa de amostragem
    # display(IPython.display.Audio(data=dados_som, rate=taxa_amostragem))

# Fonte do modelo : https://huggingface.co/openai/whisper-medium
# Define o modelo Whisper para reconhecimento de fala
modelo = 'openai/whisper-medium'
# Cria o pipeline de reconhecimento automático de fala
reconhecedor_de_fala = pipeline('automatic-speech-recognition', model=modelo,)

# Os modelos Whisper ajustam o sampling rate automaticamente se passarmos o dicionário de áudio
reconhecedor_de_fala(dados[0]['audio'])

# Seleciona um índice de áudio para teste
idx_audio = 7

audio = dados[idx_audio]['audio']
# Exibe o áudio selecionado no notebook (comentado)
# display(IPython.display.Audio(
#    data=audio['array'], rate=audio['sampling_rate']))
# Realiza o reconhecimento de fala no áudio selecionado
reconhecedor_de_fala(audio)
