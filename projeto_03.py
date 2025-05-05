# Reconhecimento de voz

from pathlib import Path
from datasets import load_dataset
import IPython

from transformers import pipeline


# Base de dados : https://huggingface.co/datasets/PolyAI/minds14

nome_dataset = "PolyAI/minds14"
lingua_dataset = "pt-PT"
dados = load_dataset(nome_dataset, name=lingua_dataset, split='train[:10]')

dados[0]

for linha in dados:
    dados_som = linha['audio']['array']
    taxa_amostragem = linha['audio']['sampling_rate']
    # display(IPython.display.Audio(data=dados_som, rate=taxa_amostragem))

# Fonte do modelo : https://huggingface.co/openai/whisper-medium
modelo = 'openai/whisper-medium'
reconhecedor_de_fala = pipeline('automatic-speech-recognition', model=modelo,)

# Poderíamos tentar ajustar o sampling rate novamente, mas os modelos Whisper tem mais uma facilidade para nós: se passarmos o dicionário de Audio diretamente, eles ajustam o sampling rate automaticamente:
reconhecedor_de_fala(dados[0]['audio'])

idx_audio = 7

audio = dados[idx_audio]['audio']
# display(IPython.display.Audio(
#    data=audio['array'], rate=audio['sampling_rate']))
reconhecedor_de_fala(audio)
