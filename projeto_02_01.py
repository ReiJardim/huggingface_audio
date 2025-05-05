# Projeto real
from pathlib import Path
from datasets import load_dataset
import IPython
import soundfile
from transformers import pipeline

import sounddevice as sd


duracao = 20
taxa_amostragem = 16000
tamanho_vetor = int(duracao * taxa_amostragem)

gravacao = sd.rec(tamanho_vetor, samplerate=taxa_amostragem, channels=1)
sd.wait()

gravacao

# Concertando o vetor

gravacao = gravacao.reshape(-1)
gravacao.shape


IPython.display.Audio(data=gravacao, rate=taxa_amostragem)


modelo = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'
classificador = pipeline('audio-classification', model=modelo)
classificador(gravacao)
