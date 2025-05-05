# Reconhecimento de voz para audio proprio

from pathlib import Path
from datasets import load_dataset
import IPython

from transformers import pipeline

import sounddevice as sd

modelo = 'openai/whisper-medium'

reconhecedor_de_fala = pipeline(
    'automatic-speech-recognition',
    model=modelo,
    generate_kwargs={"task": "transcribe", "language": "portuguese"},
)


duracao = 10
taxa_amostragem = 16000
tamanho_vetor = int(duracao * taxa_amostragem)

gravacao = sd.rec(tamanho_vetor, samplerate=taxa_amostragem, channels=1)
sd.wait()

gravacao

reconhecedor_de_fala(
    {'raw': gravacao.ravel(), 'sampling_rate': taxa_amostragem})
