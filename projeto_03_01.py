# Reconhecimento de voz para audio proprio

# Importa Path para manipulação de caminhos de arquivos
from pathlib import Path
# Importa função para carregar datasets do HuggingFace
from datasets import load_dataset
# Importa IPython para exibir áudio no notebook
import IPython

# Importa pipeline da HuggingFace Transformers para tarefas de áudio
from transformers import pipeline

# Importa sounddevice para gravação de áudio pelo microfone
import sounddevice as sd

# Define o modelo Whisper para reconhecimento de fala
modelo = 'openai/whisper-medium'

# Cria o pipeline de reconhecimento automático de fala, configurado para transcrição em português
reconhecedor_de_fala = pipeline(
    'automatic-speech-recognition',
    model=modelo,
    generate_kwargs={"task": "transcribe", "language": "portuguese"},
)

# Define a duração da gravação em segundos
duracao = 10
# Define a taxa de amostragem do áudio
taxa_amostragem = 16000
# Calcula o tamanho do vetor de áudio
tamanho_vetor = int(duracao * taxa_amostragem)

# Grava o áudio do microfone
gravacao = sd.rec(tamanho_vetor, samplerate=taxa_amostragem, channels=1)
sd.wait()

gravacao

# Realiza o reconhecimento de fala no áudio gravado
reconhecedor_de_fala(
    {'raw': gravacao.ravel(), 'sampling_rate': taxa_amostragem})
