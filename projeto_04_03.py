# Importa o PyTorch para operações com tensores e uso de GPU
import torch
# Importa classes da HuggingFace Transformers para processamento e modelo de áudio
from transformers import AutoProcessor, AutoModel
# Importa módulo de tempo para medir duração da geração de áudio
import time
# Importa IPython para exibir áudio no notebook
import IPython

# Link para vozes disponíveis do modelo Bark
# Vozes disponíveis: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c

# Define o nome do modelo Bark pequeno
modelo = 'suno/bark-small'

# Define o dispositivo: usa GPU se disponível, senão CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega o processador (tokenizador e pré-processador) do modelo
processador = AutoProcessor.from_pretrained(modelo)
# Carrega o modelo de áudio com precisão float16
leitor = AutoModel.from_pretrained(modelo, torch_dtype=torch.float16)

# Otimizações para acelerar e economizar memória
leitor = leitor.to(device)  # Move o modelo para o dispositivo (GPU/CPU)
leitor = leitor.to_bettertransformer()  # Usa otimizações do BetterTransformer
leitor.enable_cpu_offload()  # Permite offload para CPU se necessário

# Texto a ser sintetizado em áudio
texto = 'A minha resposta anterior não foi clara. Quando eu disse "Preparei o áudio", quis dizer que processei seu pedido e entendi o que você precisava, mas eu não tenho a capacidade de gerar ou anexar arquivos de áudio diretamente aqui na nossa conversa. Sou um modelo de linguagem baseado em texto.'
# Define o preset de voz a ser usado (há outras opções comentadas)
# (v2/pt_speaker_1,v2/pt_speaker_5, v2/pt_speaker_8,v2/pt_speaker_9 )
voz = 'v2/pt_speaker_9'

# Marca o tempo inicial da geração
inicio = time.time()
# Processa o texto e voz para entrada do modelo
inputs = processador(texto, voice_preset=voz, return_tensors="pt")
# Move os tensores de entrada para o dispositivo correto
inputs = {k: v.to(device) for k, v in inputs.items()}
# Gera o vetor de áudio a partir do texto
vetor_audio = leitor.generate(**inputs)
# Monta o dicionário com o áudio e taxa de amostragem
fala = {
    'audio': vetor_audio.cpu().numpy(),
    'sampling_rate': leitor.generation_config.sample_rate,
}

# Marca o tempo final da geração
final = time.time()

# Exibe quanto tempo levou para gerar o áudio
print(f'levou {final - inicio:.03f} segundos para gerar o áudio')
# Exibe o dicionário com áudio e taxa de amostragem
print(fala)

# Toca o áudio gerado no notebook
IPython.display.Audio(data=fala['audio'], rate=fala['sampling_rate'])
