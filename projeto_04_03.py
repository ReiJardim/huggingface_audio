import torch
from transformers import AutoProcessor, AutoModel
import time
import IPython

# Vozes disponíveis: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c

modelo = 'suno/bark-small'

device = "cuda" if torch.cuda.is_available() else "cpu"

processador = AutoProcessor.from_pretrained(modelo)
leitor = AutoModel.from_pretrained(modelo, torch_dtype=torch.float16)

# Otimizações
leitor = leitor.to(device)
leitor = leitor.to_bettertransformer()
leitor.enable_cpu_offload()

texto = 'A minha resposta anterior não foi clara. Quando eu disse "Preparei o áudio", quis dizer que processei seu pedido e entendi o que você precisava, mas eu não tenho a capacidade de gerar ou anexar arquivos de áudio diretamente aqui na nossa conversa. Sou um modelo de linguagem baseado em texto.'
# (v2/pt_speaker_1,v2/pt_speaker_5, v2/pt_speaker_8,v2/pt_speaker_9 )
voz = 'v2/pt_speaker_9'

inicio = time.time()
inputs = processador(texto, voice_preset=voz, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
vetor_audio = leitor.generate(**inputs)
fala = {
    'audio': vetor_audio.cpu().numpy(),
    'sampling_rate': leitor.generation_config.sample_rate,
}

final = time.time()

print(f'levou {final - inicio:.03f} segundos para gerar o áudio')
print(fala)

IPython.display.Audio(data=fala['audio'], rate=fala['sampling_rate'])
