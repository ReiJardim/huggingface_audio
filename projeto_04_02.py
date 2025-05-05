# Acelerado com GPU - ainda não finalizei , está demorando mt de carregar acredito  que se
from transformers import pipeline
import IPython
import time


import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


modelo = "suno/bark-small"

leitor = pipeline(
    'text-to-speech',
    model=modelo,
    model_kwargs={'torch_dtype': torch.float16},
    forward_params={'max_new_tokens': 50})

device = "cuda" if torch.cuda.is_available() else "cpu"
leitor.model = leitor.model.to(device)
leitor.model = leitor.model.to_bettertransformer()
leitor.model.enable_cpu_offload()

texto = '''
            Meu nome é Rei Jardim, sou graduando do Senai Cimatec e estou formando nesse semestre, 2005.1. Então é isso que eu tenho para dizer agora.'

        '''
inicio = time.time()
fala = leitor(texto)
final = time.time()

print(f'levou {final - inicio:.03f} segundos para gerar o áudio')
print(fala)

IPython.display.Audio(data=fala['audio'], rate=fala['sampling_rate'])
