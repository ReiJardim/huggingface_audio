
from transformers import pipeline
import IPython
import time


# fonte do modelo : https://huggingface.co/suno/bark-small

modelo = 'suno/bark-small'
leitor = pipeline('text-to-speech', model=modelo,
                  forward_params={'max_new_tokens': 50})


texto = 'Meu nome é Rei Jardim, sou graduando do Senai Cimatec e estou formando nesse semestre, 2005.1. Então é isso que eu tenho para dizer agora.'

inicio = time.time()
fala = leitor(texto)
final = time.time()

print(f'levou {final - inicio:.03f} segundos para gerar o áudio')
print(fala)

IPython.display.Audio(data=fala['audio'], rate=fala['sampling_rate'])
