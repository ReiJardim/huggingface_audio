# Gerar audio a partir de texto

from transformers import pipeline
import IPython
import time

modelo = "facebook/mms-tts-por"
leitor = pipeline('text-to-speech', model=modelo)


texto = 'Meu nome é Rei Jardim, sou graduando do Senai Cimatec e estou formando nesse semestre, 2005.1. Então é isso que eu tenho para dizer agora.'

inicio = time.time()
fala = leitor(texto)
final = time.time()


print(f'levou {final - inicio:.02f} segundos para gerar o áudio')
print(fala)


IPython.display.Audio(data=fala['audio'], rate=fala['sampling_rate'])
