# Importa pipeline da HuggingFace Transformers para tarefas de áudio
from transformers import pipeline
# Importa IPython para exibir áudio no notebook
import IPython
# Importa módulo de tempo para medir duração da geração de áudio
import time

# fonte do modelo : https://huggingface.co/suno/bark-small

# Define o modelo Bark pequeno para texto para fala
modelo = 'suno/bark-small'
# Cria o pipeline de texto para fala, limitando o número de tokens gerados
leitor = pipeline('text-to-speech', model=modelo,
                  forward_params={'max_new_tokens': 50})

# Texto a ser convertido em áudio
texto = 'Meu nome é Rei Jardim, sou graduando do Senai Cimatec e estou formando nesse semestre, 2005.1. Então é isso que eu tenho para dizer agora.'

# Marca o tempo inicial da geração
inicio = time.time()
# Gera o áudio a partir do texto
fala = leitor(texto)
# Marca o tempo final da geração
final = time.time()

# Exibe quanto tempo levou para gerar o áudio
print(f'levou {final - inicio:.03f} segundos para gerar o áudio')
# Exibe o dicionário com áudio e taxa de amostragem
print(fala)

# Toca o áudio gerado no notebook
IPython.display.Audio(data=fala['audio'], rate=fala['sampling_rate'])
