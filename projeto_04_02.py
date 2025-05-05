# Acelerado com GPU - ainda não finalizei , está demorando mt de carregar acredito  que se
# Importa pipeline da HuggingFace Transformers para tarefas de áudio
from transformers import pipeline
# Importa IPython para exibir áudio no notebook
import IPython
# Importa módulo de tempo para medir duração da geração de áudio
import time

# Importa torch para uso de GPU e tipos de dados
import torch

# Define o dispositivo: usa GPU se disponível, senão CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define o modelo Bark pequeno para texto para fala
modelo = "suno/bark-small"

# Cria o pipeline de texto para fala, usando float16 para acelerar na GPU
leitor = pipeline(
    'text-to-speech',
    model=modelo,
    model_kwargs={'torch_dtype': torch.float16},
    forward_params={'max_new_tokens': 50})

# Move o modelo para o dispositivo correto e aplica otimizações
device = "cuda" if torch.cuda.is_available() else "cpu"
leitor.model = leitor.model.to(device)
leitor.model = leitor.model.to_bettertransformer()
leitor.model.enable_cpu_offload()

# Texto a ser convertido em áudio
texto = '''
            Meu nome é Rei Jardim, sou graduando do Senai Cimatec e estou formando nesse semestre, 2005.1. Então é isso que eu tenho para dizer agora.'

        '''
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
