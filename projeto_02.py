from pathlib import Path
from datasets import load_dataset
import IPython
import soundfile
from transformers import pipeline


pasta_saida = Path('audios') / 'vozes'
pasta_saida.mkdir(exist_ok=True)


nome_dataset = "google/fleurs"
lingua_dataset = "pt_br"
dados = load_dataset(nome_dataset, name=lingua_dataset,
                     split='train', streaming=True)

primeiras_linhas = dados.take(5)

for linha in primeiras_linhas:
    print(linha)
    print('-----')

# https://huggingface.co/sanchit-gandhi/whisper-medium-fleurs-lang-id

# Dessa forma  o modelo de classificação vai está rodando diretamente do computador
modelo = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'
classificador = pipeline('audio-classification', model=modelo)
classificador

classificador.feature_extractor.sampling_rate
