# Importa Path para manipulação de caminhos de arquivos
from pathlib import Path
# Importa função para carregar datasets do HuggingFace
from datasets import load_dataset
# Importa IPython para exibir áudio no notebook
import IPython
# Importa soundfile para salvar arquivos de áudio
import soundfile
# Importa pipeline da HuggingFace Transformers para tarefas de áudio
from transformers import pipeline

# Cria a pasta de saída para salvar os áudios de vozes
pasta_saida = Path('audios') / 'vozes'
pasta_saida.mkdir(exist_ok=True)

# Define o nome do dataset e o idioma
nome_dataset = "google/fleurs"
lingua_dataset = "pt_br"

# O streaming=True permite rodar apenas uma instância sem baixar o dataset completo
dados = load_dataset(nome_dataset, name=lingua_dataset,
                     split='train', streaming=True)

# Seleciona as 5 primeiras linhas do dataset para inspeção
primeiras_linhas = dados.take(5)

# Exibe as primeiras linhas do dataset
for linha in primeiras_linhas:
    print(linha)
    print('-----')

# https://huggingface.co/sanchit-gandhi/whisper-medium-fleurs-lang-id

# Dessa forma o modelo de classificação vai estar rodando diretamente do computador
modelo = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'
# Cria o pipeline de classificação de áudio
classificador = pipeline('audio-classification', model=modelo)
classificador

# Exibe a taxa de amostragem esperada pelo feature extractor do classificador
classificador.feature_extractor.sampling_rate

# Converte as primeiras linhas para lista para reutilização
primeiras_linhas = list(primeiras_linhas)

# Classifica as 5 primeiras amostras de áudio
for linha in primeiras_linhas:
    predicao = classificador(linha['audio']['array'])
    # display(predicao)


##################################
# Teste completo

# Repete o carregamento do dataset para o teste completo
nome_dataset = "google/fleurs"
lingua_dataset = "pt_br"
dados = load_dataset(nome_dataset, name=lingua_dataset,
                     split='train', streaming=True)

# Repete a criação do pipeline de classificação de áudio
modelo = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'
classificador = pipeline('audio-classification', model=modelo)

# Classifica e exibe as 5 primeiras amostras do dataset
for linha in dados.take(5):
    predicao = classificador(linha["audio"]["array"])
    print(predicao)
    print('-----')
