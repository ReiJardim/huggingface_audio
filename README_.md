# Projetos de Áudio com HuggingFace, Transformers e Python
  
Este repositório reúne diversos experimentos e projetos envolvendo manipulação, classificação, reconhecimento e síntese de áudio utilizando bibliotecas modernas como HuggingFace Transformers, Datasets, PyTorch, Librosa, SoundDevice, entre outras.
  
## Índice
- [Visão Geral dos Projetos](#visão-geral-dos-projetos )
- [Fontes dos Dados](#fontes-dos-dados )
- [Instalação e Dependências](#instalação-e-dependências )
- [Requisitos de Memória](#requisitos-de-memória )
- [Tópicos Importantes](#tópicos-importantes )
- [Tabela Comparativa dos Scripts](#tabela-comparativa-dos-scripts )
- [Observações e Boas Práticas](#observações-e-boas-práticas )
  
---
  
## Visão Geral dos Projetos
  
| Script                | Objetivo Principal                                                                 | Tarefa                        |
|-----------------------|-----------------------------------------------------------------------------------|-------------------------------|
| `projeto_01.py`       | Exploração, visualização e salvamento de áudios do dataset ESC-50                 | Análise e visualização        |
| `projeto_02.py`       | Classificação de áudios do dataset FLEURS usando modelo Whisper                   | Classificação de áudio        |
| `projeto_02_01.py`    | Gravação de áudio pelo microfone e classificação com Whisper                      | Gravação e classificação      |
| `projeto_03.py`       | Reconhecimento de fala em português com Whisper usando dataset minds14            | Reconhecimento de fala        |
| `projeto_03_01.py`    | Gravação de áudio próprio e transcrição automática com Whisper                    | Gravação e transcrição        |
| `projeto_04.py`       | Síntese de fala em português com MMS-TTS                                          | Texto para fala (TTS)         |
| `projeto_04_01.py`    | Síntese de fala com Bark (modelo pequeno)                                         | Texto para fala (TTS)         |
| `projeto_04_02.py`    | TTS com Bark acelerado por GPU                                                    | Texto para fala (TTS)         |
| `projeto_04_03.py`    | TTS com Bark usando API de baixo nível e otimizações                              | Texto para fala (TTS)         |
  
---
  
## Fontes dos Dados
  
- **ESC-50**: [ashraq/esc50](https://huggingface.co/datasets/ashraq/esc50 ) — Sons ambientais organizados em 50 categorias.
- **FLEURS**: [google/fleurs](https://huggingface.co/datasets/google/fleurs ) — Áudios de fala em múltiplos idiomas.
- **MINDS-14**: [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14 ) — Comandos de voz em vários idiomas.
- **Modelos TTS**:
  - [facebook/mms-tts-por](https://huggingface.co/facebook/mms-tts-por )
  - [suno/bark-small](https://huggingface.co/suno/bark-small )
- **Modelos ASR**:
  - [openai/whisper-medium](https://huggingface.co/openai/whisper-medium )
  - [sanchit-gandhi/whisper-medium-fleurs-lang-id](https://huggingface.co/sanchit-gandhi/whisper-medium-fleurs-lang-id )
  
---
  
## Instalação e Dependências
  
**Principais bibliotecas utilizadas:**
- `transformers`
- `datasets`
- `torch`
- `librosa`
- `matplotlib`
- `soundfile`
- `sounddevice`
- `ipython`
- `numpy`
  
**Instalação sugerida:**
```bash
pip install -r requirements.txt
```
  
> **Nota:** Para uso de GPU, é necessário instalar o PyTorch com suporte CUDA. Veja instruções em: https://pytorch.org/get-started/locally/
  
---
  
## Requisitos de Memória
  
| Script                | Memória RAM Recomendada | GPU Necessária? | Observações Importantes                      |
|-----------------------|------------------------|-----------------|----------------------------------------------|
| `projeto_01.py`       | 2 GB                   | Não             | Download inicial do dataset pode ser grande  |
| `projeto_02.py`       | 4 GB                   | Não             |                                              |
| `projeto_02_01.py`    | 4 GB                   | Não             | Microfone necessário                         |
| `projeto_03.py`       | 6 GB                   | Opcional        | Modelos Whisper são pesados                  |
| `projeto_03_01.py`    | 6 GB                   | Opcional        | Microfone necessário                         |
| `projeto_04.py`       | 4 GB                   | Não             |                                              |
| `projeto_04_01.py`    | 6 GB                   | Opcional        |                                              |
| `projeto_04_02.py`    | 8 GB                   | Sim             | Uso intensivo de GPU                         |
| `projeto_04_03.py`    | 8 GB                   | Sim             | Uso intensivo de GPU e otimizações           |
  
> **Dica:** Scripts de TTS e ASR com modelos grandes (Whisper, Bark) podem consumir muita RAM e VRAM. Para uso confortável, recomenda-se pelo menos 8 GB de RAM e, se possível, uma GPU com 6 GB+ de VRAM.
  
---
  
## Tópicos Importantes
  
### 1. **Sample Rate (Taxa de Amostragem)**
- **Por que é importante?**  
  Modelos de áudio geralmente esperam uma taxa de amostragem específica (ex: 16kHz). Usar taxas diferentes pode causar erros ou resultados ruins.
- **Como garantir compatibilidade?**  
  Sempre confira a documentação do modelo e, se necessário, converta o áudio usando `librosa` ou `soundfile`.
  
### 2. **Formato dos Dados**
- Certifique-se de que os arrays de áudio estejam no formato e shape esperados (ex: 1D para modelos HuggingFace).
- Para gravação com microfone, pode ser necessário usar `.reshape(-1)` ou `.ravel()`.
  
### 3. **Uso de GPU**
- Scripts que usam modelos grandes (Whisper, Bark) são muito mais rápidos e eficientes em GPU.
- Certifique-se de que o PyTorch está instalado com suporte CUDA.
  
### 4. **Cuidados com Dados**
- Sempre valide o conteúdo dos datasets antes de treinar ou inferir.
- Atenção ao idioma, formato e integridade dos arquivos de áudio.
  
### 5. **Limitações dos Modelos TTS**
- **Bark (suno/bark-small)**:
  - Limitação de duração: aproximadamente 14 segundos por geração
  - Limitação de tokens: por padrão, gera até 50 tokens (pode ser ajustado com `max_new_tokens`)
  - Para textos longos, é necessário dividir em segmentos menores
  - Qualidade pode variar dependendo do idioma e do preset de voz escolhido
- **MMS-TTS (facebook/mms-tts-por)**:
  - Melhor para textos em português
  - Mais rápido que o Bark
  - Menos natural que o Bark
  - Também tem limitações de tamanho de texto
  
### 6. **Estratégias para Textos Longos**
- Dividir o texto em segmentos menores (frases ou parágrafos)
- Gerar áudio para cada segmento
- Concatenar os áudios usando bibliotecas como `soundfile` ou `librosa`
- Ajustar o parâmetro `max_new_tokens` conforme necessário
  
---
  
## Tabela Comparativa dos Scripts
  
| Script             | Dataset/Modelo Principal         | Tarefa                  | Entrada Esperada         | Saída Principal         | Observações                        |
|--------------------|---------------------------------|-------------------------|-------------------------|-------------------------|-------------------------------------|
| projeto_01.py      | ESC-50                          | Visualização/Exportação | Dataset                 | Gráficos, WAVs          | Requer matplotlib, soundfile        |
| projeto_02.py      | FLEURS + Whisper                | Classificação           | Dataset                 | Predições               | Streaming de dados                  |
| projeto_02_01.py   | Microfone + Whisper             | Gravação/Classificação  | Microfone               | Predições               | Necessita microfone                 |
| projeto_03.py      | MINDS-14 + Whisper              | Reconhecimento de fala  | Dataset                 | Texto transcrito        |                                     |
| projeto_03_01.py   | Microfone + Whisper             | Gravação/Transcrição    | Microfone               | Texto transcrito        | Necessita microfone                 |
| projeto_04.py      | MMS-TTS                         | Texto para fala         | Texto                   | Áudio                   | Melhor para português               |
| projeto_04_01.py   | Bark                            | Texto para fala         | Texto                   | Áudio                   | Limite de ~14s por geração          |
| projeto_04_02.py   | Bark (GPU)                      | Texto para fala         | Texto                   | Áudio                   | Otimizado para GPU, limite de ~14s  |
| projeto_04_03.py   | Bark (baixo nível, GPU)         | Texto para fala         | Texto                   | Áudio                   | Otimizações avançadas, limite de ~14s|
  
---
  
## Observações e Boas Práticas
  
- **Ambiente Virtual:** Recomenda-se o uso de ambientes virtuais (`venv` ou `conda`) para evitar conflitos de dependências.
- **Drivers de Áudio:** Para scripts que usam microfone, certifique-se de que os drivers estejam corretamente instalados.
- **Jupyter/IPython:** Para visualização e reprodução de áudio inline, use Jupyter Notebook ou IPython.
- **GPU:** Scripts pesados podem travar em máquinas sem GPU ou com pouca RAM/VRAM.
  
---
  
## Dúvidas?
  
Abra uma issue ou entre em contato! 
  