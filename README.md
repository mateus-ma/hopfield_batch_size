# CABGen Hopfield

Este repositório tem como objetivo desenvolver um modelo para prever o perfil de resistência bacteriana com base em ORFs (Open Reading Frames). O modelo utiliza como base a biblioteca [DeepRC](https://github.com/ml-jku/DeepRC) e é implementado com suporte ao [Hydra](https://hydra.cc/), o que permite a configuração flexível dos parâmetros e a reprodutibilidade dos experimentos.

---

## Instalação

### Requisitos

- **Python versão 3.10.12**

Para configurar o ambiente e instalar todas as dependências necessárias, execute o seguinte comando:

```bash
make setup
```

---

## Configurações do Hydra

As configurações do Hydra permitem gerenciar facilmente os parâmetros do treinamento e teste. A seguir, apresentamos uma descrição dos principais arquivos de configuração e seus parâmetros.

### Arquivos de Configuração

1. **`config.yaml`**  
   Configuração principal, inclui referências para os outros arquivos de configuração e parâmetros globais.

   - **`device`**: Define o dispositivo para execução do modelo. Exemplo: `"cuda:0"` para usar GPU.  
   - **`rnd_seed`**: Define a semente para inicialização aleatória, garantindo reprodutibilidade.  
   - **`results_directory`**: Diretório onde os resultados do treinamento serão salvos.  
   - **`database`**: Contém caminhos para os arquivos de entrada:
     - **`metadata_file`**: Arquivo TSV com IDs e perfis de resistência.
     - **`repertoiresdata_path`**: Diretório com arquivos de ORFs.

2. **`data_splitting.yaml`**  
   Configuração da divisão dos dados.
   - **`stratify`**: Se verdadeiro, realiza divisão estratificada para garantir equilíbrio entre as classes.  
   - **`metadata_file_id_column`**: Nome da coluna no metadata com os IDs das amostras.  
   - **`sequence_column`**: Nome da coluna com as sequências de ORFs nos arquivos de entrada.  
   - **`sequence_counts_column`**: Coluna que indica a quantidade de vezes que cada ORF aparece.  
   - **`sample_n_sequences`**: Número de sequências a serem amostradas por genoma durante o treinamento. Use `0` para carregar todas as sequências.

3. **`model.yaml`**  
   Configuração da arquitetura do modelo.
   - **`kernel_size`**: Tamanho dos kernels da CNN (quantidade de aminoácidos por filtro).  
   - **`n_kernels`**: Número de filtros na CNN, ou seja, padrões que serão aprendidos.  
   - **`sequence_embedding`**:
     - **`n_layers`**: Número de camadas na CNN.  
   - **`attention`**:
     - **`n_layers`**: Número de camadas na rede de atenção.
     - **`n_units`**: Número de neurônios na rede de atenção.
   - **`output`**:
     - **`n_layers`**: Número de camadas na rede de saída.
     - **`n_units`**: Número de neurônios na rede de saída.

4. **`task.yaml`**  
   Configuração das tarefas.
   - **`targets`**: Define as tarefas de classificação:
     - **`type`**: Tipo da tarefa, como `binary`.  
     - **`column_name`**: Nome da coluna do metadata que contém os rótulos da tarefa.  
     - **`positive_class`**: Para tarefas binárias, define a classe positiva.  
     - **`pos_weight`**: Peso aplicado à classe positiva para lidar com desbalanceamento.  
     - **`task_weight`**: Peso dessa tarefa na função de perda total (se houver múltiplas tarefas).

5. **`training.yaml`**  
   Configuração do treinamento.
   - **`n_updates`**: Número de atualizações (steps) durante o treinamento.  
   - **`evaluate_at`**: Frequência (em steps) para avaliar o modelo nos conjuntos de validação e treinamento.  
   - **`learning_rate`**: Taxa de aprendizado do otimizador Adam.

6. **`test.yaml`**  
   Configuração para avaliação do modelo.
   - **`model_path`**: Caminho do modelo treinado para ser avaliado. Padrão: `"results"`.  
   - **`metadata_file`**: Caminho para o metadata de teste.  
   - **`orfs_path`**: Caminho para os arquivos de ORFs do conjunto de teste.

---

## Execução do Modelo

### Treinamento com Valores Padrão

Para treinar o modelo usando os valores padrão do Hydra:

```bash
make run
```

### Alterando Parâmetros pela Linha de Comando

Você pode sobrescrever os valores padrão diretamente no terminal. Exemplos:

- Ajustar a taxa de aprendizado e o tamanho do kernel:

  ```bash
  python3 cabgen_hopfield_main.py training.learning_rate=1e-5 model.kernel_size=30
  ```

- Criando combinações de vários parâmetros, além de executar cada combinação sequencialmente e com log:

  ```bash
  python3 cabgen_hopfield_main.py -m model.kernel_size=30,45 training.learning_rate=0.00005 training.n_updates=10000 training.evaluate_at=100,200,500 data_splitting.sample_n_sequences=0 +hydra.job.logging=debug hydra.verbose=true hydra/launcher=basic
  ```

---

## Teste do Modelo

### Script `test_model.py`

O script `test_model.py` avalia o modelo treinado em novos dados. Ele utiliza as configurações definidas em `config/test.yaml` e por padrão avalia o modelo mais recente salvo em `results`. Para especificar um modelo diferente, ajuste o parâmetro `test.model_path`.

#### Exemplos de Execução

1. Avaliar o modelo mais recente dentro da pasta `results/`:

   ```bash
   python3 test_model.py
   ```

2. Avaliar um modelo específico com kernel size de 48:

   ```bash
   python3 test_model.py test.model_path="results/model_2050125/checkpoint/model.zip" model.kernel_size=48
   ```

3. Alterar o metadata de teste:

   ```bash
   python3 test_model.py test.metadata_file="new_metadata.tsv"
   ```

---

## Predição do Conjunto de Teste

### Script `model_prediction.py`

O script `model_prediction.py` gera, por padrão, uma tabela com os valores de predição do modelo mais recente armazenado na pasta `results/`, utilizando os dados de teste. Se desejar utilizar um modelo específico, defina o caminho do arquivo no parâmetro `test.model_path`. Além disso, outros parâmetros podem ser ajustados conforme necessário, basta verificar quais são utilizados pelo script e sobrescrevê-los diretamente na linha de comando.

1. Executar a predição utilizando o modelo mais recente:

   ```bash
   python3 model_prediction.py
   ```

2. Executar a predição utilizando um modelo específico e definir kernel_size como 48:

   ```bash
   python3 model_prediction.py test.model_path="results/model_2050125/checkpoint/model.zip" model.kernel_size=48 
   ```

---

## 5 Fold Cross Validation

### Script `fold_cross_validation.py`

O script `fold_cross_validation.py` executa, por padrão, a validação cruzada para medir a estabilidade das métricas do modelo. Ele divide o conjunto de dados em 5 subgrupos e treina 5 modelos variando o subconjunto de teste/validação. Além disso, outros parâmetros podem ser ajustados conforme necessário, basta verificar quais são utilizados pelo script e sobrescrevê-los diretamente na linha de comando.

1. Executar a validação com as configurações padrão:

   ```bash
   python3 fold_cross_validation.py
   ```

2. Executar a validação com um kernel_size de 48 e n_updates de 10.000:

   ```bash
   python3 model_prediction.py model.kernel_size=48 training.n_updates=10000
   ```

---

## Estrutura do Modelo

1. **Divisão dos Dados**:  
   A função `make_dataloaders_stratified` divide os dados para treinamento, validação e teste, garantindo equilíbrio entre as classes caso a opção `stratify` esteja habilitada.

2. **Arquitetura do Modelo**:  
   O modelo DeepRC integra três redes principais:
   - **CNN**: Captura padrões em sequências de ORFs.
   - **Rede de Atenção**: Destaca as regiões mais relevantes das ORFs.
   - **Rede de Saída**: Classifica as amostras nas classes de interesse.

3. **Treinamento**:  
   O treinamento ajusta os pesos das redes para minimizar a perda.

4. **Avaliação**:  
   A avaliação fornece métricas como ROC AUC, F1-score, Loss e acurácia balanceada (BACC).
