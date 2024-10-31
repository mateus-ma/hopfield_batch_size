# CABGen Hopfield

Este repositório tem como objetivo desenvolver um modelo para prever o perfil de resistência bacteriana com base em ORFs (Open Reading Frames). O modelo utiliza como base a biblioteca [DeepRC](https://github.com/ml-jku/DeepRC) e é implementado no arquivo `cabgen_hopfield_main.py`, seguindo um dos exemplos fornecidos no repositório DeepRC.

## Instalação

### Requisitos

- **Python versão 3.10.12**

Para configurar o ambiente e instalar todas as dependências necessárias, execute o seguinte comando:

```bash
make setup
```

### Resolução de Problemas de Dependências

Se ocorrerem erros relacionados a incompatibilidades de versão durante a execução do comando acima, siga os passos abaixo:

1. Instale o `pip-tools` manualmente:

   ```bash
   pip install pip-tools
   ```

2. Gere um novo arquivo `requirements.txt` a partir do arquivo `requirements.in`:

   ```bash
   pip-compile requirements.in
   ```

3. Após gerar o novo `requirements.txt`, execute novamente o comando de configuração:

   ```bash
   make setup
   ```

## Dados de Entrada

Os dados de entrada estão localizados no diretório `database`. A estrutura dos arquivos é descrita a seguir:

- **Arquivo de metadata** (`metadata.tsv`): Contém o perfil de resistência de cada bactéria e o ID correspondente ao arquivo que contém as ORFs da mesma.

- **Arquivos de genomas bacterianos** (`database/orfs`): Contêm as ORFs de cada genoma bacteriano, organizadas em arquivos TSV. Cada arquivo possui duas colunas principais:
  - `orf`: Sequências de ORFs.
  - `templates`: Quantidade de vezes que cada ORF aparece no genoma.

## Execução do Modelo

**⚠️ Aviso: O modelo ainda não pode ser executado pois não há dados suficientes para a análise.**

Assim que os dados estiverem disponíveis, o modelo poderá ser executado com esse comando:

```bash
make run
```

## Overview do Código

### Definição dos parâmetros da rede

Nesta seção, os argumentos definem parâmetros como o número de iterações de treinamento (n_updates), a frequência de avaliação (evaluate_at), tamanho e quantidade de kernels para a CNN, taxa de aprendizado, entre outros.

```python
parser = argparse.ArgumentParser()
parser.add_argument("--n_updates", help=(
    "Number of updates to train for. "
    "Recommended: int(1e5). Default: int(1e3)"),
    type=int, default=int(1e3))
parser.add_argument("--evaluate_at", help=(
    "Evaluate model on training and validation set every `evaluate_at` "
    "updates. This will also check for a new best model for early stopping. "
    "Recommended: int(5e3). Default: int(1e2)."),
    type=int, default=int(1e2))
parser.add_argument("--kernel_size", help=(
    "Size of 1D-CNN kernels (=how many sequence characters a "
    "CNN kernel spans). Default: 9"),
    type=int, default=9)
parser.add_argument("--n_kernels", help=(
    "Number of kernels in the 1D-CNN. This is an important hyper-parameter. "
    "Default: 32"),
    type=int, default=32)
parser.add_argument("--sample_n_sequences", help=(
    "Number of instances to reduce repertoires to during training via"
    "random dropout. This should be less than the number of instances per "
    "repertoire. Only applied during training, not for evaluation. "
    "Default: int(1e4)"),
    type=int, default=int(1e4))
parser.add_argument("--learning_rate", help=(
    "Learning rate of DeepRC using Adam optimizer. Default: 1e-4"),
    type=float, default=1e-4)
parser.add_argument("--device", help=(
    "Device to use for NN computations, as passed to `torch.device()`. "
    "Default: 'cuda:0'."),
    type=str, default="cuda:0")
parser.add_argument("--rnd_seed", help=(
    "Random seed to use for PyTorch and NumPy. Results will still be "
    "non-deterministic due to multiprocessing but weight initialization will "
    "be the same). Default: 0."),
    type=int, default=0)
args = parser.parse_args()
```

### Configuração do dispositivo e semente aleatória

Essa etapa define qual dispositivo será utilizado (CPU ou GPU) para o processamento. Além da seed que garante que todos os pesos vão inicializar sempre com os mesmos valores.

```python
device = torch.device(args.device)
torch.manual_seed(args.rnd_seed)
np.random.seed(args.rnd_seed)
```

### Definição das tarefas

Nesta parte, a tarefa de classificação é definida. Usando MulticlassTarget, o modelo será treinado para prever classes mutuamente exclusivas (neste caso, resistência de bactérias a antibióticos: R para resistente, I para intermediário, e S para sensível). Cada classe recebe um peso, ajustando a importância relativa para classes desbalanceadas.

```python
task_definition = TaskDefinition(targets=[
    MulticlassTarget(
        column_name="MEM",
        possible_target_values=["R", "I", "S"],
        class_weights=[1., 1., 1.],
        task_weight=1
    )
]).to(device=device)
```

### Divisão dos dados

A função make_dataloaders carrega os dados e retorna loaders para os conjuntos de treinamento, validação e teste. Os dados são compostos por sequências de ORFs e suas respectivas frequências, e metadados associados, como o ID.

```python
trainingset, trainingset_eval, \
    validationset_eval, testset_eval = make_dataloaders(
        task_definition=task_definition,
        metadata_file=path.abspath("database/metadata.tsv"),
        repertoiresdata_path=path.abspath("database/orfs"),
        metadata_file_id_column="ID",
        sequence_column="orf",
        sequence_counts_column="templates",
        sample_n_sequences=args.sample_n_sequences,
        sequence_counts_scaling_fn=no_sequence_count_scaling
    )
```

### CNN

A CNN é responsável por "escanear" a sequência em busca de padrões específicos. O parâmetro `kernel_size` define quantos aminoácidos serão considerados em cada subsequência examinada pela CNN. Se certos padrões nessas subsequências estiverem correlacionados com a resistência, os filtros da CNN os capturam e destacam como importantes para a predição.

```python
sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20+3, kernel_size=args.kernel_size, 
n_kernels=args.n_kernels, n_layers=1)
```

- **n_input_features:** Cada aminoácido é representado por 20 posições em uma codificação one-hot, mais 3 posições adicionais que indicam sua localização relativa na sequência (início, meio ou fim).
- **kernel_size:** Número de aminoácidos processados juntos em cada subsequência para identificação de padrões.
- **n_kernels:** Número total de filtros da CNN, representando a quantidade de padrões diferentes que serão aprendidos.
- **n_layers:** Número de camadas da CNN, ou seja, quantas vezes a entrada passará por filtros para aprimorar a detecção de padrões complexos.

### Rede de atenção

A rede de atenção permite que o modelo identifique partes importantes das sequências de aminoácidos que podem estar associadas à resistência. Esse mecanismo atribui pesos variáveis às saídas da CNN, destacando regiões específicas da sequência que possuem um impacto nas predições de resistência.

```python
attention_network = AttentionNetwork(n_input_features=args.n_kernels, n_layers=2, n_units=32)
```

- **n_input_features:** Este parâmetro indica o número de características de entrada da rede de atenção, correspondente ao número de filtros da CNN.
- **n_layers:** Refere-se ao número de camadas na rede de atenção.
- **n_units:** Este parâmetro especifica o número de neurônios em cada camada.

### Rede de saída

A rede de saída mapeia as características para as classes de saída, sendo no presente momento, as classes "R", "I", e "S" para a coluna MEM.

```python
output_network = OutputNetwork(
    n_input_features=args.n_kernels,
    n_output_features=task_definition.get_n_output_features(),
    n_layers=1, n_units=32)
```

### Modelo DeepRC

O modelo DeepRC integra todas as sub-redes. Ele adiciona informações posicionais, e configura a redução de sequência para reduzir a memória ocupada durante o treinamento.

```python
model = DeepRC(max_seq_len=30,
               sequence_embedding_network=sequence_embedding_network,
               attention_network=attention_network,
               output_network=output_network,
               consider_seq_counts=False, n_input_features=20,
               add_positional_information=True,
               sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
               device=device).to(device=device)
```

### Treinamento

O treinamento é executado pela função train, que usa o conjunto de treinamento e a taxa de aprendizado especificada. O parâmetro early_stopping_target_id configura o modelo para interromper o treinamento quando a performance sobre a classe "MEM" deixar de melhorar.

```python
train(model, task_definition=task_definition,
      trainingset_dataloader=trainingset,
      trainingset_eval_dataloader=trainingset_eval,
      learning_rate=args.learning_rate,
      early_stopping_target_id="MEM",
      validationset_eval_dataloader=validationset_eval,
      n_updates=args.n_updates, evaluate_at=args.evaluate_at,
      device=device, results_directory="results"
      )
```

### Avaliação

Finalmente, o modelo é avaliado no conjunto de testes.

```python
scores = evaluate(model=model, dataloader=testset_eval,
                  task_definition=task_definition, device=device)
```
