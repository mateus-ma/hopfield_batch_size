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
