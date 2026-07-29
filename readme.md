# Deriva — Simulador de Deriva de Óleo

Software de modelagem da dispersão de manchas de óleo no oceano, construído
sobre o OpenDrift/OpenOil. A partir de dados de corrente marinha, vento e
propriedades do óleo, o simulador prevê para onde uma mancha vai, quanto tempo
leva e qual área tem risco de ser atingida.

O sistema oferece **três modos de execução**, e a diferença entre eles está em
como o vazamento é definido e em qual pergunta cada um responde.

| Modo | Ponto de partida | Responde |
|---|---|---|
| **Simulação de Casos** | manchas observadas em shapefile | "meus parâmetros reproduzem o que foi observado?" |
| **Simulação Determinística** | ponto, data e duração digitados | "se vazar aqui agora, para onde vai?" |
| **Simulação Estocástica** | manchas observadas + incerteza | "qual a probabilidade de cada região ser atingida?" |

---

## Sumário

- [Instalação e execução](#instalação-e-execução)
- [Conceitos fundamentais](#conceitos-fundamentais)
- [Estrutura de diretórios](#estrutura-de-diretórios)
- [Documentação por arquivo](#documentação-por-arquivo)
  - [Pontos de entrada](#pontos-de-entrada)
  - [src/inputs — o que alimenta a simulação](#srcinputs--o-que-alimenta-a-simulação)
  - [src/simulation — simular e calibrar](#srcsimulation--simular-e-calibrar)
  - [src/stochastic — o ensemble Monte Carlo](#srcstochastic--o-ensemble-monte-carlo)
  - [src/outputs — o que sai](#srcoutputs--o-que-sai)
  - [src/workflow.py — a orquestração](#srcworkflowpy--a-orquestração)
  - [src/ui — a interface gráfica](#srcui--a-interface-gráfica)
  - [conf — configuração Hydra](#conf--configuração-hydra)
  - [scripts — utilitários avulsos](#scripts--utilitários-avulsos)
- [Fluxos de execução](#fluxos-de-execução)
- [Formatos de saída](#formatos-de-saída)
- [Notas operacionais](#notas-operacionais)

---

## Instalação e execução

```bash
conda env create --file env/conda.yaml   # ou mamba
conda activate deriva
pip install -r env/requirements.txt
```

Interface gráfica:

```bash
python app.py                    # janela nativa
FLET_VIEW=web python app.py      # navegador, em http://localhost:8550
```

O modo web é o caminho quando o sistema não tem a biblioteca `libmpv`, que o
cliente desktop do Flet exige.

Linha de comando (apenas o modo de validação):

```bash
python cli.py --help
```

---

## Conceitos fundamentais

Alguns termos aparecem o tempo todo no código e na interface. Entendê-los
antecipadamente torna o resto da documentação muito mais legível.

**Manchas.** São os polígonos de óleo observados, lidos de um shapefile
compactado em `.zip`. Cada polígono tem uma data e hora associadas. É o dado
de referência contra o qual as simulações são comparadas.

**Forcing.** O conjunto de dados ambientais que empurram as partículas:
corrente marinha, vento e (opcionalmente) salinidade e temperatura. São
arquivos NetCDF, e a qualidade da simulação depende inteiramente deles.

**WDF (Wind Drift Factor).** Fração da velocidade do vento transferida à
mancha. Valores típicos ficam entre 1% e 4%. É o parâmetro mais sensível do
modelo.

**CDF (Current Drift Factor).** Multiplicador aplicado à velocidade da
corrente. Serve para corrigir viés sistemático do modelo hidrodinâmico.

**Offset ambiental.** Deslocamento em horas aplicado ao eixo de tempo dos
dados de forcing. Existe porque os dados do Copernicus vêm em UTC e as
observações de mancha costumam estar em horário local. Também é usado como
parâmetro de busca: às vezes um deslocamento de algumas horas melhora
sensivelmente o ajuste, indicando defasagem do modelo de correntes.

**Lag temporal (tau).** Exclusivo do modo estocástico. É uma perturbação
aleatória no tempo do forcing, representando a incerteza da previsão
meteoceanográfica. Um membro do ensemble com lag de +30 minutos usa o campo de
corrente que o modelo previu para 30 minutos à frente.

**Skill score.** Métrica que resume, num único número, o quanto a trajetória
simulada se parece com a observada. Quanto maior, melhor.

---

## Estrutura de diretórios

```
app.py            entrada da interface gráfica
cli.py            entrada da linha de comando
conf/             configuração declarativa (Hydra)
data/             dados brutos, resultados e cache
runs/             cópias dos arquivos de entrada de cada execução
scripts/          utilitários avulsos (plots, análises, conversões)
src/
  workflow.py     conduz cada execução do início ao fim
  inputs/         o que alimenta a simulação
    config.py     parâmetros de execução e leitura dos YAMLs
    spills.py     leitura e normalização das manchas observadas
    forcing.py    download, adaptação, validação e leitura do forcing
  simulation/     simular e calibrar
    drift.py      execução de uma simulação no OpenDrift
    optimize.py   busca dos melhores WDF/CDF e a métrica de qualidade
  stochastic/     o ensemble Monte Carlo
    sampling.py   sorteio dos parâmetros de cada membro
    runner.py     execução paralela dos membros e agregação
    raster.py     grade fixa, rasterização e mapas de probabilidade
  outputs/        o que sai
    artifacts.py  escrita de diretórios, JSON, CSV e log
    plots.py      GIF de comparação e frames por instante
  ui/             interface gráfica em Flet
    app.py        estado, handlers e montagem da janela
    views.py      layout das telas
    helpers.py    utilitários de apoio da interface
```

A organização é **por assunto**, não por camada técnica. O nome do arquivo
responde à pergunta "onde eu mexo para alterar X?". As dependências fluem numa
direção só, sem ciclos: `workflow` está no topo e consome todo o resto;
`config`, `forcing` e `spills` são folhas que não dependem de mais nada interno.

---

# Documentação por arquivo

## Pontos de entrada

### `app.py`

Arquivo mínimo, com uma única responsabilidade: iniciar a interface gráfica.
Ele lê uma variável de ambiente para decidir entre abrir uma janela nativa do
sistema ou servir a aplicação num navegador, e delega toda a construção da tela
para o módulo da interface. A separação existe para que a lógica da interface
não fique presa ao modo de exibição.

### `cli.py`

Expõe o modo de validação como comando de terminal. Declara todas as opções
aceitas — caminhos de arquivos, faixas de busca de parâmetros, ambiente,
recortes geográficos, flags de pular etapas — converte o que o usuário digitou
num pedido de execução estruturado e entrega ao orquestrador.

Uma função:

- **simulate_validation** — recebe as opções da linha de comando, converte a
  lista de offsets ambientais de texto para números, monta o pedido de validação
  e dispara a execução.

Só o modo de validação está exposto por linha de comando. Os modos
determinístico e estocástico existem apenas na interface gráfica.

---

## `src/inputs` — o que alimenta a simulação

### `src/inputs/config.py`

Concentra tudo que **descreve** uma execução, sem executar nada. É onde estão a
leitura dos arquivos de configuração e as estruturas de dados que carregam os
parâmetros entre as camadas do sistema.

**EnvironmentRepository** — única classe do arquivo com comportamento.

- **compose_config** — localiza a pasta de configuração do projeto, carrega os
  arquivos YAML através do Hydra, aplica os ajustes pedidos (ambiente
  escolhido, recorte geográfico, nome da execução, caminho do shapefile) e
  devolve o objeto de configuração final que o resto do sistema consulta.

**Estruturas de pedido e resultado.** As demais são estruturas de dados
imutáveis, sem lógica. Existem para que uma execução seja descrita por um único
objeto em vez de vinte argumentos soltos.

- **ConfigRequest** — descreve qual configuração carregar: nome do arquivo
  base, ambiente, nome da execução e recortes de longitude e latitude. Tem um
  método, **to_overrides**, que traduz esses campos para a sintaxe de
  sobrescrita que o Hydra entende.
- **ObservedSpillRequest** — descreve como ler as manchas observadas: caminho
  do arquivo, deslocamento de horas a aplicar nas datas, a partir de qual
  instante observado começar e a folga a deixar em volta ao desenhar figuras.
- **ObservedSpillContext** — o resultado dessa leitura: as manchas já
  normalizadas e os limites geográficos para enquadrar as figuras.
- **ValidationRunRequest** — descreve uma execução completa do modo de casos.
  Reúne as faixas de busca de WDF e CDF, os fatores fixos quando não há busca,
  o ambiente, os arquivos de forcing, os offsets a testar, os processos físicos
  a ativar, os tipos de óleo e as flags de pular animação, simulação ou
  gráficos.
- **DeterministicRunRequest** — descreve uma simulação a partir de um ponto:
  coordenadas, instante inicial, duração total, duração do vazamento, raio
  inicial da mancha, número de partículas, passos de tempo, fatores de deriva,
  tipo de óleo e quais arquivos de forcing usar.
- **DeterministicRunResult** — o que a execução determinística devolve: nome,
  diretório e arquivo de saída, o ponto simulado, a janela temporal coberta e a
  lista de arquivos gerados.
- **StochasticValidationRunRequest** — descreve uma execução do ensemble.
  Carrega dentro de si a configuração estocástica completa, mais os mesmos
  campos de ambiente e forcing dos outros modos.
- **ValidationRunResult** — o que a validação devolve: parâmetros vencedores,
  caminhos da simulação e das figuras, e a lista de artefatos.

**Configuração do ensemble.** Um segundo grupo descreve especificamente a
execução estocástica.

- **StochasticParameterConfig** — como sortear um parâmetro numérico (WDF ou
  CDF): se a variação está ligada, média, desvio padrão, limites mínimo e
  máximo, e o valor fixo a usar quando a variação está desligada.
- **TemporalLagConfig** — o mesmo para o lag temporal, com dois campos extras:
  a unidade em que o usuário digitou os valores e a granularidade de
  arredondamento do resultado.
- **StochasticGridConfig** — a grade fixa sobre a qual os mapas de
  probabilidade são construídos: limites geográficos, resolução em graus,
  margem adicional e sistema de coordenadas.
- **StochasticRunConfig** — a execução inteira: nome, quantos membros, semente
  aleatória, as três configurações de parâmetro acima, a grade, o diretório de
  saída e quantos processos paralelos usar.
- **SampledParameterSet** — o sorteio de um membro específico. Guarda os
  valores sorteados, a semente usada, e é atualizado durante a execução com o
  status (pendente, sucesso ou falha), a mensagem de erro e o caminho da saída.
  É esta estrutura que vira linha no CSV de acompanhamento.
- **StochasticRunResult** — o resumo final: contagem de sucessos e falhas e os
  caminhos de todos os mapas gerados.

### `src/inputs/forcing.py`

Cobre todo o caminho dos dados ambientais, do download até o momento em que o
OpenDrift consegue lê-los. É o arquivo com mais responsabilidades distintas,
unificadas por tratarem do mesmo assunto.

**Normalização de entradas.**

- **normalize_forcing_source** — recebe o nome da fonte de dados, padroniza a
  grafia e rejeita fontes não suportadas. Aceita Copernicus, NOAA e REMO.
- **normalize_path_list** — recebe um caminho isolado e uma lista de caminhos,
  combina os dois numa lista única, descarta vazios e remove duplicatas
  preservando a ordem.
- **normalize_environmental_offset_values** — valida uma lista de offsets em
  horas: rejeita valores não finitos, rejeita valores acima do limite máximo
  permitido e elimina repetições.

**Validação de cobertura.** Impede que uma simulação comece para depois falhar
no meio por falta de dados.

- **_dataset_coord_range** — extrai de um arquivo a faixa mínima e máxima de
  uma coordenada, testando os nomes alternativos que os diferentes produtores
  de dados usam.
- **_has_overlap** — informa se dois intervalos numéricos se sobrepõem.
- **validate_environment_coverage** — compara a área e a janela temporal das
  manchas observadas contra a cobertura de cada grupo de arquivos de forcing.
  Verifica se os arquivos existem, extrai deles a faixa de coordenadas e de
  tempo, aplica o offset em questão, e interrompe com mensagem explicativa se
  as manchas caírem fora. Corrente e salinidade são obrigatórias; vento só é
  checado se tiver sido fornecido.
- **valid_environmental_offsets** — recebe vários offsets candidatos, testa a
  cobertura de cada um, descarta os inválidos avisando no log, e devolve apenas
  os que têm dados suficientes. Se nenhum sobreviver, interrompe informando o
  primeiro erro encontrado.

**Montagem dos leitores.**

- **resolve_forcing_paths** — decide quais arquivos de corrente, vento e
  salinidade usar. A lista com vários arquivos tem prioridade sobre o caminho
  isolado; se nenhum for informado, cai no caminho declarado no YAML do
  ambiente. Vento é opcional. Todos os arquivos escolhidos passam pelo
  adaptador antes de serem devolvidos.
- **build_reader** — abre um arquivo NetCDF como leitor do OpenDrift e ajusta
  seu eixo de tempo. Aplica o offset ambiental e, quando há lag temporal,
  desloca o eixo no sentido contrário ao lag, de forma que o campo previsto
  para um instante futuro fique disponível no instante atual da simulação.
- **build_readers** — monta a lista completa de leitores na ordem de prioridade
  que o OpenDrift respeita. Correntes vêm antes de ventos e, dentro de cada
  grupo, os arquivos mais recentes primeiro — assim, quando duas rodadas de
  previsão se sobrepõem no tempo, a mais nova prevalece. A salinidade, se
  houver, entra por último.

**ForcingDatasetAdapter** — conserta arquivos que não estão no formato que os
leitores esperam. Hoje só a fonte REMO precisa de tratamento.

- **prepare_path** — decide se um arquivo precisa de adaptação. Arquivos que
  não são REMO passam direto. Para os que precisam, consulta um cache em
  memória antes de refazer o trabalho.
- **_prepare_remo_path** — gera uma chave de cache a partir do caminho, data de
  modificação e tamanho do arquivo original. Se a versão adaptada já existe em
  disco, reaproveita; senão, adapta, grava num arquivo temporário e só então o
  move para o destino final, evitando deixar arquivos pela metade.
- **_adapt_remo_dataset** — aplica as correções: renomeia a dimensão de tempo
  para o nome padrão, remove a dimensão de profundidade quando ela tem um único
  nível, e regulariza a grade espacial das correntes.
- **_is_uniform_1d** — verifica se um eixo de coordenadas tem espaçamento
  constante, com tolerância proporcional ao próprio espaçamento.
- **_regularize_lat_lon_grid** — quando um eixo não é uniforme, constrói um
  eixo igualmente espaçado entre os mesmos extremos e interpola os dados para
  ele. Necessário porque os leitores assumem grades regulares.
- **_find_coord_name** — procura o nome da coordenada de latitude ou longitude
  entre as variantes comuns.

**CopernicusGateway** — baixa dados do serviço Copernicus Marine.

- **_resolve_credentials** — procura usuário e senha em ordem: os passados
  diretamente, variáveis de ambiente, e por fim arquivos de login em locais
  conhecidos do projeto e da pasta pessoal. Se não encontrar, informa todos os
  caminhos que tentou.
- **_dataset_specs** — declara quais conjuntos de dados baixar. Atualmente um
  só: salinidade e temperatura potencial.
- **download_environment_data** — autentica no serviço, e para cada conjunto de
  dados declarado verifica se o arquivo já existe (pulando o download, a menos
  que forçado), recorta pela área e período pedidos, e grava. Se o serviço
  reclamar de sobreposição de longitudes negativas, tenta de novo com as
  longitudes convertidas para a faixa de 0 a 360. Relata o andamento por
  callback e devolve o desfecho de cada conjunto.

### `src/inputs/spills.py`

Lê e normaliza as manchas observadas.

- **generate_random_points_in_polygon** — sorteia pontos dentro de um polígono
  pelo método da rejeição: gera candidatos dentro do retângulo que envolve o
  polígono e mantém apenas os que caem dentro dele. Usado para distribuir as
  partículas iniciais dentro da primeira mancha observada.

**SpillRepository**

- **centroid_lat_lon_from_group** — calcula o centro geométrico de um conjunto
  de polígonos. Projeta as geometrias para um sistema métrico apropriado à
  região antes de calcular, porque centroide em coordenadas geográficas é
  distorcido, e converte o resultado de volta para latitude e longitude.
- **ensure_datetime_column** — garante que exista uma coluna de data e hora
  utilizável. Os shapefiles chegam em dois formatos diferentes: um com data e
  hora em campos separados, outro com um campo único. A função detecta qual é o
  caso, monta a coluna unificada e aplica o deslocamento de horas pedido.
- **build_observed_trajectory** — condensa as manchas numa trajetória. Agrupa
  os polígonos por instante, reduz cada grupo ao seu centroide, e devolve a
  sequência ordenada de pontos. É essa trajetória simplificada que a otimização
  compara contra a simulação.

---

## `src/simulation` — simular e calibrar

### `src/simulation/drift.py`

Executa simulações no OpenDrift. Duas formas de semear partículas, que diferem
em de onde vem a janela temporal.

**SimulationService**

- **simulate_drift** — o modo baseado em observação. Determina o instante
  inicial e final a partir das próprias manchas, distribui as partículas dentro
  do polígono da primeira mancha observada, monta os leitores de forcing,
  configura o esquema de advecção e os fatores de deriva, semeia com o tipo de
  óleo escolhido e roda até o último instante observado. Grava o resultado em
  NetCDF e, se pedido, gera uma animação com o campo de correntes ao fundo.
  Aceita um lag temporal, que é o que permite ao ensemble estocástico perturbar
  o forcing de cada membro.

- **simulate_point_drift** — o modo por ponto. Não depende de observação
  nenhuma: recebe coordenadas, instante inicial e duração como parâmetros
  diretos. Monta os leitores, e quando não há arquivo de vento configura o
  vento como zero para que a simulação prossiga com deriva puramente por
  corrente em vez de abortar. A salinidade é opcional, o que permite simular em
  datas para as quais esse dado não foi baixado. Um vazamento de duração zero
  libera todas as partículas de uma vez; com duração, elas são distribuídas ao
  longo do intervalo. Devolve o caminho do arquivo gerado.

### `src/simulation/optimize.py`

Encontra os fatores de deriva que melhor reproduzem o comportamento observado.

**MetricsService**

- **haversine_m** — calcula a distância em metros entre pares de coordenadas
  sobre a superfície da Terra. Opera sobre vetores inteiros de uma vez.
- **liu_weissberg_skillscore** — mede a aderência entre trajetória observada e
  simulada. Casa os dois conjuntos pelos instantes em comum, soma as separações
  entre pontos correspondentes e divide pelo comprimento total do caminho
  percorrido; o resultado é subtraído de um, de modo que valores maiores
  indicam melhor ajuste. Quando há apenas dois instantes em comum, a fórmula
  fica instável, e a função usa a distância final convertida numa pontuação
  limitada entre zero e um.

**OptimizationService**

- **fast_grid_search_wind_drift_factor** — testa muitos valores de WDF numa
  única simulação. Em vez de rodar o modelo uma vez por valor, semeia todas as
  partículas juntas atribuindo a cada uma um WDF diferente, roda uma vez só, e
  depois separa as partículas por valor para calcular a pontuação de cada um.
  Reduz dezenas de simulações a uma. Também eleva o passo de tempo a um mínimo,
  porque uma busca com passo muito fino ficaria lenta demais sem ganho de
  precisão relevante.
- **fast_grid_search_wdf_cdf** — estende a busca para o CDF. Como o fator de
  corrente é uma configuração global do modelo e não uma propriedade de cada
  partícula, ele não pode ser vetorizado: a função roda a busca de WDF acima uma
  vez para cada valor de CDF e junta os resultados numa tabela única, apontando
  a melhor combinação.

---

## `src/stochastic` — o ensemble Monte Carlo

A ideia central deste módulo: a incerteza não entra dentro do modelo físico.
Cada membro do ensemble é uma simulação determinística comum, com seus próprios
valores de WDF, CDF e lag temporal. É a distribuição dos resultados que vira o
mapa de probabilidade.

### `src/stochastic/sampling.py`

**ParameterSampler**

- **validate_run_config** — confere a configuração inteira antes de gastar
  tempo de processamento: número de membros positivo, semente inteira,
  quantidade válida de processos, e delega a validação de cada parâmetro e da
  grade.
- **validate_parameter_config** — verifica um parâmetro numérico: distribuição
  suportada, desvio padrão não negativo, mínimo menor que máximo, média dentro
  dos limites quando a variação está ligada, e presença de valor padrão quando
  está desligada.
- **validate_temporal_lag_config** — o mesmo para o lag temporal, checando
  adicionalmente se a unidade de entrada e a granularidade de arredondamento
  são reconhecidas.
- **sample** — sorteia os parâmetros de todos os membros. Cada membro recebe
  uma semente derivada da semente base somada ao seu índice, o que torna o
  ensemble inteiro reproduzível: a mesma semente na interface gera exatamente
  os mesmos valores. Dentro de cada membro, os três parâmetros saem do mesmo
  gerador, sempre na mesma ordem.
- **to_dataframe** — converte a lista de sorteios em tabela, para gravação.
- **_sample_scalar** — devolve o valor fixo quando a variação está desligada,
  ou sorteia dentro dos limites quando ligada.
- **_sample_temporal_lag_original** — o mesmo para o lag, convertendo o valor
  padrão de segundos para a unidade em que o usuário trabalha.
- **_temporal_lag_to_seconds** — converte o valor sorteado para segundos e
  arredonda conforme a granularidade escolhida.
- **_sample_normal_bounded** — sorteia de uma distribuição normal truncada pelo
  método da rejeição: repete o sorteio até cair dentro dos limites. Preserva a
  forma da distribuição, ao contrário de simplesmente cortar os valores fora
  dos limites, o que acumularia massa artificial nas bordas. Após muitas
  tentativas sem sucesso — situação que só ocorre com limites muito estreitos
  em relação ao desvio — recorre ao corte como último recurso.

### `src/stochastic/runner.py`

Executa os membros e agrega os resultados.

**Funções de módulo**

- **_build_member_config** — copia a configuração base redirecionando a saída
  para o diretório próprio do membro. A cópia é profunda porque os membros
  rodam em processos separados e cada um reescreve o nome e o caminho de saída.
- **_normalize_simulation_output_timeline** — reindexa o arquivo de saída de um
  membro para a linha de tempo comum do ensemble. Necessário porque o lag
  temporal pode fazer membros terminarem com instantes ligeiramente diferentes;
  sem isso, a agregação por hora compararia instantes desalinhados. Instantes
  ausentes viram vazios em vez de deslocar os demais. A escrita passa por
  arquivo temporário para não corromper a saída em caso de falha.
- **_run_stochastic_member_worker** — roda um membro completo e devolve o
  sorteio com o status preenchido. Precisa ser função de módulo, e não método,
  para poder ser serializada e enviada a outro processo. Captura qualquer falha
  e a converte em status de erro, para que um membro problemático não derrube o
  ensemble inteiro.

**StochasticSimulationService**

- **run** — o método principal, que conduz todo o ensemble. Prepara os
  diretórios, valida a configuração, constrói a grade fixa, sorteia os
  parâmetros e grava a configuração e a tabela de sorteios antes de começar.
  Calcula a linha de tempo esperada. Executa os membros, sequencialmente quando
  há um único processo ou em paralelo através de um pool de processos. A cada
  membro concluído, reescreve a tabela de sorteios com o status atualizado e
  notifica o progresso — é isso que permite acompanhar a execução em tempo real
  e inspecionar resultados parciais. Terminada a execução, converte cada saída
  bem-sucedida em máscaras binárias, soma-as e divide pelo número de membros
  válidos para obter as probabilidades. Grava os mapas, monta o resumo final e
  devolve o resultado consolidado.
- **_ordered_samples** — devolve os sorteios ordenados por índice, já que a
  execução paralela os conclui fora de ordem.
- **_expected_output_times** — constrói a linha de tempo de referência do
  ensemble a partir do primeiro e do último instante observado, avançando pelo
  passo de saída configurado.
- **_pad_or_trim_hourly_binaries** — ajusta um membro que produziu menos ou
  mais instantes que os demais, completando com mapas vazios ou descartando o
  excesso, de forma que todos os membros contribuam para os mesmos índices.
- **_save_hourly_probability_rasters** — grava um arquivo de probabilidade por
  instante da linha de tempo, nomeando cada um pelo índice e pelo horário.
- **_safe_filename_label** — transforma um horário em texto seguro para nome de
  arquivo, trocando caracteres especiais por sublinhados.

### `src/stochastic/raster.py`

Converte posições de partículas em mapas.

**FixedGrid** — descreve a grade sobre a qual tudo é rasterizado. A grade é
fixa justamente para que membros diferentes sejam somáveis célula a célula,
independentemente de onde cada um espalhou suas partículas.

- **width** e **height** — quantas colunas e linhas a grade tem, calculadas a
  partir dos limites e da resolução.
- **transform** — a relação entre índices de célula e coordenadas geográficas,
  no formato que as bibliotecas de raster esperam.
- **validate** — confere que os limites fazem sentido, que a resolução é
  positiva e que as dimensões resultantes não são degeneradas.

**DriftRasterConverter**

- **fixed_grid_from_config** — constrói a grade a partir da configuração,
  aplicando a margem adicional aos quatro lados e validando o resultado.
- **convert_simulation_to_binary_array** — reduz uma simulação a um mapa de
  presença: célula com pelo menos uma partícula vira um, o resto vira zero.
  Sem instante especificado, considera todas as posições de todos os tempos —
  produzindo a área varrida pela mancha ao longo de toda a simulação.
- **convert_simulation_to_count_array** — o mesmo, mas preservando quantas
  partículas caíram em cada célula.
- **convert_simulation_to_binary_arrays_for_times** — produz um mapa por
  instante da linha de tempo esperada. Casa cada instante esperado com o
  correspondente no arquivo e, quando um instante não existe na simulação,
  devolve um mapa vazio no lugar, mantendo o alinhamento entre membros.
- **_format_time_value** — formata um instante como texto legível.
- **rasterize_points** — o núcleo da conversão. Descarta posições inválidas ou
  fora dos limites, converte cada coordenada em índices de linha e coluna, e
  acumula as contagens. Opera sobre vetores inteiros de uma vez.
- **save_geotiff** — grava um mapa como GeoTIFF comprimido, com o sistema de
  coordenadas e o posicionamento geográfico corretos, de modo que o arquivo
  abra alinhado em qualquer software de SIG.

**EnsembleAggregationResult** — carrega o número de membros válidos, o mapa de
contagens, o mapa de probabilidades e os caminhos onde foram gravados.

**EnsembleAggregator**

- **save_maps** — grava o par de mapas de uma agregação: as contagens como
  números inteiros e as probabilidades como decimais, e devolve o resultado
  acrescido dos caminhos.

---

## `src/outputs` — o que sai

### `src/outputs/artifacts.py`

Escrita dos arquivos que registram uma execução.

**WorkspaceRepository** — artefatos das execuções de validação e determinística.

- **simulation_output_dir** — monta e cria o diretório de saída da execução.
- **write_json** — grava um dicionário como JSON legível, criando os
  diretórios necessários.
- **write_csv** — grava uma tabela como CSV, sem a coluna de índice.

**StochasticOutputService** — artefatos do ensemble, que tem estrutura própria.

- **prepare_run_directory** — monta a árvore de diretórios da execução —
  raiz, membros individuais e agregados — e devolve o mapa de todos os caminhos
  que serão usados.
- **write_config** — grava a configuração completa da execução, para que ela
  possa ser reproduzida depois.
- **write_samples** — grava a tabela de sorteios com o status de cada membro.
  Chamada repetidamente durante a execução, refletindo o progresso.
- **write_summary** — grava o resumo final com contagens, caminhos e tempo
  decorrido.
- **append_log** — acrescenta uma linha ao arquivo de log da execução.
- **write_json** — grava qualquer conteúdo como JSON, convertendo antes o que
  não é serializável.
- **_to_jsonable** — percorre estruturas aninhadas convertendo objetos de
  configuração e caminhos de arquivo em tipos que o JSON aceita.

### `src/outputs/plots.py`

Geração de figuras.

- **_parse_extent** — interpreta o enquadramento geográfico escrito como texto
  com quatro números separados por vírgula.
- **_nearest_time** — encontra, numa lista de instantes, o mais próximo de um
  instante alvo.
- **_pick_times_from_list** — escolhe uma quantidade desejada de instantes
  distribuídos uniformemente ao longo de uma lista.
- **_prepare_real** — carrega as manchas observadas para desenho: resolve a
  coluna de data e hora nos dois formatos possíveis, aplica o deslocamento de
  horas, ordena e recorta a partir do instante inicial escolhido.
- **_sim_points_at_time** — extrai as posições das partículas no instante mais
  próximo de um alvo, descartando as inválidas.
- **_real_polygon_at_time** — devolve a mancha observada mais próxima de um
  instante, unindo os polígonos daquele momento num só.
- **generate_comparison_gif** — monta a animação que compara simulação e
  observação. Para cada instante, desenha as manchas observadas ao fundo e a
  mancha simulada por cima, esta como envoltória convexa das partículas ou como
  nuvem de pontos. Limita a quantidade de pontos desenhados para manter o
  arquivo leve, e junta os quadros num GIF.
- **render_comparison_frames** — gera uma imagem separada por instante
  observado, sobrepondo as partículas simuladas e a mancha real. Ao contrário
  do GIF, produz arquivos individuais que a interface exibe num controle
  deslizante. Verifica pedidos de cancelamento entre quadros.

---

## `src/workflow.py` — a orquestração

É o topo do sistema: conhece todos os outros módulos e os coordena. Nenhum
outro módulo depende dele.

**_ProgressPrinter** — acompanha o andamento de operações longas.

- **tick** — registra mais uma unidade concluída, notifica quem estiver
  ouvindo, imprime o progresso periodicamente e interrompe a execução se um
  cancelamento tiver sido pedido.

**_ValidationContext** — reúne tudo que uma execução de validação precisa
carregar entre suas etapas: configuração, offsets, arquivos de forcing, manchas
carregadas, trajetória observada e os parâmetros escolhidos. Existe para que as
etapas não precisem repassar dezenas de argumentos entre si. É a única
estrutura mutável do sistema, porque a etapa de otimização escreve nela os
parâmetros vencedores.

**SimulationController**

Utilitários internos:

- **_parse_bool_string** — converte as várias grafias de verdadeiro e falso
  aceitas na interface e na linha de comando num booleano, rejeitando o resto.
- **_load_oil_types** — carrega os tipos de óleo pedidos, de um arquivo ou de
  uma lista separada por vírgulas, e confere cada um contra o catálogo do
  OpenOil ignorando diferenças de maiúsculas e espaçamento. Rejeita nomes
  desconhecidos listando quais falharam.
- **_build_sim_filename** — monta o nome do arquivo de saída embutindo os
  parâmetros usados, de modo que execuções diferentes não se sobrescrevam e o
  nome já indique o que foi rodado.
- **_check_cancelled** — interrompe a execução se um cancelamento tiver sido
  pedido. Chamado nos pontos seguros entre etapas.

Preparação:

- **load_config** — carrega a configuração aplicando os ajustes do pedido.
- **download_environment_data** — carrega a configuração do ambiente e aciona o
  download dos dados de salinidade e temperatura, repassando credenciais,
  recorte geográfico e callback de log.
- **load_observed_spills** — lê o shapefile, converte para coordenadas
  geográficas, normaliza a coluna de data e hora, descarta os instantes
  anteriores ao início escolhido e calcula os limites com folga para as figuras.
- **_build_validation_context** — a preparação completa de uma validação.
  Carrega a configuração, normaliza a fonte de forcing, resolve os offsets,
  determina quais arquivos usar, carrega as manchas, valida a cobertura
  ambiental (testando vários offsets quando é o caso e mantendo apenas os
  viáveis), resolve o tipo de óleo e monta o contexto que as etapas seguintes
  consomem.

Etapas de execução:

- **_run_fast_optimization_phase** — a busca de parâmetros. Valida as faixas
  pedidas, monta as listas de valores a testar, roda a busca para cada offset
  ambiental candidato, junta os resultados numa tabela única, identifica a
  melhor combinação e a escreve de volta no contexto. Grava a tabela completa e
  um resumo com os parâmetros vencedores. Devolve um indicador de sucesso —
  quando nenhuma combinação produz pontuação válida, a execução é abortada.
- **_run_simulation_phase** — a simulação propriamente dita. Monta o nome do
  arquivo, grava o registro dos parâmetros usados e roda a simulação com os
  valores do contexto. Pode ser pulada quando se quer apenas regerar figuras de
  uma simulação anterior.
- **_run_visualization_phase** — gera o GIF de comparação e os quadros
  individuais. Falhas na geração do GIF são registradas mas não derrubam a
  execução, já que os dados numéricos já estão salvos.

Execuções completas:

- **run_validation** — o modo de casos de ponta a ponta: monta o contexto,
  opcionalmente otimiza, simula, gera figuras, lista os artefatos produzidos e
  devolve o resultado consolidado.
- **run_deterministic** — o modo por ponto. Não passa pelo contexto de
  validação, porque sem manchas observadas não há o que validar contra. Define
  a área de forcing como uma caixa em volta do ponto de vazamento, carrega a
  configuração, grava o registro dos parâmetros e roda a simulação por ponto.
- **run_stochastic_validation** — o modo ensemble. Converte o pedido
  estocástico num pedido de validação equivalente para reaproveitar toda a
  preparação de contexto — carga de manchas, resolução de forcing, validação de
  cobertura — e então delega a execução ao serviço do ensemble, repassando os
  callbacks de progresso, cancelamento e log.

---

## `src/ui` — a interface gráfica

### `src/ui/app.py`

O arquivo mais extenso do projeto. Concentra o estado da aplicação, todos os
controles de tela, os handlers de interação e a comunicação com as execuções em
segundo plano.

**Arquitetura interna.** A interface roda numa thread; as simulações rodam em
outras. Como controles gráficos não podem ser tocados de fora da thread da
interface, a comunicação acontece por uma fila de eventos: os workers publicam
eventos, e um consumidor os aplica na tela. A saída de texto das simulações é
redirecionada para essa mesma fila, o que faz o log do OpenDrift aparecer ao
vivo na aba de execução.

- **_enum_value** — busca um valor de enumeração da biblioteca gráfica com
  alternativa, para tolerar diferenças entre versões do Flet.
- **_expand_bounds** — expande os limites das manchas observadas com uma folga
  fixa e garante uma extensão mínima. Esses limites definem a área de download
  do forcing e a grade padrão do ensemble.
- **_parse_optional_float** — converte texto em número, aceitando campo vazio
  como ausência de valor.
- **main** — constrói a aplicação inteira. Todas as funções abaixo vivem dentro
  dela, compartilhando o mesmo estado por fechamento léxico.

*Estado e mensagens*

- **show_message** — exibe uma notificação flutuante, em cor neutra ou de erro,
  tolerando as duas formas de fazê-lo conforme a versão do Flet.
- **_sync_credentials** — devolve um handler que copia o valor de um campo de
  credencial para o equivalente da outra aba. É o que permite digitar o login
  numa aba e vê-lo preenchido na outra, já que a biblioteca não aceita o mesmo
  controle em duas telas.

*Atualização de tela*

- **append_log** — acrescenta uma linha ao log da aba de execução, descartando
  as mais antigas quando a lista fica longa demais.
- **set_download_status** — reflete a mesma mensagem de status de download nas
  duas abas que oferecem o botão.
- **reset_execution_panel** — zera título, subtítulo, barra de progresso e log
  antes de uma nova execução.
- **refresh_artifacts** — varre o diretório de saída e monta a lista de
  arquivos gerados, cada um com tamanho e botão de abrir.
- **refresh_results** — lê as métricas da execução e preenche os indicadores de
  pontuação e parâmetros vencedores; carrega os quadros de comparação e
  configura o controle deslizante que os navega.
- **on_frame_slider_change** — troca a imagem exibida conforme o usuário move o
  controle deslizante.
- **recover_output_from_run_id** — dada uma identificação de execução, procura
  o diretório correspondente e a simulação mais recente dentro dele. Serve para
  recuperar resultados quando um cancelamento chega depois de a execução já ter
  terminado de gravar.

*Fila de eventos*

- **handle_queue_event** — recebe cada evento publicado pelos workers e o
  aplica na tela: linhas de log, progresso de otimização ou de ensemble, início
  e fim de cada lote, conclusão de cada tipo de execução, cancelamento e erro.
  É o único ponto onde os controles são tocados, o que mantém a interface
  segura em relação às threads.
- **event_consumer_loop** — laço que retira eventos da fila indefinidamente e
  os entrega ao tratador acima.

*Montagem de pedidos*

- **build_offset_values** — converte a faixa de offset ambiental digitada numa
  lista simétrica de valores inteiros a testar, rejeitando faixas não inteiras
  ou acima do limite.
- **offset_suffix** — gera o sufixo de nome de execução que identifica o offset
  usado, distinguindo positivos de negativos.
- **build_request** — monta o pedido de validação a partir dos campos da aba de
  casos, escolhendo entre faixas de busca ou fatores fixos conforme o modo.
- **build_deterministic_request** — monta o pedido determinístico. Combina data
  e hora num instante único, valida faixas de latitude e longitude, e recusa
  duração, quantidade de partículas ou passo de tempo não positivos, com
  mensagem específica para cada campo.
- **build_stochastic_request** — monta o pedido do ensemble, incluindo as três
  configurações de sorteio e a grade, que assume os limites das manchas quando
  os campos são deixados em branco.

*Execução em segundo plano*

- **worker_run_validation** — roda a lista de validações, publicando início e
  fim de cada uma, e interrompe se um cancelamento chegar entre elas.
- **worker_run_deterministic** — roda a simulação por ponto.
- **worker_run_stochastic** — roda o ensemble, repassando o progresso de cada
  membro concluído.
- **worker_download_environment** — roda o download dos dados ambientais,
  publicando as mensagens do serviço à medida que chegam.
- **on_progress** e **on_log** — pequenos repassadores que convertem chamadas
  dos serviços em eventos da fila.

*Handlers dos botões*

- **start_execution** — inicia o modo de casos. Bloqueia se já houver execução
  ou download em andamento, valida o shapefile e os arquivos de forcing, monta
  um pedido por offset quando a otimização está desligada, copia o shapefile
  para o diretório da execução e dispara a thread.
- **start_deterministic_execution** — inicia o modo determinístico, validando
  os campos e registrando no log o ponto, a janela e as características do
  vazamento.
- **start_stochastic_execution** — inicia o ensemble, registrando no log os
  limites da grade e a resolução.
- **start_environment_download** — inicia o download. Recebe quais controles
  consultar, de modo que o botão funcione nas duas abas usando o ambiente e as
  credenciais da aba que o acionou. Quando há shapefile selecionado, usa seus
  limites como área e força a sobrescrita do arquivo existente.
- **cancel_execution** — sinaliza o pedido de parada, que as execuções checam
  nos pontos seguros.

*Seleção de arquivos*

- **choose_zip**, **choose_current_dataset**, **choose_wind_dataset** — abrem o
  seletor filtrando pela extensão apropriada e registram qual tratador deve
  receber o resultado.
- **_on_file_picker_result** — encaminha o resultado ao tratador registrado.
- **_result_choose_zip** — guarda o shapefile escolhido e atualiza seu rótulo
  em todas as abas.
- **_result_choose_dataset** — guarda a lista de arquivos escolhidos e monta um
  rótulo resumido, propagando-o para todas as abas que o exibem.
- **_warn_no_path** — avisa quando o navegador não expõe o caminho local do
  arquivo, situação que ocorre no modo web e exige rodar em modo desktop.

*Navegação*

- **set_screen** — troca a tela ativa e destaca o botão correspondente no menu
  lateral.

### `src/ui/views.py`

Contém apenas layout — nenhuma lógica de negócio, nenhum acesso a dados. Cada
tela recebe uma estrutura de vínculos com os controles e callbacks já prontos,
e apenas os organiza visualmente. Essa separação permite alterar a aparência
sem tocar no comportamento.

- **_icon_value** e **_enum_value** — resolvem ícones e enumerações com
  alternativa, para tolerar diferenças entre versões do Flet.

**Estruturas de vínculo** — uma por tela (**SetupViewBindings**,
**StochasticViewBindings**, **DeterministicViewBindings**,
**ExecutionViewBindings**, **ResultsViewBindings**, **ArtifactsViewBindings**).
Cada uma declara exatamente quais controles e callbacks aquela tela precisa. Se
um campo for esquecido na montagem, o erro aparece imediatamente, e não como
tela quebrada em tempo de uso.

**Construtores de tela**

- **build_setup_view** — a tela de Simulação de Casos: seleção do shapefile,
  fonte e arquivos de forcing, configuração de ambiente com credenciais e
  download, modo de execução com faixas de busca ou fatores fixos, e o botão de
  iniciar.
- **build_deterministic_view** — a tela de Simulação Determinística, em três
  blocos: local e momento do vazamento, arquivos de forcing, e parâmetros do
  modelo.
- **build_stochastic_view** — a tela de Simulação Estocástica: entradas e
  ambiente, configuração de cada parâmetro sorteado com seus limites,
  configuração da grade fixa, e o botão de iniciar.
- **build_execution_view** — a tela de acompanhamento: título e subtítulo de
  status, barra de progresso, log rolante e botão de cancelar.
- **build_results_view** — a tela de resultados: indicadores dos parâmetros
  vencedores e do tempo decorrido, e o navegador de quadros com controle
  deslizante.
- **build_artifacts_view** — a tela de arquivos gerados, com tamanho e botão
  para abrir cada um no sistema.
- **build_sidebar** — o menu lateral com as seis entradas, devolvendo também os
  botões para que a navegação possa destacar o ativo.

### `src/ui/helpers.py`

Utilitários de apoio da interface.

**QueueWriter** — faz um destino de texto se comportar como arquivo, para
receber a saída redirecionada das simulações.

- **write** — acumula o texto recebido e, a cada quebra de linha, publica a
  linha completa como evento de log.
- **flush** — publica o que restou no acumulador, garantindo que a última linha
  não se perca.

Funções:

- **parse_float** — converte texto em número aceitando vírgula como separador
  decimal, e produz mensagem de erro nomeando o campo problemático.
- **validate_observed_zip** — confere que o arquivo escolhido existe, é um
  `.zip`, e contém os três componentes obrigatórios de um shapefile. Devolve se
  há também o arquivo de projeção, cuja ausência é apenas um aviso.
- **extract_observed_bounds** — lê o shapefile e devolve seus limites
  geográficos em coordenadas geográficas.
- **build_run_id** — gera um identificador único de execução a partir do
  instante atual, com precisão de microssegundos.
- **stage_observed_zip** — copia o shapefile escolhido para um diretório
  próprio da execução, preservando a entrada exata usada mesmo que o original
  seja alterado depois.
- **open_path** — abre um arquivo ou pasta no gerenciador do sistema,
  escolhendo o comando conforme o sistema operacional.
- **list_environments** — lista os ambientes disponíveis a partir dos arquivos
  de configuração encontrados.
- **extract_metrics** — lê a tabela de otimização de uma execução e extrai a
  melhor pontuação com seus parâmetros, devolvendo campos vazios quando o
  arquivo não existe ou não tem resultados válidos.
- **build_artifact_list** — lista os arquivos gerados por uma execução,
  incluindo os agregados do ensemble, ordenados do mais recente para o mais
  antigo.
- **build_frame_list** — localiza e ordena os quadros de comparação de uma
  simulação.

---

## `conf` — configuração Hydra

A configuração é declarativa e composta por partes que se combinam.

- **main.yaml** — o arquivo raiz. Declara quais variantes usar por padrão e
  define os caminhos de dados e os parâmetros de download: pasta de destino,
  margem e o recorte geográfico padrão.
- **environment/** — uma variante por período de interesse. Cada arquivo
  declara qual conjunto de dados baixar, o intervalo de datas, o deslocamento
  de horas a aplicar nas observações e os caminhos dos arquivos de forcing.
- **simulation/** — variantes de parâmetros de simulação: número de partículas,
  passos de tempo, duração e variáveis a exportar.
- **grid_format/** — variantes de resolução para a conversão em raster.
- **experiment/** — reservado para agrupamentos de experimento.

O ambiente é escolhido na interface por menu suspenso, e o recorte geográfico é
sobrescrito em tempo de execução a partir dos dados observados.

---

## `scripts` — utilitários avulsos

Programas independentes, que não fazem parte do pacote e são executados
diretamente. Não são importados pela aplicação.

- **run_forecast_hindcast_pairs.py** — gera pares de simulações que diferem
  apenas na origem dos dados de corrente (previsão contra reanálise), mantendo
  ponto, data, parâmetros e sorteios idênticos, para isolar o efeito da fonte
  de dados.
- **make_pair_gifs.py** e **make_pair_gifs_polygon.py** — geram animações
  comparando os resultados desses pares, como nuvem de pontos e como polígono.
- **convert.py** — converte trajetórias simuladas em rasters de grade regular.
- **grid_utils.py** — funções de grade e gravação de raster usadas pela
  conversão, incluindo filtros de suavização.
- **download_data.py** — download de dados por linha de comando, anterior à
  interface gráfica.
- **plot_offset_skillscore.py**, **plot_parametros_otimizados_1h.py**,
  **plot_parametros_otimizados_4h.py**, **plot_real_spills.py** — gráficos de
  análise dos resultados de otimização e das manchas observadas.
- **descompact.py** — utilitário de descompactação de arquivos.
- **analysis/** — estudos exploratórios de correntes e comparação entre modelos
  hidrodinâmicos.

---

## Fluxos de execução

### Simulação de Casos

1. O usuário escolhe o shapefile das manchas, os arquivos de forcing e o
   ambiente, e define se quer buscar os melhores parâmetros ou usar valores
   fixos.
2. O sistema copia o shapefile para o diretório da execução, lê seus limites e
   os expande com folga.
3. A preparação carrega a configuração, lê e normaliza as manchas, e confere se
   os dados de forcing cobrem a área e o período observados. Quando vários
   offsets ambientais estão em jogo, os inviáveis são descartados.
4. Se a busca estiver ligada, o sistema testa as combinações de WDF e CDF para
   cada offset viável e escolhe a de melhor pontuação.
5. A simulação roda com os parâmetros escolhidos, do primeiro ao último
   instante observado.
6. São geradas a animação comparativa e as imagens quadro a quadro.
7. A tela de resultados mostra os parâmetros vencedores e a pontuação; a de
   artefatos lista os arquivos.

### Simulação Determinística

1. O usuário digita coordenadas, data e hora, duração, características do
   vazamento e parâmetros do modelo.
2. O sistema valida cada campo, define a área de forcing como uma caixa em
   volta do ponto e carrega a configuração.
3. Grava o registro dos parâmetros e roda a simulação a partir do ponto.
4. A saída aparece na aba de artefatos e o resumo no log.

Este modo não depende de observação, então não há pontuação de qualidade a
calcular — a aba de resultados permanece vazia.

### Simulação Estocástica

1. O usuário escolhe o shapefile, define quantos membros rodar, a semente, as
   distribuições de WDF, CDF e lag temporal, e a grade dos mapas.
2. A preparação é a mesma da validação, reaproveitada integralmente.
3. Os parâmetros de todos os membros são sorteados de uma vez e gravados antes
   da execução começar.
4. Os membros rodam em paralelo. A cada conclusão, a tabela de sorteios é
   reescrita com o status e o progresso é atualizado na tela.
5. As saídas bem-sucedidas são convertidas em mapas de presença, somadas e
   divididas pelo número de membros válidos.
6. São gravados os mapas de área varrida, de instante final e um por hora.

---

## Formatos de saída

**Validação e determinística** — em `data/2-simulated/<nome_da_execução>/`:

| Arquivo | Conteúdo |
|---|---|
| `*.nc` | trajetórias das partículas ao longo do tempo |
| `*.json` | parâmetros exatos usados na execução |
| `*_compare.gif` | animação comparando simulado e observado |
| `*_frames/` | imagens quadro a quadro |
| `wdf_cdf_optimization_fast.csv` | pontuação de cada combinação testada |
| `wdf_cdf_optimization_fast.json` | a melhor combinação e as faixas testadas |

**Estocástica** — em `data/2-simulated/stochastic/<nome_da_execução>/`:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | configuração completa, para reprodução |
| `sampled_parameters.csv` | sorteio e status de cada membro |
| `summary.json` | contagens, caminhos e tempo decorrido |
| `logs.txt` | log da execução |
| `individual_runs/run_NNNN/simulation.nc` | trajetórias de cada membro |
| `aggregated/probability_map.tif` | probabilidade na área varrida |
| `aggregated/probability_final_timestep_map.tif` | probabilidade no instante final |
| `aggregated/hit_count_*.tif` | contagem bruta de membros por célula |
| `aggregated/hourly_probability_rasters/` | um mapa por instante |

Os mapas de probabilidade têm valores entre zero e um, indicando a fração de
membros do ensemble que atingiram cada célula. Abrem diretamente em qualquer
software de SIG, já georreferenciados.

---

## Notas operacionais

**Consumo de memória do ensemble.** Os mapas por instante ficam todos em
memória durante a agregação. Com a grade padrão — limites das manchas expandidos
em um grau e resolução de um milésimo de grau — cada mapa ocupa dezenas de
megabytes, e uma simulação de um dia inteiro pode passar de um gigabyte e meio.
Para execuções longas, aumente a resolução ou reduza a área.

**O download traz apenas salinidade e temperatura.** Corrente e vento não são
baixados pela interface: precisam ser fornecidos pelos seletores de arquivo. A
configuração declara identificadores de conjunto de dados de corrente que hoje
não são usados por esse caminho.

**Nem todo ambiente declara arquivo de corrente padrão.** Alguns arquivos de
configuração não trazem o caminho do dado de corrente. Se nenhum arquivo for
selecionado ao usar esses ambientes, a execução falha ao tentar
consultar esse campo. Na prática, selecionar os arquivos manualmente evita o
problema.

**Simulação sem vento.** No modo determinístico, não selecionar arquivo de
vento é uma escolha válida: o vento é fixado em zero e a deriva fica puramente
por corrente. Útil para isolar o efeito da corrente ou simular em períodos sem
dado de vento disponível.

**Reprodutibilidade do ensemble.** A mesma semente produz exatamente o mesmo
conjunto de sorteios. Repetir uma execução com a mesma configuração gera os
mesmos parâmetros para cada membro.

**Cancelamento.** O pedido de cancelamento é cooperativo: a execução para no
próximo ponto seguro, não instantaneamente. No ensemble em paralelo, os membros
já iniciados terminam antes de a execução encerrar.

**Ondas estão desligadas.** Os dados de forcing disponíveis não trazem campo de
onda, então a deriva de Stokes fica desabilitada em todos os modos.
