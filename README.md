# Previsão de Entrega

Projeto de portfólio para estágio em AI Engineering. No momento em que o pedido é criado, uma regressão Ridge estima a duração em minutos. A API também estima a chance de ultrapassar um prazo definido pelo projeto e mostra as variáveis que mais contribuíram para a previsão linear.

## Dados e definição do alvo

O `historical_data.csv` reúne entregas da DoorDash do início de 2015. A fonte descreve ruído adicionado para ocultar detalhes operacionais. O CSV local tem 197.428 linhas e contém horários de criação e entrega, categoria e identificador de loja, região, protocolo, itens e preços, contagens operacionais e duas estimativas parciais. A licença MIT do CSV permanece pendente de confirmação. Baixe o arquivo da fonte e mantenha a cópia local em `data/raw/`, ignorada pelo Git. [Dataset no Kaggle](https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction) · [Dicionário de dados](https://platform.stratascratch.com/data-projects/delivery-duration-prediction)

- **Duração real em minutos** = (`actual_delivery_time` − `created_at`) em segundos ÷ 60.
- **Prazo de referência** = estimativa de transmissão do pedido + estimativa de trajeto loja–cliente + tempo adicional histórico da categoria + margem de 5 minutos. Os dois primeiros campos vêm em segundos e são convertidos para minutos. O prazo completo mostrado ao cliente está ausente do CSV.
- **Evento de atraso no projeto** ocorre quando a duração real é maior que o prazo de referência. O tempo adicional histórico é a mediana suavizada por categoria da duração real menos as duas estimativas parciais, com piso de 2 minutos. Pode incluir preparo, espera, deslocamento até a loja e erros nas estimativas. Categorias sem histórico usam a mediana global do treino.

O pipeline imputa valores numéricos ausentes, não finitos ou negativos com medianas do treino. Categorias ausentes usam a moda histórica da loja ou `unknown`. A categoria, a região e o protocolo entram como indicadores, e as variáveis temporais são calculadas no fuso `America/Los_Angeles`. A duração real e a parcela histórica calculada com o alvo ficam fora das entradas do Ridge.

## Instalação e execução

Execute os comandos na raiz do projeto. A avaliação foi reproduzida com Python 3.14.10 e as versões de dependências em `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Com o CSV em `data/raw/historical_data.csv`:

```powershell
.\.venv\Scripts\python.exe -m src.train --data data/raw/historical_data.csv
.\.venv\Scripts\python.exe -m uvicorn src.api:create_app --factory
```

Abra [o painel local](http://127.0.0.1:8000/) ou [a documentação interativa da API](http://127.0.0.1:8000/docs). O painel consulta `POST /predict` e mostra a origem do modelo. O campo `created_at` recebe data e hora ISO 8601. Horários sem fuso são tratados como UTC. A API lê o artefato `models/model_bundle.pkl` ao iniciar. Carregue apenas arquivos de modelo produzidos localmente ou de fonte confiável.

Exemplo no PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' -Body '{"created_at":"2015-02-06T19:30:00-08:00"}'
```

O exemplo usa os valores de demonstração definidos para os demais campos. A resposta inclui duração estimada, referência, diferença em minutos, risco, contribuições e origem do modelo. `is_high_risk` usa corte ilustrativo de 50%. A previsão e a explicação padrão funcionam sem LLM.

## Testes e demonstração sintética

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --synthetic-check
$env:ETA_MODEL_BUNDLE = 'models_synthetic/model_bundle.pkl'
.\.venv\Scripts\python.exe -m uvicorn src.api:create_app --factory
```

Os dados sintéticos exercitam o código e refletem apenas as regras do gerador. Os artefatos da demonstração ficam em `models_synthetic/`. Para voltar a usar o modelo histórico, encerre o servidor, rode `Remove-Item Env:ETA_MODEL_BUNDLE` e inicie a API novamente. Com o servidor ativo, `python -m notebooks.demo_cenarios` consulta a API e imprime dois exemplos.

## Como a avaliação foi feita

1. Calculei a duração pelo par de timestamps e removi sete linhas com horário ausente ou inválido. O recorte principal mantém durações de 5 a 180 minutos.
2. Ordenei os pedidos pela criação e usei cortes de aproximadamente 70%, 15% e 15%. Timestamps iguais permanecem na mesma janela. Excluí 461 pedidos que ainda estavam em andamento no início da validação e 181 que ainda estavam em andamento no início do teste. Restaram 137.635 pedidos de treino, 29.411 de validação e 29.592 de teste.
3. Comparei a mediana da duração no treino, um baseline operacional que usa a referência sem a margem e a regressão Ridge. O baseline de risco usa a frequência de ultrapassagem observada no treino. O Ridge estima o risco pela fração de erros de validação que excede a diferença entre prazo de referência e previsão. O teste temporal mede o desempenho dos modelos já ajustados.

### Resultado no recorte principal

| Modelo | MAE | RMSE | P90 do erro absoluto | Average Precision | Brier |
|---|---:|---:|---:|---:|---:|
| Mediana do treino | 13,85 min | 19,24 min | 28,98 min | 0,4099 | 0,2432 |
| Baseline operacional | 13,23 min | 18,59 min | 27,86 min | 0,4099 | 0,2432 |
| Ridge | **11,62 min** | **15,92 min** | **22,81 min** | **0,6758** | **0,1986** |

O MAE mede a magnitude média do erro. O RMSE dá mais peso aos erros grandes, e o P90 indica o valor que cerca de 90% dos erros absolutos ficaram abaixo ou igual. Average Precision mede a qualidade do ranking de risco com os empates agrupados. A métrica Brier é a média do erro quadrático das probabilidades, em que valores menores indicam menor erro.

O risco empírico médio foi 42,45%, e a frequência observada foi 40,99%. O Brier foi 0,1986, ante 0,2041 da versão gaussiana original avaliada com a mesma regressão. Na faixa de 80% a 100%, a previsão média foi 88,90% e a frequência observada foi 81,77%. O método ainda superestima o risco nessa faixa. No corte ilustrativo de 50%, o Ridge teve precisão de 68,21% e recall de 51,03%.

### Sensibilidade às durações extremas

Com o mesmo modelo e o mesmo início da janela de teste, incluir todas as durações positivas elevou a amostra de 29.592 para 29.611 pedidos. O MAE do Ridge passou de 11,62 para 11,97 minutos, e o RMSE subiu de 15,92 para 36,61 minutos. O relatório completo em `models/evaluation_report.json` contém os dois recortes, as faixas de calibração e os períodos em UTC. Uma cópia da avaliação histórica acompanha o código em `evaluation_report.json`.

## Limitações e próximos passos

- Os resultados descrevem a amostra histórica da DoorDash com ruído e uma janela de teste de poucos dias. A aplicação a outras regiões, períodos ou ao iFood exigiria avaliação própria.
- O prazo de referência tem margem fixa de 5 minutos escolhida para a demonstração. O risco depende dessa definição. A distribuição de erros é global e pode mudar por região, horário ou nível de demanda.
- As contribuições em minutos descrevem a previsão linear antes do limite aplicado à saída. Variáveis correlacionadas podem dividir influência. As contribuições não estabelecem causas.
- Investigue os registros extremos, avalie a calibração por região e horário e compare um modelo não linear no mesmo protocolo temporal.

## Estrutura

| Caminho | Responsabilidade |
|---|---|
| `src/data_processing.py` | Alvo, separação temporal, prazo e variáveis |
| `src/models.py` | Baselines, Ridge, risco, contribuições e métricas |
| `src/train.py` | Treino, validação, teste e geração de artefatos |
| `src/api.py`, `src/dashboard.html` | API e painel |
| `src/explainer.py` | Explicação em português |
| `tests/`, `notebooks/` | Testes e demonstração por linha de comando |
