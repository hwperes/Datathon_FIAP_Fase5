# 📊 Datathon POSTECH - Passos Mágicos

## 1. Introdução

Este projeto foi desenvolvido como parte do Datathon da POSTECH, com o objetivo de aplicar técnicas de análise de dados e machine learning para gerar impacto social real.

A iniciativa utiliza dados da ONG Passos Mágicos, que atua na transformação da vida de crianças e jovens em situação de vulnerabilidade social por meio da educação.

O projeto combina:
- Análise exploratória de dados (EDA)
- Engenharia de features
- Modelagem preditiva
- Desenvolvimento de aplicação interativa com Streamlit

---

## 2. Contexto do Negócio

A Associação Passos Mágicos atua há mais de 30 anos promovendo desenvolvimento educacional e social, utilizando indicadores próprios para mensurar evolução dos alunos.

Principais indicadores utilizados:

- IAN – Adequação ao nível
- IDA – Desempenho acadêmico
- IEG – Engajamento
- IAA – Autoavaliação
- IPS – Aspectos psicossociais
- IPP – Aspectos psicopedagógicos
- IPV – Ponto de virada
- INDE – Índice consolidado do aluno

O desafio central é transformar esses dados em insights acionáveis e previsões de risco, apoiando a tomada de decisão da instituição.

---

## 3. Objetivos do Projeto

- Analisar a evolução educacional dos alunos ao longo dos anos
- Identificar fatores que influenciam desempenho e engajamento
- Detectar padrões de risco de defasagem
- Construir um modelo preditivo de risco
- Disponibilizar uma aplicação interativa para uso da ONG

---

## 4. Metodologia

### 4.1 Preparação dos Dados e Feature Engineering

Foram realizadas etapas de tratamento e enriquecimento dos dados para melhorar a capacidade preditiva do modelo.

Principais ações:

- Tratamento de valores nulos e inconsistências
- Padronização de variáveis categóricas
- Encoding de variáveis
- Criação de features derivadas:

- media_notas = (nota_port + nota_mat + nota_ing) / 3
- engajamento_geral = (ieg + iaa + ips) / 3
- score_pedagogico = (ida + ipp + ipv) / 3

Essas variáveis agregadas permitem capturar melhor o comportamento multidimensional dos alunos.

### 4.2 Definição da Variável Target

A variável target utilizada foi:

- risco_defasagem

Classificação:

- 0 → Sem risco
- 1 → Com risco de defasagem

A construção dessa variável considerou padrões de desempenho acadêmico, engajamento e adequação ao nível.

### 4.3 Separação dos Dados

Foi utilizada divisão estratificada para preservar a distribuição da variável target:

from sklearn.model_selection import train_test_split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 4.4 Modelagem Preditiva

O modelo final escolhido foi o XGBoost Classifier, devido à sua alta performance em problemas de classificação tabular.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
```
📌 Features utilizadas no modelo
```python
features = [
    "idade_22",
    "genero_bin",
    "ieg",
    "ida",
    "iaa",
    "ips",
    "ipp",
    "ipv",
    "media_notas",
    "pedra_num"
]
```

O modelo combina variáveis originais e derivadas, capturando tanto aspectos individuais quanto agregados do desempenho dos alunos.

### 4.5 Avaliação dos Resultados

Foram utilizadas as seguintes métricas:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
```
📊 Resultados do Modelo

⚠️ Substituir pelos valores reais obtidos no notebook

- Accuracy: 86,64%
- Precision: 86,93%
- Recall: 91,07%
- F1 Score: 88,95%
- ROC AUC: 92,3%

Foco principal do projeto: maximizar o Recall, garantindo maior identificação de alunos em risco.

---

## 5. Principais Insights
### 5.1 Evolução e Impacto do Programa
Observa-se evolução consistente dos indicadores ao longo dos anos
Alunos em fases mais avançadas (Topázio) apresentam melhor desempenho
Tempo no programa está positivamente correlacionado com o INDE

### 5.2 Correlações Multidimensionais

Principais relações identificadas:

- Forte correlação entre IEG (engajamento) e IDA (aprendizado)
- Indicadores psicossociais (IPS) impactam diretamente o desempenho
- Baixo engajamento é um forte preditor de risco
- INDE é altamente influenciado por IDA, IEG e IPP

### 5.3 Performance do Modelo Preditivo

O modelo foi capaz de identificar padrões relevantes de risco com base em:

- Engajamento (IEG)
- Desempenho acadêmico (IDA)
- Indicadores psicossociais (IPS)
- Média de notas
- Classificação do aluno (pedra)

Variáveis mais importantes (feature importance):

model.feature_importances_

---

## 6. Aplicação em Streamlit

Foi desenvolvida uma aplicação interativa para disponibilizar o modelo para uso prático.

### 6.1 Funcionalidades
- Input manual dos dados do aluno
- Cálculo automático de features derivadas
- Previsão de risco em tempo real
- Exibição da probabilidade de defasagem
- Interface simples e intuitiva

## 🔗 Link da Aplicação
Acesse o modelo preditivo em tempo real: [https://datathonfiapfase5-grupo165.streamlit.app/](https://datathonfiapfase5-grupo165.streamlit.app/)

---

## Dashboard Analítico (Power BI)

Foi construída uma base analítica em português, pronta para consumo no Power BI:


Arquivo principal:
- `base_tratada.xlsx`

[Dashboard - Case Passos Mágicos - Grupo 165](https://app.powerbi.com/view?r=eyJrIjoiMWYxZWRlNmQtM2RkMC00MzQ4LWE3ZGYtNTZlNzkwMzVlMmQ0IiwidCI6ImNmNzJlMmJkLTdhMmItNDc4My1iZGViLTM5ZDU3YjA3Zjc2ZiIsImMiOjR9)

## 7. Estrutura do Repositório
```bash
├── .streamlit/
│   └── config.toml
├── notebooks/
│   └── Datathon_Fase5.ipynb
│   └── Tratamento_Datathon_Fase5.ipynb
├── data/
│   ├── raw/
│   │   └── base_tratada.xlsx
│   └── processed/
│       └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
├── docs/
│   ├── doc_modelagem.md
│   │── Estudo de Obesidade.pbix
│   └── index.md
│       
├── modelos/
│   ├── config_passos_magicos.joblib
│   └── modelo_passos_magicos.joblib
├── referências/
│   └── Dicionário Dados Datathon.pdf
│   └── POSTECH - DTAT - Datathon - Fase 5.pdf
├── streamlit_app.py
├── requirements.txt
└── README.md
```
---

## 8. Documentação Técnica
Tecnologias utilizadas
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
  

---

📌 Conclusão

Este projeto demonstra como técnicas de ciência de dados podem ser aplicadas para gerar impacto social, permitindo identificar alunos em risco de defasagem de forma antecipada.

A solução desenvolvida contribui diretamente para a tomada de decisão da Passos Mágicos, possibilitando intervenções mais rápidas e assertivas.

## Equipe

- **Fabiana Cardoso da Silva**  
- **Henrique do Couto Santos**  
- **Henrique Waideman Peres**
