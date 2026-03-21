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

```python
media_notas = (nota_port + nota_mat + nota_ing) / 3
engajamento_geral = (ieg + iaa + ips) / 3
score_pedagogico = (ida + ipp + ipv) / 3
