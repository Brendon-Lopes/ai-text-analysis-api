# AI Text Analysis API

API backend em Python para análise de texto com NLP (Natural Language Processing).

Três funcionalidades principais expostas via HTTP:

- Extração de palavras-chave
- Análise de sentimento
- Resumo automático de texto

---

## Tecnologias

| Biblioteca                | Papel no projeto                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------- |
| **FastAPI**               | Framework HTTP — define rotas, valida dados, gera docs automáticas                     |
| **Pydantic**              | Validação de entrada e saída — garante que o JSON recebido/enviado tem o formato certo |
| **spaCy**                 | Processamento de linguagem natural — tokenização, análise morfológica (POS tagging)    |
| **PyTorch / HuggingFace** | Deep Learning e modelos de IA (Transformers) — sumarização abstractiva via Seq2Seq     |
| **Uvicorn**               | Servidor ASGI — executa a aplicação FastAPI                                            |
| **Pytest**                | Framework de testes                                                                    |
| **HTTPX / TestClient**    | Cliente HTTP usado nos testes de integração                                            |

---

## Estrutura do Projeto

```
ai-text-analysis-api/
├── app/
│   ├── main.py                  # Cria o app FastAPI e registra as rotas
│   ├── routes/
│   │   └── text_routes.py       # Define os endpoints HTTP (/text/*)
│   ├── schemas/
│   │   └── text_schema.py       # Modelos de request e response (Pydantic)
│   └── services/
│       ├── text_service.py          # Lógica de NLP clássico (keywords, sentiment)
│       └── summarization_service.py # Lógica de geração de resumos com IA (PyTorch)
├── tests/
│   ├── test_health.py           # Testes do endpoint de saúde
│   └── test_text_routes.py      # Testes dos endpoints de texto
├── requirements.txt
└── README.md
```

Cada pasta tem uma responsabilidade clara:

- **`routes`** recebe a requisição HTTP e devolve a resposta — sem lógica de negócio.
- **`schemas`** define o contrato: qual JSON entra, qual JSON sai.
- **`services`** contém toda a inteligência — é aqui que o NLP acontece.
- **`main.py`** conecta tudo: cria o app e registra as rotas.

---

## Como Instalar e Rodar

O modo recomendado e mais fácil de rodar este projeto é utilizando o **Docker**. Ele vai lidar automaticamente com as dependências complexas (como o PyTorch) e vai fazer o download prévio dos modelos de inteligência artificial durante a fase de build.

```bash
docker compose up --build
```

A API estará disponível em `http://localhost:8000`.
Documentação automática e interativa do FastAPI disponível em `http://localhost:8000/docs`.

_(Se você preferir rodar localmente sem o Docker, é necessário criar um `venv`, instalar as dependências do `requirements.txt` e `requirements.prod.txt`, baixar os modelos do spaCy via comando e rodar o `uvicorn` manualmente)._

---

## Endpoints

### `GET /health`

Verifica se a API está no ar.

**Response:**

```json
{
  "status": "ok"
}
```

---

### `POST /text/keywords`

Extrai palavras-chave do texto.

**Request:**

```json
{
  "text": "Artificial intelligence is transforming software engineering."
}
```

**Response:**

```json
{
  "keywords": ["intelligence", "software", "engineering"]
}
```

**Como funciona:**

O spaCy analisa cada palavra do texto e identifica sua classe gramatical (POS tagging). O serviço mantém apenas substantivos e nomes próprios (`NOUN`, `PROPN`), remove stopwords (palavras sem significado semântico como "is", "the", "a") e retorna até 10 palavras únicas.

---

### `POST /text/sentiment`

Analisa se o texto tem sentimento positivo, negativo ou neutro.

**Request:**

```json
{
  "text": "This product is absolutely amazing and wonderful!"
}
```

**Response:**

```json
{
  "sentiment": "positive",
  "score": 1.0
}
```

O campo `score` varia de `-1.0` (totalmente negativo) a `1.0` (totalmente positivo).

**Como funciona:**

A implementação atual é uma abordagem baseline baseada em léxico: conta palavras positivas e negativas conhecidas no texto e calcula um score proporcional. É uma técnica educacional — sistemas em produção usam modelos de machine learning treinados para isso.

```
score = (positivas - negativas) / (positivas + negativas)
```

---

### `POST /text/summary`

Gera um resumo do texto mantendo as frases mais relevantes.

**Request:**

```json
{
  "text": "FastAPI is a modern web framework for building APIs with Python. It is based on standard Python type hints and provides automatic documentation. Many developers love FastAPI for its speed and simplicity. The framework is built on top of Starlette and Pydantic."
}
```

**Response:**

```json
{
  "summary": "FastAPI is a modern web framework for building APIs with Python. Many developers love FastAPI for its speed and simplicity."
}
```

**Como funciona:**

A API utiliza **sumarização abstractiva**, que emprega modelos de Deep Learning da arquitetura _Seq2Seq_ (Transformers via biblioteca da Hugging Face e PyTorch).

Diferente do método _"extrativo"_ (que apenas seleciona frases inteiras baseando-se em suas pontuações e frequências), este método abstractivo permite que a inteligência artificial leia o texto, **compreenda o seu significado** e gere um texto completamente novo, com as próprias palavras do modelo.

- **Modelos Intercambiáveis**: Suportado por meio da variável de ambiente `SUMMARIZATION_MODEL` no `docker-compose.yml`. O padrão do projeto é o `distilbart` (mais rápido e leve), mas tem suporte programado também para o `flan-t5` do Google.
- A configuração da geração é feita explicitamente interatuando com os tensores do PyTorch (sem abstração por cima), usando técnicas como tokenização por ID, formatação de máscaras de atenção e _beam search_ pra explorar as respostas mais coerentes.

---

## Como Rodar os Testes

```bash
python -m pytest tests/ -v
```

Saída esperada:

```
tests/test_health.py::test_health_returns_ok PASSED
tests/test_text_routes.py::test_keywords_returns_list PASSED
tests/test_text_routes.py::test_keywords_are_strings PASSED
tests/test_text_routes.py::test_keywords_returns_at_most_ten PASSED
tests/test_text_routes.py::test_sentiment_returns_valid_label PASSED
tests/test_text_routes.py::test_sentiment_returns_score PASSED
tests/test_text_routes.py::test_sentiment_positive_text PASSED
tests/test_text_routes.py::test_sentiment_negative_text PASSED
tests/test_text_routes.py::test_summary_returns_string PASSED
tests/test_text_routes.py::test_summary_is_shorter_than_original PASSED
tests/test_text_routes.py::test_endpoints_reject_missing_text PASSED

11 passed in 1.45s
```

Os testes cobrem:

- Código de status HTTP correto
- Formato e tipos da resposta
- Comportamento com texto positivo e negativo
- Validação: todos os endpoints rejeitam requisições sem o campo `text` (HTTP 422)

---

## Fluxo de uma Requisição

```
Cliente HTTP (curl, /docs, frontend)
         │
         │  POST /text/keywords  {"text": "..."}
         ▼
   app/main.py  ──── registra as rotas e recebe a requisição
         │
         ▼
   app/routes/text_routes.py  ──── valida o JSON de entrada via Pydantic
         │
         ▼
   app/services/text_service.py  ──── processa o texto com spaCy
         │
         ▼
   app/routes/text_routes.py  ──── serializa o resultado no schema de response
         │
         ▼
   Cliente recebe  {"keywords": [...]}
```

---

## Conceitos de NLP Usados

| Conceito                   | O que é                                                                      | Onde é usado                         |
| -------------------------- | ---------------------------------------------------------------------------- | ------------------------------------ |
| **Tokenização**            | Dividir o texto em unidades (palavras, pontuação)                            | Em todos os endpoints                |
| **POS Tagging**            | Identificar a classe gramatical de cada token (substantivo, verbo, etc.)     | `extract_keywords`                   |
| **Stopwords**              | Palavras sem carga semântica ("the", "is", "a") que são removidas            | `extract_keywords`, `summarize_text` |
| **Lematização**            | Reduzir palavras à forma base ("running" → "run")                            | `extract_keywords`                   |
| **Sumarização abstrativa** | Usar modelos Seq2Seq de Deep Learning para compreender e gerar novos resumos | `summarization_service`              |
