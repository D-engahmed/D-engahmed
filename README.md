<div align="center">

# Ahmed Abdallah

### AI Engineer · Generative AI · LLM Systems · AI Infrastructure

<p>
<a href="https://github.com/D-engahmed"><img src="https://img.shields.io/badge/GitHub-D--engahmed-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
<a href="https://www.linkedin.com/in/ahmed-elkossairy/"><img src="https://img.shields.io/badge/LinkedIn-Ahmed_Abdallah-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
<a href="https://www.kaggle.com/ahmedelkossairy"><img src="https://img.shields.io/badge/Kaggle-AhmedElkossairy-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle"/></a>
</p>

![Profile Views](https://komarev.com/ghpvc/?username=D-engahmed&label=Profile%20Views&style=flat-square)

</div>

---

## About

I'm an **AI Engineer and independent AI researcher** focused on building systems from model fundamentals to production infrastructure.

My work spans:

- **Generative AI & LLM engineering** — RAG, agents, tool use, inference, fine-tuning, evaluation, and model architecture.
- **AI systems engineering** — APIs, orchestration, multi-tenant SaaS, streaming, observability, and production workflows.
- **Model research** — studying and implementing modern architectures such as Titans, MIRAS, MoE systems, long-context methods, and efficient attention/memory mechanisms.
- **Applied AI** — healthcare, pharmaceutical quality systems, NLP, computer vision, and multimodal applications.
- **AI infrastructure** — Docker, CI/CD, GitHub Actions, cloud deployment, model serving, and reproducible engineering.

I'm currently completing a **B.Sc. in Electronics & Communications Engineering at Helwan University (expected 2027)**.

> I care about understanding the system underneath the abstraction—not only calling an API, but understanding the model, data pipeline, runtime, and infrastructure around it.

---

## What I'm Building

### 🧠 ATHLLM — From the Model Up

An open-source Arabic/English LLM research project focused on building the stack rather than treating the model as a black box.

Current direction:

- Custom tokenizer and bilingual data pipeline
- Arabic + English pretraining
- Transformer and memory/attention research
- Titans / MIRAS-inspired experiments
- MoE experimentation
- Efficient training and inference
- Long-context research
- Evaluation and reproducible experiments

**Research direction:** build progressively from a text foundation model toward multimodal capabilities.

[→ ATHLLM](https://github.com/D-engahmed/ATHLLM)

---

### ⚙️ Ancient — AI Coding-Agent Infrastructure

A unified platform for building AI coding agents and AI-native developer products.

The architecture is intended to support:

- CLI
- IDE integrations
- Web applications
- Coding agents
- Design agents
- Cowork-style workflows
- MCP/tool integration
- Multi-provider routing
- BYOK provider connections
- Streaming
- Sessions and context
- Guardrails
- GitHub workflows and CI/CD
- APIs for building AI applications on top of the platform

The goal is not to build another thin chat wrapper. The goal is to provide the **runtime and engineering foundation around coding agents**.

[→ Ancient](https://github.com/D-engahmed/ancient)

---

### 🧪 QCSTS — Pharmaceutical Quality & Stability SaaS

A multi-tenant quality and stability management platform for pharmaceutical organizations and laboratories.

Engineering focus includes:

- Organization/site isolation
- RBAC and object authorization
- Stability studies and protocols
- Controlled results and approval workflows
- Audit trails and immutable records
- OOS / OOT / Deviation / CAPA / Change Control
- Reporting and exports
- Subscription and entitlement architecture
- Paymob payment integration
- PostgreSQL, Redis, Celery, Docker and Nginx
- Production CI/CD and release gates

QCSTS is positioned as **designed for GxP-regulated environments with a validation-ready architecture**, not as automatically certified regulatory software.

[→ QCSTS](https://github.com/D-engahmed/QCSTS)

---

### 🏥 Medical AI & Healthcare Systems

I also build healthcare-oriented AI systems covering:

- Medical imaging
- Clinical decision-support prototypes
- Multimodal AI
- Medical NLP
- Healthcare data pipelines
- AI-assisted hospital systems

The emphasis is on combining **AI with real software architecture**, rather than building isolated notebooks.

[→ Medical AI Platform](https://github.com/D-engahmed/medical_ai_platform)

---

## Research

### Current Areas

| Area | Focus |
|---|---|
| **LLM Architecture** | Transformers, attention, memory, MoE, long context |
| **Generative AI** | RAG, agents, tool use, structured generation |
| **Model Training** | Pretraining, fine-tuning, optimization |
| **Inference** | Quantization, serving, latency and memory efficiency |
| **AI Agents** | Coding agents, orchestration, MCP, tool execution |
| **Multimodal AI** | Text, vision, and future audio integration |
| **Arabic AI** | Arabic/English datasets, tokenization, evaluation |
| **AI Infrastructure** | Docker, CI/CD, model serving, cloud systems |

### Papers / Architectures I Study

- Titans
- MIRAS
- Mixture-of-Experts architectures
- Long-context architectures
- Modern open-weight LLMs
- Efficient attention and memory mechanisms
- Retrieval-augmented generation
- Agentic software engineering

---

## Technical Stack

<div align="center">

### Languages

<img src="https://skillicons.dev/icons?i=python,cpp,js,ts,bash,sql" alt="Languages"/>

### AI / ML

<img src="https://skillicons.dev/icons?i=pytorch,tensorflow,opencv" alt="AI ML"/>

<br/>

PyTorch · Transformers · Hugging Face · Scikit-learn · RAG · LangChain · LangGraph · FAISS · Vector Databases

### Backend & Systems

<img src="https://skillicons.dev/icons?i=fastapi,django,nodejs,postgres,redis,docker,kubernetes,nginx" alt="Backend and Systems"/>

### Engineering & Infrastructure

<img src="https://skillicons.dev/icons?i=git,github,githubactions,linux,aws,azure" alt="Engineering and Infrastructure"/>

CI/CD · Docker Compose · Celery · MLflow · REST APIs · MCP · Observability · Production Testing

</div>

---

## Selected Engineering Work

### LLM / Generative AI

- Building RAG systems with retrieval, reranking, context construction, and evaluation.
- Designing coding-agent runtimes rather than only prompt-based assistants.
- Working with multiple model providers and local inference.
- Fine-tuning and experimenting with open-weight models.
- Studying model internals and implementing architectural ideas from research papers.

### Production AI

- Designing multi-tenant SaaS architectures.
- Building API-first AI systems.
- Implementing authorization and tenant isolation at the backend boundary.
- Containerizing services and supporting production deployment.
- Building CI/CD pipelines with automated tests and release gates.
- Integrating AI into domain-specific business workflows.

### Classical ML / Deep Learning

- NLP classification and sequence models.
- Computer vision and object detection.
- Medical imaging.
- Feature engineering and model evaluation.
- Model compression and inference optimization.

---

## Engineering Principles

```text
Understand the abstraction
        ↓
Measure the system
        ↓
Design the boundary
        ↓
Implement the smallest correct primitive
        ↓
Test failure modes
        ↓
Automate the workflow
        ↓
Deploy reproducibly
        ↓
Observe and iterate
```

I prefer systems that are:

- **Explicit** over magical
- **Testable** over optimistic
- **Observable** over opaque
- **Reproducible** over environment-dependent
- **Secure by architecture** over convention alone
- **Simple to understand** without sacrificing capability

---

## GitHub Projects

| Project | What it demonstrates |
|---|---|
| [**Ancient**](https://github.com/D-engahmed/ancient) | AI coding-agent infrastructure and multi-provider runtime |
| [**ATHLLM**](https://github.com/D-engahmed/ATHLLM) | LLM training, data, tokenizer and architecture research |
| [**QCSTS**](https://github.com/D-engahmed/QCSTS) | Multi-tenant pharmaceutical SaaS and production engineering |
| [**Medical AI Platform**](https://github.com/D-engahmed/medical_ai_platform) | Applied AI and healthcare systems |
| [**RAG Learning**](https://github.com/D-engahmed/RAG_learning) | Retrieval-augmented generation experiments |
| [**PyTorch RNN Text Classification**](https://github.com/D-engahmed/pytorch_rnn_text_classification) | Deep learning and sequence-model fundamentals |
| [**Airline Delay Cause**](https://github.com/D-engahmed/Airline_Delay_Cause) | Large-scale data analysis and predictive modeling |

---

## Education

**B.Sc. Electronics & Communications Engineering**  
Helwan University · Expected 2027 · Egypt

Relevant areas:

Machine Learning · Deep Learning · NLP · Computer Vision · Signal Processing · Pattern Recognition · Embedded Systems

---

## Current Roadmap

```text
AI Engineering
     │
     ├── Production LLM Systems
     │      ├── RAG
     │      ├── Agents
     │      ├── Tool Use
     │      └── Evaluation
     │
     ├── Model Research
     │      ├── ATHLLM
     │      ├── Memory / Attention
     │      ├── MoE
     │      └── Long Context
     │
     ├── AI Infrastructure
     │      ├── Containers
     │      ├── CI/CD
     │      ├── Model Serving
     │      └── Cloud
     │
     └── AI Products
            ├── Developer Tools
            ├── Healthcare
            └── Pharmaceutical Systems
```

---

## Connect

<div align="center">

<a href="https://www.linkedin.com/in/ahmed-elkossairy/">LinkedIn</a> ·
<a href="https://github.com/D-engahmed">GitHub</a> ·
<a href="https://www.kaggle.com/ahmedelkossairy">Kaggle</a>

<br/><br/>

**Open to AI engineering, research, and serious systems-building opportunities.**

</div>

---

<div align="center">

<sub>Building AI systems from the model layer to production.</sub>

</div>