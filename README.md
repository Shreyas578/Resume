# Senior AI Engineer Candidate Ranker

A Streamlit dashboard for ranking AI Engineer candidates using heuristic scoring.

---

## Features

- Upload candidates.jsonl
- Rank thousands of candidates
- Production-ready scoring algorithm
- Candidate reasoning
- Score visualization
- Download submission.csv

---

## Installation

Clone the repository.

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app.py
```

---

## Project Structure

```
CandidateRanker/
│
├── app.py
├── ranker.py
├── requirements.txt
├── README.md
├── submission_metadata.yaml
└── output/
```

---

## Input

A JSONL file where every line is one candidate object.

Example:

```json
{
  "candidate_id":"123",
  "profile":{...},
  "skills":[...]
}
```

---

## Output

A CSV file

```
candidate_id,rank,score,reasoning
```

---

## Ranking Criteria

The scoring algorithm evaluates:

- Embedding experience
- Vector databases
- Evaluation metrics
- Production deployment
- Product-company experience
- Experience level
- Location
- Notice period
- Platform engagement
- Fine-tuning
- Learning-to-Rank
- NLP background

It also filters:

- Consulting-only careers
- CV/Speech-only profiles
- Honeypot profiles
- Wrong job roles
- Pure researchers
- LangChain-only beginners

---

## Technologies

- Python
- Streamlit
- Pandas
- python-dateutil

---

## Output Example

| Rank | Candidate | Score |
|------|-----------|------|
|1|abc123|0.842|
|2|xyz765|0.811|