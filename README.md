# C3 — Introduction to Agents

An agent-powered business assistant for **CedarFlow Studio**, a fictional workflow automation business created for the EECE 503P/798S assignment. Its assistant, **Nour**, answers business questions, collects leads, and logs feedback or unanswered questions.

## Setup

Use Python 3.11 (developed with 3.11.5). From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally to set `OPENAI_API_KEY`. `OPENAI_MODEL` defaults to `gpt-5.4-mini`; the configured model must be available to your API account. Never upload `.env` or an API key.

Open `business_agent.ipynb` in Jupyter or VS Code and select the virtual environment as its Python kernel. If the environment is not available as a kernel, install `ipykernel` in it:

```bash
python -m pip install ipykernel
```

Run the notebook from the repository root so it can locate the `me/` folder. Run the cells in order, then use the final build and launch cells to start Gradio. Open the local URL shown in the output. Public-link sharing is disabled by default.

## Implemented tasks

1. Fictional business identity in `me/about_business.pdf` and `me/business_summary.txt`.
2. Separately labeled persona and operational instructions.
3. A simple router for general questions, lead collection, and feedback/unanswered questions.
4. `record_customer_interest(email, name, message)` and `record_feedback(question)`, with MCP-style schema documentation and JSON print logs.
5. A system prompt grounded in both business documents.
6. An OpenAI interaction loop with conversation history and model-requested tool handling.
7. A Gradio `ChatInterface` with separate session memory and clear/retry support.
8. A reflection pass that checks each draft and logs the original and revised replies.

Only the reviewed final reply enters customer-facing chat history. Reflection adds one model completion per successful turn; pass `reflect=False` to `chat()` to disable it. Tool records and reflection logs appear in notebook/server output. They are printed logs, not a persistent customer database.

`app.py` is an optional deployment placeholder; run the notebook to use this implementation. Hugging Face Spaces deployment is not included.

## Verification and demonstration

Run the offline tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The notebook includes clearly labeled scripted demonstrations that require no API key. These exercise the code path, but do not establish live model behavior. Live verification remains outstanding.

To verify the real model, configure `.env`, set `RUN_LIVE_DEMO = True`, and run the live demonstration cell. API usage charges apply. Confirm that the model calls both tools and that reflection logs are printed. Leave the flag disabled for ordinary offline setup runs.

For the submission video, demonstrate:

- A business question answered from the references.
- A lead conversation that collects missing contact details and prints a successful lead log.
- An unanswered question or feedback message that prints a feedback log.
- Follow-up memory in Gradio, plus clearing the conversation.
- Reflection logs showing the draft and reviewed reply.

A submission video has not yet been added. The assignment handout is included as `C3 - Introduction to Agents.pdf`.

## Project files

```text
me/
  about_business.pdf
  business_summary.txt
business_agent.ipynb
tests/
requirements.txt
.env.example
app.py
```

`.env`, `.venv`, Python caches, and notebook checkpoints are excluded from Git.
