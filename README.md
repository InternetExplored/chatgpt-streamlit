# ChatGPT × Claude Discussion

A Streamlit app where ChatGPT and Claude discuss your question. One AI answers,
the other critiques and scores the answer, and they revise it back and forth.
The loop stops when a review score reaches your threshold or when it hits the
max-rounds cap.

## Features

- **Log in to both**: paste an OpenAI API key and an Anthropic API key in the
  sidebar. You can also set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in
  `.streamlit/secrets.toml` or as environment variables.
- **Pick models**: the model dropdowns load the live list from each API, and you
  can type any custom model ID. You can also set Claude's effort level.
- **Choose the flow**: pick a preset such as
  *ChatGPT answers → Claude improves → ChatGPT validates ↺*, or build a
  **Custom** flow from `Answer`, `Improve` and `Review` steps. Custom flows have
  two parts: *Opening* steps run once and *Loop* steps repeat.
- **Score-based stopping**: every `Review` step returns a 0–10 score and a list
  of issues, and the next `Improve` step has to address those issues. The loop
  ends when the score is at or above your threshold, or after the max number of
  rounds. You can optionally require every reviewer to approve.
- Each turn streams live, and you get a score chart, the final answer and a
  Markdown transcript you can download.

## Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

> Note: this app uses the OpenAI and Anthropic **APIs**, so you need
> pay-per-use API keys from platform.openai.com and console.anthropic.com.
> ChatGPT Plus and Claude Pro subscriptions can't be used to sign in to the API.

## Files

- `main.py`: Streamlit UI
- `discussion.py`: flow engine, prompts, review-score parsing and stopping logic
- `providers.py`: ChatGPT and Claude API wrappers (streaming, model listing)
