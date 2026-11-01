import os

import pandas as pd
import streamlit as st

from discussion import ACTIONS, AGENTS, PRESETS, Step, run_discussion
from providers import (
    DEFAULT_CLAUDE_MODELS,
    DEFAULT_OPENAI_MODELS,
    ChatGPT,
    Claude,
)

st.set_page_config(page_title="ChatGPT × Claude Discussion", page_icon="🤝", layout="wide")

AVATARS = {"ChatGPT": "🟢", "Claude": "🟠"}
ACTION_LABEL = {"Answer": "answers", "Improve": "improves the answer", "Review": "reviews & scores"}


def secret(name: str) -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml
        pass
    return os.environ.get(name, "")


ss = st.session_state
ss.setdefault("openai_key", secret("OPENAI_API_KEY"))
ss.setdefault("anthropic_key", secret("ANTHROPIC_API_KEY"))
ss.setdefault("openai_models", None)
ss.setdefault("claude_models", None)
ss.setdefault("history", [])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_models(provider: str, key: str) -> list[str]:
    cls = ChatGPT if provider == "openai" else Claude
    return cls(key).list_models()


def connect(provider: str):
    key = ss.openai_key if provider == "openai" else ss.anthropic_key
    if not key:
        return
    try:
        models = fetch_models(provider, key)
        ss[f"{'openai' if provider == 'openai' else 'claude'}_models"] = models or None
        ss[f"{provider}_error"] = None
    except Exception as e:
        ss[f"{'openai' if provider == 'openai' else 'claude'}_models"] = None
        ss[f"{provider}_error"] = str(e)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🔐 Log in")
    st.caption("Paste API keys. They stay in this browser session only. "
               "Or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in Streamlit secrets.")

    st.text_input("OpenAI API key (ChatGPT)", type="password", key="openai_key",
                  on_change=connect, args=("openai",))
    st.text_input("Anthropic API key (Claude)", type="password", key="anthropic_key",
                  on_change=connect, args=("anthropic",))

    if ss.openai_key and ss.openai_models is None and not ss.get("openai_error"):
        connect("openai")
    if ss.anthropic_key and ss.claude_models is None and not ss.get("anthropic_error"):
        connect("anthropic")

    for label, key_name, models, err in [
        ("ChatGPT", "openai_key", ss.openai_models, ss.get("openai_error")),
        ("Claude", "anthropic_key", ss.claude_models, ss.get("anthropic_error")),
    ]:
        if not ss[key_name]:
            st.info(f"{AVATARS[label]} {label}: not logged in")
        elif err:
            st.error(f"{AVATARS[label]} {label}: {err[:200]}")
        else:
            st.success(f"{AVATARS[label]} {label}: connected ({len(models or [])} models)")

    st.divider()
    st.header("🧠 Models")
    oa_options = ss.openai_models or DEFAULT_OPENAI_MODELS
    cl_options = ss.claude_models or DEFAULT_CLAUDE_MODELS
    openai_model = st.selectbox(
        "ChatGPT model", oa_options,
        index=oa_options.index("gpt-5") if "gpt-5" in oa_options else 0,
    )
    claude_model = st.selectbox(
        "Claude model", cl_options,
        index=cl_options.index("claude-opus-5-5") if "claude-opus-5-5" in cl_options else 0,
    )
    with st.expander("Custom model IDs / Claude options"):
        openai_model = st.text_input("Override ChatGPT model ID", "") or openai_model
        claude_model = st.text_input("Override Claude model ID", "") or claude_model
        claude_effort = st.selectbox(
            "Claude effort", ["(model default)", "low", "medium", "high", "xhigh", "max"],
            help="Thinking depth for Claude 4.6+ models. Leave default for older models.",
        )
        refusal_fallback = st.checkbox(
            "Claude refusal fallback", value=True,
            help="If a newer Claude model declines a request, the API retries it on "
                 "another Claude model automatically.",
        )

    st.divider()
    st.header("🔁 Stopping rule")
    threshold = st.slider("Stop when review score ≥", 1.0, 10.0, 9.0, 0.5)
    max_rounds = st.slider("Max rounds (safety cap)", 1, 10, 4)
    require_all = st.checkbox(
        "Require every reviewer in the loop to approve", value=False,
        help="In flows where both AIs review, only stop once both latest scores clear the threshold.",
    )

# --------------------------------------------------------------------------- main
st.title("🤝 ChatGPT × Claude")
st.caption("Ask a question. The two AIs answer, critique, score and refine each other's "
           "work until the reviewer's score clears your threshold.")

st.subheader("Flow")
preset = st.selectbox("Discussion flow", list(PRESETS) + ["Custom…"])

if preset == "Custom…":
    st.caption("**Opening** steps run once. **Loop** steps repeat until a Review "
               "score clears the threshold or max rounds is reached.")
    col_cfg = {
        "Agent": st.column_config.SelectboxColumn("Agent", options=AGENTS, required=True),
        "Action": st.column_config.SelectboxColumn("Action", options=ACTIONS, required=True),
    }
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Opening**")
        op_df = st.data_editor(
            pd.DataFrame([{"Agent": "ChatGPT", "Action": "Answer"},
                          {"Agent": "Claude", "Action": "Improve"}]),
            column_config=col_cfg, num_rows="dynamic", key="opening_editor",
            use_container_width=True,
        )
    with c2:
        st.markdown("**Loop ↺**")
        lp_df = st.data_editor(
            pd.DataFrame([{"Agent": "ChatGPT", "Action": "Review"},
                          {"Agent": "Claude", "Action": "Improve"}]),
            column_config=col_cfg, num_rows="dynamic", key="loop_editor",
            use_container_width=True,
        )
    opening = [(r.Agent, r.Action) for r in op_df.dropna().itertuples()]
    loop = [(r.Agent, r.Action) for r in lp_df.dropna().itertuples()]
else:
    opening, loop = PRESETS[preset]["opening"], PRESETS[preset]["loop"]

flow_txt = " → ".join(f"{AVATARS[a]} {a}: {act}" for a, act in opening)
loop_txt = " → ".join(f"{AVATARS[a]} {a}: {act}" for a, act in loop)
st.markdown(f"{flow_txt}  \n↺ **repeat:** {loop_txt or '—'}")


def step_header(agent: str, action: str, rnd: int) -> str:
    tag = "opening" if rnd == 0 else f"round {rnd}"
    return f"**{agent}** {ACTION_LABEL[action]} · _{tag}_"


def render_step(step: Step):
    with st.chat_message(step.agent, avatar=AVATARS[step.agent]):
        st.markdown(step_header(step.agent, step.action, step.round))
        if step.action == "Review":
            render_review(step)
        else:
            st.markdown(step.content)


def render_review(step: Step):
    score = "?" if step.score is None else f"{step.score:g}/10"
    st.metric("Score", score)
    if step.verdict:
        st.markdown(f"_{step.verdict}_")
    if step.issues:
        st.markdown("\n".join(f"- {i}" for i in step.issues))
    if step.score is None:
        st.markdown(step.content)


def render_summary(run: dict):
    reviews = [s for s in run["steps"] if s.action == "Review" and s.score is not None]
    scores = [(str(i), s.agent, s.score) for i, s in enumerate(reviews, 1)]
    st.success(f"**Stopped:** {run['reason']}")
    if len(scores) > 1:
        df = pd.DataFrame(scores, columns=["review #", "reviewer", "score"])
        st.line_chart(df, x="review #", y="score", color="reviewer", height=220)
    st.markdown(f"### ✅ Final answer (last written by {run['author']})")
    with st.container(border=True):
        st.markdown(run["answer"])
    st.download_button("Download transcript (.md)", transcript_md(run),
                       file_name="discussion.md", key=f"dl_{run['id']}")


def transcript_md(run: dict) -> str:
    out = [f"# {run['question']}\n", f"_Models: ChatGPT={run['models'][0]}, "
           f"Claude={run['models'][1]}_\n"]
    for s in run["steps"]:
        out.append(f"## {s.agent} — {s.action} (round {s.round})\n")
        if s.action == "Review":
            out.append(f"**Score:** {s.score}\n\n{s.verdict}\n")
            out += [f"- {i}" for i in s.issues]
            out.append("")
        else:
            out.append(s.content + "\n")
    out.append(f"## Final answer\n\n{run['answer']}\n\n_{run['reason']}_")
    return "\n".join(out)


# Previous runs
for run in ss.history:
    st.divider()
    with st.chat_message("user"):
        st.markdown(run["question"])
    with st.expander(f"Discussion ({len(run['steps'])} steps)", expanded=False):
        for s in run["steps"]:
            render_step(s)
    render_summary(run)

question = st.chat_input("Ask both AIs a question…")

if question:
    missing = [n for n, k in [("ChatGPT", ss.openai_key), ("Claude", ss.anthropic_key)]
               if not k and any(a == n for a, _ in opening + loop)]
    if missing:
        st.error(f"Log in to {' and '.join(missing)} in the sidebar first.")
        st.stop()
    if not opening or opening[0][1] == "Review":
        st.error("The flow must start with an Answer step.")
        st.stop()

    agents = {
        "ChatGPT": ChatGPT(ss.openai_key or "missing", openai_model),
        "Claude": Claude(
            ss.anthropic_key or "missing", claude_model,
            effort=None if claude_effort == "(model default)" else claude_effort,
            refusal_fallback=refusal_fallback,
        ),
    }

    st.divider()
    with st.chat_message("user"):
        st.markdown(question)

    current = {}

    def on_step_start(agent, action, rnd):
        current["msg"] = st.chat_message(agent, avatar=AVATARS[agent])
        current["msg"].markdown(step_header(agent, action, rnd))
        current["action"] = action

    def on_stream(chunks):
        with current["msg"]:
            if current["action"] == "Review":
                with st.spinner("Reviewing…"):
                    return "".join(chunks)
            return st.write_stream(chunks)

    def on_step_done(step: Step):
        if step.action == "Review":
            with current["msg"]:
                render_review(step)

    try:
        result = run_discussion(
            question, agents, opening, loop, threshold, max_rounds, require_all,
            on_step_start, on_stream, on_step_done,
        )
    except Exception as e:  # provider/API errors, bad flow config
        st.error(str(e))
        st.stop()

    run = {
        "id": len(ss.history),
        "question": question,
        "steps": result.steps,
        "answer": result.final_answer,
        "author": result.final_author,
        "reason": result.stopped_reason,
        "models": (openai_model, claude_model),
    }
    render_summary(run)
    ss.history.append(run)
