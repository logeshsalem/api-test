import streamlit as st
import os
from openai import OpenAI

st.set_page_config(page_title="OpenAI Streamlit Chat", page_icon="🤖", layout="wide")

st.title("🤖 Streamlit OpenAI Chat (Streaming)")
st.caption("Enter your OpenAI API key (or set OPENAI_API_KEY in env) and start chatting. Responses stream as they arrive.")

# --- Helpers -----------------------------------------------------------------

def get_api_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    return st.session_state.get("api_key_input", "")


# --- Sidebar / Settings -----------------------------------------------------
with st.sidebar.form("settings"):
    st.write("### Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password", key="api_key_input")
    model = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"], index=3)
    max_tokens = st.number_input("Max tokens (response)", min_value=50, max_value=4096, value=512, step=50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    submit_settings = st.form_submit_button("Save")

# initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Setup OpenAI client
api_key = get_api_key()
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("No API key found. Please paste your OpenAI key in the sidebar or set OPENAI_API_KEY environment variable.")
    client = None

# --- Chat UI ----------------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "system":
                continue
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

    user_input = st.text_area("", placeholder="Type your message here and press Ctrl+Enter (or click Send)", key="user_input", height=120)
    send = st.button("Send")
    clear = st.button("Clear chat")

with col2:
    st.write("### Controls")
    if st.button("Export chat as JSON"):
        import json
        st.download_button("Download chat JSON", data=json.dumps(st.session_state["messages"], indent=2), file_name="chat_history.json")
    st.write("---")
    st.write("Current model: ", model)
    st.write("Max tokens: ", max_tokens)
    st.write("Temperature: ", temperature)

if clear:
    st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]
    # Clear the input box and let Streamlit naturally re-run on the next interaction
    try:
        st.session_state["user_input"] = ""
    except Exception:
        pass

# --- Send message & get response (compatible and robust) -------------------
if send and user_input.strip() != "" and client is not None:
    # append user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    placeholder = st.empty()
    assistant_content = ""
    st.session_state["messages"].append({"role": "assistant", "content": assistant_content})

    try:
        # Attempt streaming first (best-effort). If streaming fails or yields nothing,
        # fall back to a single non-streaming request so the user always gets a reply.
        streamed = False
        try:
            with client.chat.completions.stream(
                model=model,
                messages=st.session_state["messages"],
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream:
                for event in stream:
                    # try multiple ways to extract delta text
                    delta = None
                    try:
                        if hasattr(event, "delta") and isinstance(event.delta, dict):
                            delta = event.delta.get("content")
                    except Exception:
                        delta = None

                    if delta is None:
                        try:
                            if hasattr(event, "get"):
                                choices = event.get("choices")
                                if choices:
                                    delta = choices[0].get("delta", {}).get("content")
                        except Exception:
                            delta = None

                    if delta is None:
                        try:
                            delta = getattr(event, "text", None)
                        except Exception:
                            delta = None

                    if delta:
                        assistant_content += delta
                        st.session_state["messages"][-1]["content"] = assistant_content
                        placeholder.markdown(f"**Assistant:** {assistant_content}")
                        streamed = True
        except Exception:
            streamed = False

        # If streaming didn't produce content, use a non-streaming call as a fallback.
        if not streamed:
            resp = client.chat.completions.create(
                model=model,
                messages=st.session_state["messages"],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Try various ways to extract assistant text from response object/dict
            content = None
            try:
                # 1) dict-like responses
                if hasattr(resp, "get"):
                    choices = resp.get("choices")
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        # older shape: first["message"]["content"] or first["message"]["content"][0]["text"]
                        if isinstance(first, dict):
                            msg = first.get("message") or first.get("message", {})
                            if isinstance(msg, dict):
                                # message.content might be a list of parts
                                content = msg.get("content")
                                if isinstance(content, list) and len(content) > 0:
                                    # join parts
                                    content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
                            # fallback to text
                            if not content:
                                content = first.get("text")
                # 2) object-like responses (SDK objects)
                if not content and hasattr(resp, "choices"):
                    ch = getattr(resp, "choices")
                    if ch and len(ch) > 0:
                        first = ch[0]
                        # try common attributes
                        msg = getattr(first, "message", None)
                        if msg:
                            content = getattr(msg, "content", None) or getattr(msg, "text", None)
                        if not content:
                            content = getattr(first, "text", None)
            except Exception:
                content = None

            # final fallback: stringify response
            if not content:
                try:
                    content = str(resp)
                except Exception:
                    content = "(no text could be extracted from the model response)"

            assistant_content = content
            st.session_state["messages"][-1]["content"] = assistant_content
            placeholder.markdown(f"**Assistant:** {assistant_content}")

    except Exception as e:
        placeholder.markdown(f"**Assistant:** Error generating response: {e}")
        st.session_state["messages"][-1]["content"] = f"Error: {e}"

    # clear input box (safe)
    try:
        st.session_state["user_input"] = ""
    except Exception:
        pass

# If user clicked send but client isn't configured, show an explanation
if send and client is None:
    st.warning("Cannot send message because OpenAI client is not configured. Provide an API key in the sidebar or set OPENAI_API_KEY environment variable.")

st.markdown("---")
st.caption("Built with Streamlit + OpenAI SDK >= 1.0.0. Make sure your key has permission for the selected model.")
st.caption("Built with Streamlit + OpenAI SDK >= 1.0.0. Make sure your key has permission for the selected model.")
