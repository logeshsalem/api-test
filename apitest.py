import streamlit as st
import openai
import os
from typing import List, Dict

st.set_page_config(page_title="OpenAI Streamlit Chat", page_icon="🤖", layout="wide")

st.title("🤖 Streamlit OpenAI Chat (Streaming)")
st.caption("Enter your OpenAI API key (or set OPENAI_API_KEY in env) and start chatting. Responses stream as they arrive.")

# --- Helpers -----------------------------------------------------------------

def get_api_key() -> str:
    # prefer environment variable, otherwise allow user input
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    return st.session_state.get("api_key_input", "")


def openai_client_setup(api_key: str):
    openai.api_key = api_key


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
    # store as list of dicts: {"role": "user"/"assistant"/"system", "content": str}
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if "placeholder_id" not in st.session_state:
    st.session_state["placeholder_id"] = None

# Setup client
api_key = get_api_key()
if api_key:
    openai_client_setup(api_key)
else:
    st.warning("No API key found. Please paste your OpenAI key in the sidebar or set OPENAI_API_KEY environment variable.")

# --- Chat UI ----------------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "system":
                # don't display system messages in chat window
                continue
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

    # input area
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
    st.experimental_rerun()

# --- Send message & stream response -----------------------------------------
if send and user_input.strip() != "":
    # append user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # prepare assistant placeholder for streaming
    placeholder = st.empty()
    assistant_content = ""
    st.session_state["messages"].append({"role": "assistant", "content": assistant_content})

    # stream response from OpenAI
    try:
        # Use ChatCompletion streaming interface
        # NOTE: this code uses the classic openai.ChatCompletion.stream generator
        resp_stream = openai.ChatCompletion.create(
            model=model,
            messages=st.session_state["messages"],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in resp_stream:
            # each chunk is a dict-like object; extract delta
            if not chunk:
                continue
            # older openai library formats: chunk['choices'][0]['delta'].get('content', '')
            delta = ""
            try:
                delta_obj = chunk["choices"][0]["delta"]
                delta = delta_obj.get("content") or ""
            except Exception:
                # some transports return a text field
                delta = chunk.get("text", "")

            if delta:
                assistant_content += delta
                # replace last assistant message in session
                st.session_state["messages"][-1]["content"] = assistant_content
                # update on screen (render markdown to preserve newlines)
                placeholder.markdown(f"**Assistant:** {assistant_content}")

    except Exception as e:
        placeholder.markdown(f"**Assistant:** Error generating response: {e}")
        st.session_state["messages"][-1]["content"] = f"Error: {e}"

    # clear input box
    st.session_state["user_input"] = ""
    st.experimental_rerun()

# small friendly footer
st.markdown("---")
st.caption("Built with Streamlit + OpenAI. Make sure your key has permission for the selected model.")
