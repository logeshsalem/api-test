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
    st.experimental_rerun()

# --- Send message & stream response (OpenAI >=1.0.0) -------------------------
if send and user_input.strip() != "" and client is not None:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    placeholder = st.empty()
    assistant_content = ""
    st.session_state["messages"].append({"role": "assistant", "content": assistant_content})

    try:
        with client.chat.completions.stream(
            model=model,
            messages=st.session_state["messages"],
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            for event in stream:
                if event.type == "message.delta" and event.delta.get("content"):
                    delta = event.delta["content"]
                    assistant_content += delta
                    st.session_state["messages"][-1]["content"] = assistant_content
                    placeholder.markdown(f"**Assistant:** {assistant_content}")

            final = stream.get_final_message()
            if final:
                st.session_state["messages"][-1]["content"] = final.content[0].text

    except Exception as e:
        placeholder.markdown(f"**Assistant:** Error generating response: {e}")
        st.session_state["messages"][-1]["content"] = f"Error: {e}"

    st.session_state["user_input"] = ""
    st.experimental_rerun()

st.markdown("---")
st.caption("Built with Streamlit + OpenAI SDK >= 1.0.0. Make sure your key has permission for the selected model.")
