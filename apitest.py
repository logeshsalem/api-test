# test_openai_key.py
import os
import json
import requests
import streamlit as st

OPENAI_MODELS_ENDPOINT = "https://api.openai.com/v1/models"
TIMEOUT_SECONDS = 10

st.set_page_config(page_title="OpenAI API Key Tester", page_icon="🔑")

st.title("🔑 OpenAI API Key Tester")
st.markdown(
    "Paste an OpenAI API key below or choose to use the `OPENAI_API_KEY` environment variable. "
    "This app will attempt a safe call to `/v1/models` to verify the key."
)

use_env = st.checkbox("Use OPENAI_API_KEY environment variable (instead of pasting)", value=False)

if use_env:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.warning("Environment variable OPENAI_API_KEY not found.")
else:
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-... (never share this publicly)")

help_expand = st.expander("Why this endpoint?")
help_expand.write(
    "The `/v1/models` endpoint is a safe read-only call that returns the list of available models for the key. "
    "If the key is valid you'll get a 200 response with JSON; if not you will typically get 401 Unauthorized or 403."
)

def test_key(key: str):
    headers = {
        "Authorization": f"Bearer {key}",
        "User-Agent": "openai-key-tester/1.0"
    }
    try:
        resp = requests.get(OPENAI_MODELS_ENDPOINT, headers=headers, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Request timed out."}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Network error: {e}"}

    # Try to parse JSON safely
    content_type = resp.headers.get("Content-Type", "")
    body_preview = None
    try:
        if "application/json" in content_type:
            body_preview = resp.json()
        else:
            body_preview = resp.text[:2000]
    except Exception:
        body_preview = resp.text[:2000]

    if resp.status_code == 200:
        return {"ok": True, "status_code": resp.status_code, "body": body_preview}
    else:
        # common cases: 401 invalid, 429 rate limit, 403 forbidden
        return {"ok": False, "status_code": resp.status_code, "body": body_preview}

st.write("---")

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🔎 Test Key"):
        if not api_key:
            st.error("Please provide an API key (or set OPENAI_API_KEY and check the box).")
        else:
            with st.spinner("Testing key..."):
                result = test_key(api_key)
            if result["ok"]:
                st.success(f"✅ Key is valid — HTTP {result['status_code']}")
                st.subheader("Response preview")
                st.json(result["body"])
            else:
                code = result.get("status_code")
                if code:
                    if code == 401:
                        st.error("❌ Unauthorized (HTTP 401): The API key is invalid or revoked.")
                    elif code == 403:
                        st.error("❌ Forbidden (HTTP 403): The key may lack permissions.")
                    elif code == 429:
                        st.error("⚠️ Rate limited (HTTP 429): Too many requests for this key.")
                    else:
                        st.error(f"❌ Request failed — HTTP {code}")
                else:
                    st.error("❌ Request failed.")
                st.subheader("Server response (preview)")
                st.json(result["body"] if isinstance(result.get("body"), dict) else {"text": result.get("body")})
with col2:
    st.markdown("#### Quick help")
    st.write(
        "- Use a **test key** (not your production key you can't share).  \n"
        "- If using env var: `export OPENAI_API_KEY='sk-...'` (Linux/Mac) or set via system environment on Windows.  \n"
        "- If you get 401: confirm the key hasn't expired/revoked.  \n"
        "- If you get network errors: check internet access or that your environment blocks outbound HTTPS."
    )

st.write("---")
st.markdown("### Advanced: curl equivalent")
st.code(
    "curl https://api.openai.com/v1/models -H \"Authorization: Bearer $OPENAI_API_KEY\"",
    language="bash",
)

st.caption("This tool DOES NOT log or send your key anywhere — it runs locally in your machine when you run Streamlit.")
