import streamlit as st
from openai import OpenAI, APIError

# --- Configuration ---
# Use the modern gpt-3.5-turbo model for testing
TEST_MODEL = "gpt-3.5-turbo" 

def test_openai_key(api_key: str, prompt: str):
    """
    Attempts to initialize the OpenAI client and make a test API call.
    Returns (success_status, message).
    """
    if not api_key:
        return False, "Please enter your OpenAI API Key."
    
    if not prompt:
        return False, "Please enter a test prompt."

    try:
        # 1. Initialize the client with the provided key
        # The base URL is explicitly set to ensure robustness, though usually optional.
        client = OpenAI(
            api_key=api_key,
        )
        
        # 2. Make a small, low-cost API call
        st.info(f"Attempting to connect to OpenAI and test with model: **{TEST_MODEL}**...")
        
        # The 'stream=True' is used here to demonstrate real-time output, but 
        # using client.chat.completions.create() without streaming is often simpler for a single test.
        # Let's use a non-streaming call for robustness in testing the key itself.
        
        response = client.chat.completions.create(
            model=TEST_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise API key tester."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50, # Keep the cost low
            temperature=0.1
        )
        
        # 3. Success check
        generated_text = response.choices[0].message.content.strip()
        
        # Check if text was returned and the key is functional
        if generated_text:
            return (
                True, 
                f"**SUCCESS!** The API key is valid and the `{TEST_MODEL}` model responded.\n\n"
                f"**Test Response:**\n\n"
                f"{generated_text}"
            )
        else:
            return (
                True, 
                "**SUCCESS!** The API key is valid, but the model returned no content (this is rare, but usually indicates the key works)."
            )

    except APIError as e:
        # Handle specific OpenAI errors (e.g., Invalid API Key, Rate Limit, Model not found)
        error_type = type(e).__name__
        if error_type == "AuthenticationError":
            return False, f"**ERROR:** Authentication Failed (Status: {e.status_code}). The API key is likely **invalid** or incorrectly formatted. Check for typos or leading/trailing spaces."
        elif error_type == "RateLimitError":
            return False, f"**ERROR:** Rate Limit Exceeded. The key is likely **valid**, but you are sending too many requests too quickly, or have exceeded your quota."
        else:
            return False, f"**ERROR:** An OpenAI API Error occurred: {error_type}. Message: {e}"
    
    except Exception as e:
        # Handle other exceptions (e.g., network issues, library initialization problems)
        return False, f"**UNEXPECTED ERROR:** Could not complete the request. Check your network or permissions. Error: {e}"


# --- Streamlit UI Layout ---

st.set_page_config(
    page_title="OpenAI Key Validator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔑 OpenAI API Key Validator")
st.markdown(
    """
    Enter your OpenAI API key and a test prompt below. This app will attempt a simple, 
    low-cost call to the `gpt-3.5-turbo` model to verify the key's validity and functionality.
    """
)

# 1. API Key Input (Sidebar is great for credentials)
st.sidebar.header("API Key Input")
api_key_input = st.sidebar.text_input(
    "Your OpenAI API Key",
    type="password",
    help="Your key starts with 'sk-'. It is not stored and is only used for the test request."
)

st.sidebar.markdown(
    """
    ---
    ### How to Run:
    1. Save the code as `key_tester.py`.
    2. Run in your terminal: `streamlit run key_tester.py`
    """
)


# 2. Test Prompt Input
st.header("Test Query")
prompt_input = st.text_area(
    "Enter a short test prompt:",
    value="Write a single, encouraging sentence about testing your code.",
    height=100,
    help="The prompt used for the test API call."
)

# 3. Test Button
if st.button("🚀 Test API Key"):
    # Clear previous messages
    st.session_state.result_placeholder = None 
    
    # Run the test
    success, message = test_openai_key(api_key_input, prompt_input)
    
    # Display the result
    if success:
        st.success(message)
    else:
        st.error(message)

# Initialize a placeholder for the result display area
if 'result_placeholder' not in st.session_state:
    st.session_state.result_placeholder = st.empty()

st.markdown("---")
st.caption("Powered by Streamlit and OpenAI.")
