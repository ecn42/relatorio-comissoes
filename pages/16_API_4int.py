import streamlit as st
import requests
import pandas as pd

# --- Configuration ---
API_URL = "https://apis.4intelligence.ai/api-feature-store/api/v1/indicators"

def fetch_indicators(api_key):
    # 1. SANITIZE INPUT: Remove accidental spaces or newlines
    clean_key = api_key.strip()
    
    # 2. HANDLE PREFIX: If the user pasted "Bearer xyz...", don't add "Bearer" again.
    if clean_key.lower().startswith("bearer"):
        auth_value = clean_key
    else:
        auth_value = f"Bearer {clean_key}"

    headers = {
        "Authorization": auth_value,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(API_URL, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP Error: {err}")
        # If it's a 431, we give a specific hint
        if response.status_code == 431:
            st.warning("Hint: Your API Key might be too long or contain spaces. Make sure you aren't pasting the entire JSON login response.")
        return None
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None

def main():
    st.set_page_config(layout="wide", page_title="4i Data Viewer")
    st.title("4Intelligence Feature Store")

    st.markdown("Enter your token below. It usually starts with `ey...` or similar characters.")
    
    # Use a text area in case the token is very long, to verify it looks right
    api_key = st.text_input("API Key / Token", type="password")

    if st.button("List Indicators"):
        if not api_key:
            st.warning("Please enter your API Key.")
            return

        with st.spinner("Fetching indicators..."):
            data = fetch_indicators(api_key)

            if data:
                # --- DATA PARSING LOGIC ---
                # Logic: The API returns metadata (page: 0, total: 4720) + a list of items.
                # We need to find that list.
                
                items_list = []

                # Case 1: It's already a list
                if isinstance(data, list):
                    items_list = data
                
                # Case 2: It's a dictionary (pagination wrapper)
                elif isinstance(data, dict):
                    # Look for the standard keys 4i uses
                    if "items" in data:
                        items_list = data["items"]
                    elif "content" in data:
                        items_list = data["content"]
                    elif "data" in data:
                        items_list = data["data"]
                    else:
                        # Last resort: find the first key that is a list
                        for k, v in data.items():
                            if isinstance(v, list):
                                items_list = v
                                break

                if items_list:
                    df = pd.DataFrame(items_list)
                    st.success(f"Loaded {len(df)} indicators successfully.")
                    st.dataframe(
                        df, 
                        use_container_width=True,
                        column_config={
                            "id": "ID",
                            "name": "Name",
                            "description": "Description",
                            "source": "Source"
                        }
                    )
                else:
                    st.warning("Connected, but couldn't find the list of indicators in the response.")
                    st.write("Response keys found:", list(data.keys()))
                    st.json(data)

if __name__ == "__main__":
    main()