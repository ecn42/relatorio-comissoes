import streamlit as st

st.set_page_config(page_title="Last Token Extractor", page_icon="🔎")

st.title("Last Token Extractor")
st.caption("Enter one string per line. We'll return the substring after the last space.")

default_text = "Alpha Bravo Charlie\nHello World\nsingleword\ntrailing space "
user_input = st.text_area(
    "Strings (one per line):",
    value=default_text,
    height=200,
    help="Each line is treated as a separate string.",
)

strip_mode = st.selectbox(
    "Whitespace handling",
    options=["strip (trim ends)", "preserve"],
    index=0,
    help="Choose whether to trim leading/trailing whitespace before extracting.",
)

def last_token(s: str) -> str:
    # Optionally strip ends to avoid empty trailing tokens from spaces at the end
    if strip_mode.startswith("strip"):
        s = s.strip()
    # Find last space; if none, the whole string is the last token
    idx = s.rfind(" ")
    return s[idx + 1 :] if idx != -1 else s

if st.button("Extract"):
    lines = [line for line in user_input.splitlines()]
    results = [last_token(line) for line in lines]

    st.subheader("Results")
    # Show side-by-side original and extracted token
    for original, token in zip(lines, results):
        st.write(f"• '{original}' → '{token}'")

    # Downloadable CSV
    import csv
    from io import StringIO

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["original", "last_token"])
    for o, t in zip(lines, results):
        writer.writerow([o, t])
    st.download_button(
        "Download CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name="last_tokens.csv",
        mime="text/csv",
    )