
import requests
import re
import zipfile
import io
import pandas as pd
from urllib.parse import urljoin

BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/LAMINA/DADOS/"

def get_latest_zip_url():
    print(f"Fetching index from {BASE_URL}...")
    headers = {"User-Agent": "Streamlit-LaminaFI/1.0"}
    r = requests.get(BASE_URL, timeout=30, headers=headers)
    r.raise_for_status()

    matches = re.findall(r"lamina_fi_(\d{6})\.zip", r.text, flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError("No zip files found.")
    
    yyyymms = sorted(set(matches))
    latest = yyyymms[-1]
    zip_name = f"lamina_fi_{latest}.zip"
    print(f"Latest file identified: {zip_name}")
    return urljoin(BASE_URL, zip_name), latest

def inspect_zip():
    zip_url, _ = get_latest_zip_url()
    print(f"Downloading {zip_url}...")
    headers = {"User-Agent": "Streamlit-LaminaFI/1.0"}
    resp = requests.get(zip_url, timeout=180, headers=headers)
    resp.raise_for_status()
    print(f"Downloaded {len(resp.content)} bytes.")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        print(f"Files in zip: {names}")
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        
        for name in csv_names:
            print(f"--- Inspecting {name} ---")
            with zf.open(name) as f:
                head = f.read(1000)
                print(f"First 1000 bytes (repr): {head!r}")
                try:
                    print(f"First 1000 bytes (decoded latin1): {head.decode('latin1')}")
                except Exception as e:
                    print(f"Decode latin1 failed: {e}")
                
            print("--- Attempting pandas read ---")
            with zf.open(name) as f:
                content = f.read()
            
            try:
                df = pd.read_csv(io.BytesIO(content), sep=";", decimal=",", encoding="latin1", nrows=5)
                print("Read successfully with latin1, sep=;")
                print(df.columns)
            except Exception as e:
                print(f"Failed read with latin1: {e}")

            try:
                df = pd.read_csv(io.BytesIO(content), sep=";", decimal=",", encoding="cp1252", nrows=5)
                print("Read successfully with cp1252, sep=;")
            except Exception as e:
                print(f"Failed read with cp1252: {e}")

if __name__ == "__main__":
    inspect_zip()
