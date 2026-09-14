from pathlib import Path
import pandas as pd
import requests

# Percorsi
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Dataset ufficiale ISTAT - elenco comuni italiani
URL = (
    "https://www.istat.it/storage/codici-unita-amministrative/"
    "Elenco-comuni-italiani.csv"
)

OUTPUT_RAW = RAW_DIR / "elenco_comuni_istat.csv"
OUTPUT_CLEAN = INTERIM_DIR / "municipalities_base.csv"


def download_file():
    print("Downloading ISTAT municipalities...")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    OUTPUT_RAW.write_bytes(response.content)

    print(f"Saved: {OUTPUT_RAW}")


def prepare_file():
    print("Reading ISTAT file...")

    # Il CSV ISTAT usa ; come separatore
    df = pd.read_csv(
    OUTPUT_RAW,
    sep=";",
    encoding="latin1",
    dtype=str,
    keep_default_na=False
    )

    print("\nColumns found:")
    for col in df.columns:
        print("-", col)

    print("\nRows:", len(df))
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    # Per ora salviamo la versione completa senza modificarla
    df.to_csv(
        OUTPUT_CLEAN,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved: {OUTPUT_CLEAN}")


if __name__ == "__main__":
    download_file()
    prepare_file()