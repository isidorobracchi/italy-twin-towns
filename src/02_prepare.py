from pathlib import Path
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "interim" / "municipalities_base.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "municipalities_master.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD
# --------------------------------------------------

print("Loading municipalities base file...")

df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    keep_default_na=False
)

print(f"Rows loaded: {len(df):,}")


# --------------------------------------------------
# CLEAN COLUMN NAMES
# --------------------------------------------------

# Alcune intestazioni ISTAT contengono \n e spazi doppi.
df.columns = (
    df.columns
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("\nCleaned column names:")
for col in df.columns:
    print("-", col)


# --------------------------------------------------
# SELECT VARIABLES
# --------------------------------------------------

columns = {
    "Codice Comune formato alfanumerico": "istat_code",
    "Denominazione in italiano": "name",
    "Codice Regione": "region_code",
    "Denominazione Regione": "region",
    "Codice Provincia (Storico)(1)": "province_code",
    "Denominazione dell'Unità territoriale sovracomunale (valida a fini statistici)": "province",
    "Sigla automobilistica": "province_abbr",
    "Codice Ripartizione Geografica": "macro_area_code",
    "Ripartizione geografica": "macro_area",
    "Codice Catastale del comune": "cadastral_code",
    "Codice NUTS1 2024": "nuts1",
    "Codice NUTS2 2024 (3)": "nuts2",
    "Codice NUTS3 2024": "nuts3",
}

missing_columns = [
    col for col in columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR - Missing columns:")
    for col in missing_columns:
        print("-", col)

    raise KeyError("One or more expected ISTAT columns were not found.")


df = df[list(columns.keys())].rename(columns=columns)


# --------------------------------------------------
# BASIC CLEANING
# --------------------------------------------------

# Manteniamo i codici come stringhe per non perdere gli zeri iniziali.
df["istat_code"] = df["istat_code"].str.zfill(6)
df["region_code"] = df["region_code"].str.zfill(2)
df["province_code"] = df["province_code"].str.zfill(3)

# Rimuove eventuali spazi accidentali.
text_columns = [
    "name",
    "region",
    "province",
    "province_abbr",
    "macro_area",
]

for col in text_columns:
    df[col] = df[col].str.strip()


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\n--- VALIDATION ---")

print(f"Municipalities: {len(df):,}")
print(f"Unique ISTAT codes: {df['istat_code'].nunique():,}")

duplicates = df[df["istat_code"].duplicated(keep=False)]

print(f"Duplicated ISTAT codes: {len(duplicates)}")

missing_codes = (
    df["istat_code"].isna() |
    df["istat_code"].str.strip().eq("")
).sum()

missing_names = (
    df["name"].isna() |
    df["name"].str.strip().eq("")
).sum()

print(f"Missing ISTAT codes: {missing_codes}")
print(f"Missing municipality names: {missing_names}")


# --------------------------------------------------
# CHECK GIARRE
# --------------------------------------------------

giarre = df[df["name"].str.casefold() == "giarre"]

print("\n--- GIARRE TEST ---")

if len(giarre) == 0:
    print("Giarre not found!")
else:
    print(giarre.to_string(index=False))


# --------------------------------------------------
# SORT
# --------------------------------------------------

df = df.sort_values(
    ["region", "province", "name"]
).reset_index(drop=True)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print(f"\nSaved clean master file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))