#!/usr/bin/env python3
"""
Convert a downloaded Google service-account JSON key into the [gsheets]
TOML block for Streamlit secrets.

Usage:
    python3 json_key_to_toml.py ~/Downloads/my-key.json SHEET_KEY

Paste the output into the Streamlit Cloud app's Settings -> Secrets
(and/or into .streamlit/secrets.toml for local testing).
"""
import json
import sys

if len(sys.argv) != 3:
    sys.exit(__doc__)

key = json.load(open(sys.argv[1]))
print(f'[gsheets]\nsheet_key = "{sys.argv[2]}"\n\n[gsheets.service_account]')
for k, v in key.items():
    v = str(v).replace("\n", "\\n")
    print(f'{k} = "{v}"')
