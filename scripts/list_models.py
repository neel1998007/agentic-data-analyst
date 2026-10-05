"""
scripts/list_models.py
----------------------
Throwaway utility: prints all model IDs available to your Anthropic API key.

Run from the project root:
    python -m scripts.list_models
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY is not set in your .env file.")

client = anthropic.Anthropic(api_key=api_key)

models = client.models.list()

print(f"{'ID':<45} {'Display Name'}")
print("-" * 80)
for m in models.data:
    print(f"{m.id:<45} {getattr(m, 'display_name', '')}")
