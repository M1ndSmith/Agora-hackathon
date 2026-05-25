"""
Step 1 of Circle setup — register your entity secret with Circle.

Run this ONCE before creating a wallet. After success, the recovery file
(circle_recovery_*.dat) saved here is your backup — keep it safe.

Usage:
    python -m onchain.circle_setup.register_secret
    # or called automatically by: python main.py circle-init
"""
from circle.web3 import utils
from dotenv import load_dotenv
import os

load_dotenv()


def run() -> bool:
    """Register entity secret. Returns True on success."""
    api_key = os.getenv("CIRCLE_API_KEY")
    entity_secret = os.getenv("CIRCLE_ENTITY_SECRET")

    if not api_key or not entity_secret:
        print("ERROR: CIRCLE_API_KEY or CIRCLE_ENTITY_SECRET missing from .env")
        return False

    print("Registering entity secret with Circle...")
    print(f"  API key : {api_key[:24]}...")
    print(f"  Secret  : {entity_secret[:8]}...{entity_secret[-4:]}")
    print()

    result = utils.register_entity_secret_ciphertext(
        api_key=api_key,
        entity_secret=entity_secret,
        recoveryFileDownloadPath=".",
    )

    print("SUCCESS — entity secret registered.")
    print("Recovery file saved in the project root.")
    return True


if __name__ == "__main__":
    run()
