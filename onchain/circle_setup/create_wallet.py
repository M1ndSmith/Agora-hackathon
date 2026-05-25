"""
Step 2 of Circle setup — create a wallet set and wallet on Arc testnet.

Run AFTER register_secret.py has succeeded.

Usage:
    python -m onchain.circle_setup.create_wallet
    # or called automatically by: python main.py circle-init
"""
import json
import os

from circle.web3 import developer_controlled_wallets, utils
from dotenv import load_dotenv

load_dotenv()


def run() -> dict:
    """
    Create a Circle wallet set + wallet on ARC-TESTNET.

    Returns dict with wallet_set_id, wallet_id, address.
    Raises on any API error.
    """
    api_key = os.getenv("CIRCLE_API_KEY")
    entity_secret = os.getenv("CIRCLE_ENTITY_SECRET")

    if not api_key or not entity_secret:
        raise RuntimeError("CIRCLE_API_KEY or CIRCLE_ENTITY_SECRET missing from .env")

    client = utils.init_developer_controlled_wallets_client(
        api_key=api_key,
        entity_secret=entity_secret,
    )

    wallet_sets_api = developer_controlled_wallets.WalletSetsApi(client)
    wallets_api = developer_controlled_wallets.WalletsApi(client)

    print("Creating wallet set...")
    wallet_set = wallet_sets_api.create_wallet_set(
        developer_controlled_wallets.CreateWalletSetRequest.from_dict(
            {"name": "Agora-wallet-set"}
        )
    )
    wallet_set_id = wallet_set.data.wallet_set.actual_instance.id
    print(f"  Wallet set ID : {wallet_set_id}")

    print("Creating wallet on ARC-TESTNET...")
    wallet = wallets_api.create_wallet(
        developer_controlled_wallets.CreateWalletRequest.from_dict(
            {
                "walletSetId": wallet_set_id,
                "blockchains": ["ARC-TESTNET"],
                "count": 1,
                "accountType": "EOA",
            }
        )
    )

    wallet_data = wallet.data.wallets[0].actual_instance
    wallet_id = wallet_data.id
    address = wallet_data.address

    print(f"  Wallet ID     : {wallet_id}")
    print(f"  Address       : {address}")
    print()

    return {
        "wallet_set_id": wallet_set_id,
        "wallet_id": wallet_id,
        "address": address,
    }


if __name__ == "__main__":
    try:
        result = run()
        print("Add these to your .env:")
        print(f"  CIRCLE_WALLET_SET_ID={result['wallet_set_id']}")
        print(f"  CIRCLE_WALLET_ID={result['wallet_id']}")
        print(f"  AGENT_ADDRESS={result['address']}")
    except developer_controlled_wallets.ApiException as e:
        print(f"Circle API error: {e}")
