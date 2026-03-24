"""
test_bluesky_tokens.py
"""

import os
from dotenv import load_dotenv
from atproto import Client

load_dotenv()

HANDLE       = os.environ["BLUESKY_HANDLE"]
APP_PASSWORD = os.environ["BLUESKY_APP_PASSWORD"]

# Step 1 - Login (handles session creation internally, no manual JWT needed)
print("Testing tokens...")
# print(f"Handle: {HANDLE}")

client = Client()

try:
    profile = client.login(HANDLE, APP_PASSWORD)
    print(f"Logged in as: {profile.display_name}")
except Exception as e:
    print(f"\nFAILED - Could not log in")
    print(f"Error: {e}")
    print("\nCheck your BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in .env")
    exit(1)

# Step 2 - Post a simple test message
try:
    post = client.send_post(text="Token test from my Bluesky agent. Ignore this post.")
    # rkey     = post.uri.split("/")[-1]

    print("Posted successfully")

    print("\nTokens are working correctly. You can delete that test post from your profile.")
except Exception as e:
    print(f"\nFAILED - Could not post")
    print(f"Error: {e}")
    exit(1)