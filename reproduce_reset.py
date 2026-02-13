
import asyncio
import os
import sys
from datetime import datetime

# Adjust path to include project root
sys.path.append(os.getcwd())

from memory.storage import MemoryStorage, GLOBAL_USER_ID
from sqlalchemy import select, text

async def main():
    print(f"Global User ID: {GLOBAL_USER_ID}")
    storage = MemoryStorage("data/sessions.db")
    await storage.initialize()

    # 1. Force set state to 1
    print("Setting onboarding_prompted = 1...")
    await storage.mark_onboarding_prompted()

    state = await storage.get_global_user_state()
    print(f"State before reset: onboarding_prompted={state['onboarding_prompted']}")

    # 2. Run reset
    print("Running reset_all_memory()...")
    await storage.reset_all_memory()

    # 3. Check state
    state = await storage.get_global_user_state()
    print(f"State after reset: onboarding_prompted={state['onboarding_prompted']}")

    # 4. Check raw DB value
    async with storage.session_factory() as session:
        result = await session.execute(text("SELECT user_id, onboarding_prompted FROM users"))
        rows = result.fetchall()
        print("Raw users table content:")
        for row in rows:
            print(f"User: {row[0]}, Onboarding: {row[1]}")

if __name__ == "__main__":
    asyncio.run(main())
