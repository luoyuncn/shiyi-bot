
import asyncio
import os
import sys
from datetime import datetime

sys.path.append(os.getcwd())
from memory.storage import MemoryStorage
from sqlalchemy import text

async def main():
    storage = MemoryStorage("data/sessions.db")
    await storage.initialize()

    async with storage.session_factory() as session:
        # 1. Check Messages
        print("\n=== Recent Messages ===")
        result = await session.execute(text("SELECT role, content FROM messages ORDER BY timestamp DESC LIMIT 10"))
        rows = result.fetchall()
        for row in rows:
            print(f"[{row[0]}] {row[1][:50]}...")

        # 2. Check Embeddings
        print("\n=== Embeddings Count ===")
        result = await session.execute(text("SELECT count(*) FROM memory_embeddings"))
        count = result.scalar()
        print(f"Total embeddings: {count}")

        # 3. Check specific beef noodle embedding if it exists
        print("\n=== Searching for '牛肉面' in messages ===")
        result = await session.execute(text("SELECT id, content FROM messages WHERE content LIKE '%牛肉面%'"))
        beef_rows = result.fetchall()
        if not beef_rows:
            print("No '牛肉面' message found in DB.")
        else:
            for row in beef_rows:
                mid = row[0]
                content = row[1]
                print(f"Found message: {content[:30]}... (ID: {mid})")
                # Check if embedding exists for this ID
                emb_res = await session.execute(text("SELECT id FROM memory_embeddings WHERE source_id = :mid"), {"mid": mid})
                if emb_res.scalar():
                    print("  -> Embedding EXISTS.")
                else:
                    print("  -> Embedding MISSING.")

if __name__ == "__main__":
    asyncio.run(main())
