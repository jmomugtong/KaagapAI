import asyncio
import os
import sys

# Ensure src module can be imported
sys.path.append(os.getcwd())

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.db.models import Base

# Using the password from .env
DATABASE_URL = (
    "postgresql+asyncpg://medquery_user:medquery_password@localhost:5432/medquery"
)


async def main():
    print(f"Testing Schema Creation on: {DATABASE_URL}")
    try:
        engine = create_async_engine(DATABASE_URL)
        async with engine.begin() as conn:
            print("Creating extension...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("Dropping tables...")
            await conn.run_sync(Base.metadata.drop_all)
            print("Creating tables...")
            await conn.run_sync(Base.metadata.create_all)
            print("Schema Creation Successful!")
    except Exception as e:
        print(f"Schema Creation Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
