"""
Deduplicate documents in the database.

Keeps the latest upload of each unique filename and deletes older duplicates
along with their associated chunks.
"""

import asyncio
import logging

from sqlalchemy import delete, func, select

from src.db.models import ClinicalDoc, DocumentChunk
from src.db.postgres import AsyncSessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def deduplicate_documents():
    """Remove duplicate documents, keeping only the latest upload of each filename."""
    async with AsyncSessionLocal() as session:
        # Get all documents grouped by filename
        result = await session.execute(
            select(ClinicalDoc.filename, func.count(ClinicalDoc.id).label("count"))
            .group_by(ClinicalDoc.filename)
            .order_by(func.count(ClinicalDoc.id).desc())
        )
        file_counts = result.all()

        logger.info("=== Document Duplicates ===")
        total_docs = 0
        duplicates = 0
        for filename, count in file_counts:
            total_docs += count
            if count > 1:
                duplicates += count - 1
                logger.info(f"  {filename}: {count} copies")

        logger.info(f"\nTotal documents: {total_docs}")
        logger.info(f"Duplicates to remove: {duplicates}")
        logger.info(f"Will keep: {total_docs - duplicates}\n")

        if duplicates == 0:
            logger.info("No duplicates found!")
            return

        # Confirm deletion
        response = input(
            f"Delete {duplicates} duplicate documents and their chunks? (yes/no): "
        )
        if response.lower() != "yes":
            logger.info("Cancelled.")
            return

        # For each filename with duplicates, keep the latest (highest ID) and delete the rest
        deleted_docs = 0
        deleted_chunks = 0

        for filename, count in file_counts:
            if count <= 1:
                continue

            # Get all IDs for this filename, sorted newest first
            result = await session.execute(
                select(ClinicalDoc.id)
                .where(ClinicalDoc.filename == filename)
                .order_by(ClinicalDoc.id.desc())
            )
            doc_ids = [row[0] for row in result.all()]

            # Keep the first (newest), delete the rest
            to_delete = doc_ids[1:]

            for doc_id in to_delete:
                # Count chunks before deleting
                result = await session.execute(
                    select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.document_id == doc_id
                    )
                )
                chunk_count = result.scalar()

                # Delete chunks first (foreign key constraint)
                await session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == doc_id)
                )

                # Delete document
                await session.execute(
                    delete(ClinicalDoc).where(ClinicalDoc.id == doc_id)
                )

                deleted_docs += 1
                deleted_chunks += chunk_count
                logger.info(
                    f"  Deleted document {doc_id} ({filename}) with {chunk_count} chunks"
                )

        await session.commit()

        logger.info(f"\n=== Cleanup Complete ===")
        logger.info(f"Deleted {deleted_docs} duplicate documents")
        logger.info(f"Deleted {deleted_chunks} chunks")

        # Show final counts
        result = await session.execute(select(func.count(ClinicalDoc.id)))
        final_docs = result.scalar()
        result = await session.execute(select(func.count(DocumentChunk.id)))
        final_chunks = result.scalar()

        logger.info(f"\nRemaining: {final_docs} documents, {final_chunks} chunks")


if __name__ == "__main__":
    asyncio.run(deduplicate_documents())
