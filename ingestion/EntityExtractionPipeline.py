import asyncio
from concurrent.futures import ThreadPoolExecutor

from ingestion.ner_extractor import NERExtractor
from ingestion.entity_linker import EntityLinker
from ingestion.entity_canonicalizer import EntityCanonicalizer

class EntityExtractionPipeline:

    def __init__(self, config: dict):

        self.ner = NERExtractor(config)
        self.linker = EntityLinker(config)

        self.canonicalizer = EntityCanonicalizer()

        self.max_workers = config.get("max_workers", 4)

    async def process_chunk(self, chunk):

        entities = self.ner.extract(chunk["text"])

        entities = await self.linker.link_entities(entities)
   
        for e in entities:

            e.source_doc = chunk.get("title")
            e.source_chunk = chunk.get("id")
            e.canonical = self.canonicalizer.canonicalize(e)


        return entities

    async def process_chunks(self, chunks):

        tasks = []

        for chunk in chunks:
            tasks.append(self.process_chunk(chunk))

        results = await asyncio.gather(*tasks)

        return [e for group in results for e in group]

    def run(self, chunks):

        return asyncio.run(self.process_chunks(chunks))
