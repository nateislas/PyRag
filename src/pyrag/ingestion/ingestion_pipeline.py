"""Main documentation ingestion pipeline."""

import asyncio
import re
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass

from ..vector_store import VectorStore
from ..embeddings import EmbeddingService
from ..graph.knowledge_graph import KnowledgeGraph
from ..libraries.manager import LibraryManager
from ..logging import get_logger
from ..llm.client import LLMClient
from .firecrawl_client import FirecrawlClient, ScrapedDocument
from .documentation_processor import DocumentationProcessor, ProcessingResult

logger = get_logger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for a single library's documentation ingestion."""
    library_name: str
    version: str
    docs_url: str
    crawl_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.crawl_options is None:
            self.crawl_options = {
                "crawl_entire_site": True,
                "max_pages": 20,
                "use_llm_filtering": True,
                "exclude_patterns": [
                    "linkedin.com", "twitter.com", "github.com",
                    "zapier.com", "pabbly.com", "facebook.com",
                    "youtube.com", "discord.com", "slack.com"
                ],
                "include_patterns": [
                    "/docs/", "/guide/", "/tutorial/", "/api/", "/reference/",
                    "/introduction", "/quickstart", "/examples"
                ]
            }

@dataclass
class IngestionResult:
    """Result of a documentation ingestion operation."""
    success: bool
    total_documents: int
    total_chunks: int
    processing_stats: Dict[str, Any]
    errors: List[str] = None
    crawled_urls: List[str] = None

class DocumentationIngestionPipeline:
    """Pipeline for ingesting documentation from various sources."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        library_manager: Optional[LibraryManager] = None,
        firecrawl_api_key: Optional[str] = None,
        llm_client: Optional[LLMClient] = None  # Add LLM client for smart filtering
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.knowledge_graph = knowledge_graph
        self.library_manager = library_manager
        self.firecrawl_api_key = firecrawl_api_key
        self.llm_client = llm_client
        self.processor = DocumentationProcessor()
        self.logger = get_logger(__name__)

    async def ingest_library_documentation(self, config: IngestionConfig) -> IngestionResult:
        """Ingest documentation for a specific library."""
        self.logger.info(f"Starting ingestion for {config.library_name}")
        
        try:
            # Scrape documentation (single page or multi-page crawl)
            if config.crawl_options.get("crawl_entire_site", False):
                documents = await self._crawl_documentation_site(config)
            else:
                documents = await self._scrape_documentation(config)
            
            if not documents:
                return IngestionResult(
                    success=False,
                    total_documents=0,
                    total_chunks=0,
                    processing_stats={},
                    errors=["No documents scraped"]
                )
            
            # Process documents into chunks
            processing_results = await self._process_documents(documents, config)
            
            # Store in vector database
            stored_chunks = await self._store_documents(processing_results)
            
            # Build knowledge graph if available
            if self.knowledge_graph:
                await self._build_knowledge_graph(processing_results, config)
            
            # Calculate statistics
            total_chunks = sum(len(result.chunks) for result in processing_results)
            total_content_length = sum(len(chunk.content) for result in processing_results for chunk in result.chunks)
            
            stats = {
                "total_documents": len(documents),
                "total_chunks": total_chunks,
                "total_content_length": total_content_length,
                "total_api_references": sum(len(result.metadata.get("api_references", [])) for result in processing_results),
                "content_type_distribution": self._calculate_content_distribution(processing_results),
                "average_chunks_per_document": total_chunks / len(documents) if documents else 0
            }
            
            self.logger.info(f"Successfully ingested {config.library_name}: {len(documents)} documents, {total_chunks} chunks")
            
            return IngestionResult(
                success=True,
                total_documents=len(documents),
                total_chunks=total_chunks,
                processing_stats=stats,
                crawled_urls=[doc.url for doc in documents]
            )
            
        except Exception as e:
            self.logger.error(f"Error ingesting {config.library_name}: {e}")
            return IngestionResult(
                success=False,
                total_documents=0,
                total_chunks=0,
                processing_stats={},
                errors=[str(e)]
            )

    async def _crawl_documentation_site(self, config: IngestionConfig) -> List[ScrapedDocument]:
        """Crawl entire documentation site with smart link filtering."""
        self.logger.info(f"Crawling documentation site: {config.docs_url}")
        
        try:
            async with FirecrawlClient(api_key=self.firecrawl_api_key) as client:
                # 1. Start with base page
                base_doc = await client.scrape_url(config.docs_url)
                documents = [base_doc]
                
                # 2. Extract all links from base page
                all_links = self._extract_links_from_content(base_doc.content, config.docs_url)
                self.logger.info(f"Found {len(all_links)} links on base page")
                
                # 3. Filter relevant documentation links
                relevant_links = await self._filter_relevant_links(config, all_links)
                self.logger.info(f"Filtered to {len(relevant_links)} relevant documentation links")
                
                # 4. Crawl each relevant page (up to max_pages)
                max_pages = config.crawl_options.get("max_pages", 20)
                crawled_urls = {config.docs_url}  # Track to avoid duplicates
                
                for link in relevant_links[:max_pages]:
                    if link not in crawled_urls:
                        try:
                            self.logger.info(f"Crawling: {link}")
                            doc = await client.scrape_url(link)
                            documents.append(doc)
                            crawled_urls.add(link)
                            
                            # Add small delay to be respectful
                            await asyncio.sleep(1)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to crawl {link}: {e}")
                            continue
                
                return documents
                
        except Exception as e:
            self.logger.error(f"Error crawling documentation site: {e}")
            raise

    async def _filter_relevant_links(self, config: IngestionConfig, all_links: List[str]) -> List[str]:
        """Filter links to only include relevant documentation pages."""
        relevant_links = []
        
        for link in all_links:
            # Basic filtering first
            if not self._is_relevant_link(link, config):
                continue
                
            # If LLM filtering is enabled, use it for additional validation
            if config.crawl_options.get("use_llm_filtering", False) and self.llm_client:
                if await self.llm_client.validate_link(link, config.docs_url, config.library_name):
                    relevant_links.append(link)
            else:
                relevant_links.append(link)
        
        return relevant_links

    def _is_relevant_link(self, link: str, config: IngestionConfig) -> bool:
        """Basic filtering to determine if a link is relevant."""
        parsed_url = urlparse(link)
        base_parsed = urlparse(config.docs_url)
        
        # Must be same domain
        if parsed_url.netloc != base_parsed.netloc:
            return False
        
        # Check exclude patterns
        exclude_patterns = config.crawl_options.get("exclude_patterns", [])
        for pattern in exclude_patterns:
            if pattern in link.lower():
                return False
        
        # Check include patterns
        include_patterns = config.crawl_options.get("include_patterns", [])
        for pattern in include_patterns:
            if pattern in link.lower():
                return True
        
        # If no include patterns match, be conservative
        return False



    def _extract_links_from_content(self, content: str, base_url: str) -> List[str]:
        """Extract all links from content (markdown and HTML)."""
        links = []
        
        # Pattern 1: Markdown links [text](url)
        markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(markdown_pattern, content):
            link_url = match.group(2)
            links.append(link_url)
        
        # Pattern 2: HTML anchor tags <a href="url">text</a>
        html_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
        for match in re.finditer(html_pattern, content, re.IGNORECASE):
            link_url = match.group(1)
            links.append(link_url)
        
        # Pattern 3: Plain URLs (http/https)
        url_pattern = r'https?://[^\s\)\]\>\"]+'
        for match in re.finditer(url_pattern, content):
            link_url = match.group(0)
            links.append(link_url)
        
        # Pattern 4: Relative URLs starting with /
        relative_pattern = r'["\'](/[^"\']+)["\']'
        for match in re.finditer(relative_pattern, content):
            link_url = match.group(1)
            links.append(link_url)
        
        # Pattern 5: Navigation links in various formats
        nav_patterns = [
            r'href=["\']([^"\']*/(?:docs|guide|tutorial|api|reference|learn|examples)[^"\']*)["\']',
            r'href=["\']([^"\']*/(?:introduction|quickstart|overview)[^"\']*)["\']',
        ]
        for pattern in nav_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                link_url = match.group(1)
                links.append(link_url)
        
        # Convert relative URLs to absolute and clean up
        absolute_links = []
        for link_url in links:
            # Skip anchors, fragments, and invalid URLs
            if (link_url.startswith('#') or 
                link_url.startswith('mailto:') or 
                link_url.startswith('javascript:') or
                not link_url.strip()):
                continue
                
            # Convert relative URLs to absolute
            if link_url.startswith('/'):
                link_url = urljoin(base_url, link_url)
            elif not link_url.startswith('http'):
                link_url = urljoin(base_url, link_url)
            
            # Clean up URL (remove fragments, normalize)
            clean_url = link_url.split('#')[0].split('?')[0]
            if clean_url:
                absolute_links.append(clean_url)
        
        return list(set(absolute_links))  # Remove duplicates

    async def _scrape_documentation(self, config: IngestionConfig) -> List[ScrapedDocument]:
        """Scrape single page documentation."""
        try:
            async with FirecrawlClient(api_key=self.firecrawl_api_key) as client:
                self.logger.info(f"Scraping single page: {config.docs_url}")
                document = await client.scrape_url(config.docs_url)
                return [document]
        except Exception as e:
            self.logger.error(f"Error scraping documentation: {e}")
            raise

    async def _process_documents(self, documents: List[ScrapedDocument], config: IngestionConfig) -> List[ProcessingResult]:
        """Process scraped documents into chunks."""
        processing_results = []
        
        for doc in documents:
            try:
                self.logger.info(f"Processing document: {doc.url}")
                result = self.processor.process_scraped_document(
                    scraped_doc=doc,
                    library_name=config.library_name,
                    version=config.version
                )
                processing_results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing document {doc.url}: {e}")
                continue
        
        return processing_results

    async def _store_documents(self, processing_results: List[ProcessingResult]) -> List[str]:
        """Store processed documents in vector store."""
        stored_chunks = []
        
        for result in processing_results:
            for chunk in result.chunks:
                try:
                    embedding = await self.embedding_service.embed_text(chunk.content)
                    document_data = {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "embedding": embedding
                    }
                    chunk_ids = await self.vector_store.add_documents([document_data])
                    stored_chunks.extend(chunk_ids)
                except Exception as e:
                    self.logger.error(f"Error storing chunk: {e}")
                    continue
        
        return stored_chunks

    async def _build_knowledge_graph(self, processing_results: List[ProcessingResult], config: IngestionConfig):
        """Build knowledge graph from processed documents."""
        if not self.knowledge_graph:
            return
        
        # Add library entity
        await self.knowledge_graph.add_entity(
            entity_id=f"library:{config.library_name}",
            entity_type="library",
            properties={
                "name": config.library_name,
                "version": config.version,
                "docs_url": config.docs_url
            }
        )
        
        # Add document entities and relationships
        for result in processing_results:
            for chunk in result.chunks:
                doc_id = f"doc:{chunk.metadata.get('url', 'unknown')}"
                
                await self.knowledge_graph.add_entity(
                    entity_id=doc_id,
                    entity_type="document",
                    properties=chunk.metadata
                )
                
                # Link document to library
                await self.knowledge_graph.add_relationship(
                    source_id=f"library:{config.library_name}",
                    target_id=doc_id,
                    relationship_type="contains",
                    properties={"chunk_index": chunk.metadata.get("chunk_index", 0)}
                )

    def _calculate_content_distribution(self, processing_results: List[ProcessingResult]) -> Dict[str, int]:
        """Calculate distribution of content types."""
        distribution = {}
        
        for result in processing_results:
            for chunk in result.chunks:
                content_type = chunk.metadata.get("content_type", "unknown")
                distribution[content_type] = distribution.get(content_type, 0) + 1
        
        return distribution
