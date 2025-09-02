"""PyRAG MCP Server using FastMCP with security features."""

import asyncio
import hashlib
import hmac
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastmcp import Context, FastMCP

from ..core import PyRAG
from ..logging import get_logger

logger = get_logger(__name__)


# Security configuration
class SecurityConfig:
    """Security configuration for the MCP server."""

    def __init__(self):
        self.enable_rate_limiting = (
            os.getenv("MCP_ENABLE_RATE_LIMIT", "true").lower() == "true"
        )
        self.rate_limit_requests = int(
            os.getenv("MCP_RATE_LIMIT_REQUESTS", "100")
        )  # requests per window
        self.rate_limit_window = int(
            os.getenv("MCP_RATE_LIMIT_WINDOW", "3600")
        )  # seconds (1 hour)
        self.enable_api_keys = (
            os.getenv("MCP_ENABLE_API_KEYS", "false").lower() == "true"
        )
        self.api_key_secret = os.getenv(
            "MCP_API_KEY_SECRET", "your-secret-key-change-this"
        )
        self.max_request_size = int(os.getenv("MCP_MAX_REQUEST_SIZE", "1048576"))  # 1MB
        self.allowed_ips = (
            os.getenv("MCP_ALLOWED_IPS", "").split(",")
            if os.getenv("MCP_ALLOWED_IPS")
            else []
        )
        self.enable_ip_whitelist = (
            os.getenv("MCP_ENABLE_IP_WHITELIST", "false").lower() == "true"
        )


# Initialize security config
security_config = SecurityConfig()

# Rate limiting storage
rate_limit_store = defaultdict(lambda: deque(maxlen=1000))

# Initialize PyRAG core system immediately
_pyrag_instance = None


def initialize_pyrag():
    """Initialize PyRAG system immediately."""
    global _pyrag_instance
    if _pyrag_instance is None:
        logger.info("Initializing PyRAG system for MCP server")
        _pyrag_instance = PyRAG()
        logger.info("PyRAG system initialized and ready")


def get_pyrag() -> PyRAG:
    """Get PyRAG instance (assumes it's already initialized)."""
    global _pyrag_instance
    if _pyrag_instance is None:
        raise RuntimeError(
            "PyRAG system not initialized. Call initialize_pyrag() first."
        )
    return _pyrag_instance


# Create FastMCP server instance
mcp = FastMCP("PyRAG 🐍")

# Initialize PyRAG system immediately
initialize_pyrag()


def validate_api_key(api_key: str, timestamp: str, signature: str) -> bool:
    """Validate API key signature."""
    if not security_config.enable_api_keys:
        return True

    try:
        # Check if timestamp is recent (within 5 minutes)
        request_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if (
            abs((datetime.now(request_time.tzinfo) - request_time).total_seconds())
            > 300
        ):
            return False

        # Verify signature
        expected_signature = hmac.new(
            security_config.api_key_secret.encode(),
            f"{api_key}:{timestamp}".encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.warning(f"API key validation error: {e}")
        return False


def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit."""
    if not security_config.enable_rate_limiting:
        return True

    now = time.time()
    client_requests = rate_limit_store[client_id]

    # Remove old requests outside the window
    while (
        client_requests and client_requests[0] < now - security_config.rate_limit_window
    ):
        client_requests.popleft()

    # Check if limit exceeded
    if len(client_requests) >= security_config.rate_limit_requests:
        return False

    # Add current request
    client_requests.append(now)
    return True


def sanitize_input(text: str) -> str:
    """Sanitize input text to prevent injection attacks."""
    if not text:
        return ""

    # Remove potentially dangerous characters
    dangerous_chars = ["<", ">", '"', "'", "&", ";", "|", "`", "$", "(", ")"]
    sanitized = text
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")

    # Limit length
    if len(sanitized) > security_config.max_request_size:
        sanitized = sanitized[: security_config.max_request_size]

    return sanitized


async def security_middleware(ctx: Context, **kwargs) -> bool:
    """Security middleware for MCP requests."""
    try:
        # Get client IP (if available)
        client_ip = getattr(ctx, "client_ip", "unknown")

        # IP whitelist check
        if security_config.enable_ip_whitelist and security_config.allowed_ips:
            if client_ip not in security_config.allowed_ips:
                logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
                return False

        # Rate limiting check
        if not check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False

        # API key validation (if enabled)
        if security_config.enable_api_keys:
            api_key = kwargs.get("api_key")
            timestamp = kwargs.get("timestamp")
            signature = kwargs.get("signature")

            if not all([api_key, timestamp, signature]):
                logger.warning(
                    f"Missing authentication parameters from IP: {client_ip}"
                )
                return False

            if not validate_api_key(api_key, timestamp, signature):
                logger.warning(f"Invalid API key from IP: {client_ip}")
                return False

        # Input sanitization
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = sanitize_input(value)

        return True

    except Exception as e:
        logger.error(f"Security middleware error: {e}")
        return False


@mcp.tool
async def search_python_docs(
    query: str,
    library: Optional[str] = None,
    version: Optional[str] = None,
    content_type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Search current Python library documentation with semantic understanding.

    Args:
        query: Natural language query about Python functionality
        library: Specific library to search within (optional)
        version: Specific version constraint (optional)
        content_type: Type of documentation to prioritize (examples, reference, tutorials, all)
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Searching Python docs for: {query}")
        if library:
            await ctx.info(f"Filtering by library: {library}")

    try:
        # Map content_type to our internal format
        mapped_content_type = None
        if content_type:
            if content_type in ["examples", "tutorials"]:
                mapped_content_type = "examples"
            elif content_type in ["reference", "api_reference"]:
                mapped_content_type = "api_reference"
            elif content_type == "overview":
                mapped_content_type = "overview"

        # Search documentation
        results = await get_pyrag().search_documentation(
            query=query,
            library=library,
            version=version,
            content_type=mapped_content_type,
            max_results=10,
        )

        if not results:
            return "No relevant documentation found for your query."

        # Format results
        response_parts = []
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
            response_parts.append(f"**Result {i}** (Score: {result['score']:.2f})")
            response_parts.append(f"Library: {result['library']} v{result['version']}")
            response_parts.append(f"Type: {result['content_type']}")
            response_parts.append("")
            response_parts.append(
                result["content"][:500] + "..."
                if len(result["content"]) > 500
                else result["content"]
            )
            response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error searching documentation: {e}")
        if ctx:
            await ctx.error(f"Error searching documentation: {e}")
        return f"Error searching documentation: {e}"


@mcp.tool
async def get_api_reference(
    library: str,
    api_path: str,
    include_examples: bool = True,
    ctx: Context = None,
) -> str:
    """Get detailed API reference for specific functions/classes.

    Args:
        library: Library name (e.g., "requests", "pandas")
        api_path: API path (e.g., "pandas.DataFrame.merge")
        include_examples: Whether to include usage examples
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Getting API reference for {api_path} in {library}")

    try:
        result = await get_pyrag().get_api_reference(
            library=library,
            api_path=api_path,
            include_examples=include_examples,
        )

        if not result:
            return f"No API reference found for {api_path} in {library}"

        # Format response
        response_parts = []
        response_parts.append(f"# {api_path}")
        response_parts.append(f"**Library:** {result['library']}")
        response_parts.append(f"**Score:** {result['score']:.2f}")
        response_parts.append("")
        response_parts.append("## Description")
        response_parts.append(result["content"])
        response_parts.append("")

        if result["examples"]:
            response_parts.append("## Examples")
            for i, example in enumerate(result["examples"], 1):
                response_parts.append(f"### Example {i}")
                response_parts.append("```python")
                response_parts.append(example)
                response_parts.append("```")
                response_parts.append("")

        if result["related_apis"]:
            response_parts.append("## Related APIs")
            for related in result["related_apis"]:
                response_parts.append(
                    f"- **{related['api_path']}** (Score: {related['score']:.2f})"
                )
                response_parts.append(f"  {related['content']}")
                response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error getting API reference: {e}")
        if ctx:
            await ctx.error(f"Error getting API reference: {e}")
        return f"Error getting API reference: {e}"


@mcp.tool
async def check_deprecation(
    library: str,
    apis: List[str],
    ctx: Context = None,
) -> str:
    """Check if APIs are deprecated and get replacement suggestions.

    Args:
        library: Library name to check
        apis: List of API paths to check for deprecation
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Checking deprecation status for {len(apis)} APIs in {library}")

    try:
        result = await get_pyrag().check_deprecation(library=library, apis=apis)

        if not result["deprecated_apis"]:
            return f"✅ All checked APIs in {library} are current and not deprecated."

        # Format response
        response_parts = []
        response_parts.append(f"# Deprecation Check for {library}")
        response_parts.append("")

        for api in result["deprecated_apis"]:
            response_parts.append(f"## ⚠️ {api}")
            if api in result["replacement_suggestions"]:
                replacement = result["replacement_suggestions"][api]
                response_parts.append(f"**Replacement:** {replacement}")
            response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error checking deprecation: {e}")
        if ctx:
            await ctx.error(f"Error checking deprecation: {e}")
        return f"Error checking deprecation: {e}"


@mcp.tool
async def find_similar_patterns(
    code_snippet: str,
    intent: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Find similar usage patterns or alternative approaches.

    Args:
        code_snippet: Python code snippet to find patterns for
        intent: Optional description of what the code is trying to achieve
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Finding similar patterns for code snippet")
        if intent:
            await ctx.info(f"Intent: {intent}")

    try:
        results = await get_pyrag().find_similar_patterns(
            code_snippet=code_snippet,
            intent=intent,
        )

        if not results:
            return "No similar patterns found for the provided code snippet."

        # Format response
        response_parts = []
        response_parts.append("# Similar Usage Patterns")
        response_parts.append("")

        for i, pattern in enumerate(results[:5], 1):  # Limit to top 5
            response_parts.append(f"## Pattern {i}")
            response_parts.append(f"**Library:** {pattern.get('library', 'Unknown')}")
            response_parts.append(f"**Score:** {pattern.get('score', 0):.2f}")
            response_parts.append("")
            response_parts.append("**Code:**")
            response_parts.append("```python")
            response_parts.append(pattern.get("code", ""))
            response_parts.append("```")
            response_parts.append("")
            response_parts.append("**Description:**")
            response_parts.append(pattern.get("description", ""))
            response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error finding similar patterns: {e}")
        if ctx:
            await ctx.error(f"Error finding similar patterns: {e}")
        return f"Error finding similar patterns: {e}"


@mcp.tool
async def list_available_libraries(ctx: Context = None) -> str:
    """List all available libraries in the PyRAG system.

    Args:
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info("Listing available libraries")

    try:
        libraries = await get_pyrag().list_libraries()

        if not libraries:
            return "No libraries are currently indexed in the system."

        # Format response
        response_parts = []
        response_parts.append("# Available Libraries")
        response_parts.append("")

        for lib in libraries:
            status_emoji = (
                "✅"
                if lib["status"] == "indexed"
                else "⏳"
                if lib["status"] == "indexing"
                else "❌"
            )
            response_parts.append(f"## {status_emoji} {lib['name']}")
            if lib["description"]:
                response_parts.append(f"**Description:** {lib['description']}")
            response_parts.append(f"**Status:** {lib['status']}")
            if lib["latest_version"]:
                response_parts.append(f"**Latest Version:** {lib['latest_version']}")
            if lib["chunk_count"]:
                response_parts.append(f"**Documentation Chunks:** {lib['chunk_count']}")
            response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error listing libraries: {e}")
        if ctx:
            await ctx.error(f"Error listing libraries: {e}")
        return f"Error listing libraries: {e}"


@mcp.tool
async def get_library_status(
    library_name: str,
    ctx: Context = None,
) -> str:
    """Get detailed status of a specific library.

    Args:
        library_name: Name of the library to check
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Getting status for library: {library_name}")

    try:
        status = await get_pyrag().get_library_status(library_name)

        if not status:
            return f"Library '{library_name}' not found in the system."

        # Format response
        response_parts = []
        response_parts.append(f"# Library Status: {status['name']}")
        response_parts.append("")

        status_emoji = (
            "✅"
            if status["status"] == "indexed"
            else "⏳"
            if status["status"] == "indexing"
            else "❌"
        )
        response_parts.append(f"**Status:** {status_emoji} {status['status']}")

        if status["latest_version"]:
            response_parts.append(f"**Latest Version:** {status['latest_version']}")

        if status["chunk_count"]:
            response_parts.append(f"**Documentation Chunks:** {status['chunk_count']}")

        if status["last_checked"]:
            response_parts.append(f"**Last Checked:** {status['last_checked']}")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error getting library status: {e}")
        if ctx:
            await ctx.error(f"Error getting library status: {e}")
        return f"Error getting library status: {e}"


async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting PyRAG MCP Server")

    # Get transport configuration from environment variables
    transport_type = os.getenv("MCP_TRANSPORT", "stdio").lower()

    if transport_type == "stdio":
        logger.info("Starting MCP server in STDIO mode")
        # Run the server in stdio mode (synchronous)
        mcp.run(transport="stdio")
    elif transport_type == "http":
        logger.info("Starting MCP server in HTTP mode")
        # Get network configuration from environment variables
        host = os.getenv("MCP_HOST", "0.0.0.0")  # Bind to all interfaces by default
        port = int(os.getenv("MCP_PORT", "8000"))  # Default MCP port

        logger.info(f"Binding MCP server to {host}:{port}")

        # Run the server with network binding
        await mcp.run(host=host, port=port)
    else:
        logger.error(f"Unsupported transport type: {transport_type}")
        logger.info("Supported transport types: stdio, http")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
