# Secure PyRAG MCP Client Setup Guide

## Overview

This guide explains how to securely connect to your PyRAG MCP server running on Jetson Nano. The server is configured with security features to protect your system while allowing legitimate users to access the service.

## Security Features

### 🔒 What's Protected
- **Your WiFi network**: Only the MCP server is accessible
- **Your personal data**: Internal services are isolated
- **System access**: No administrative privileges granted
- **Resource abuse**: Rate limiting prevents DoS attacks

### ✅ What's Accessible
- **MCP Server**: Python documentation queries
- **Health endpoint**: Basic service status
- **Rate-limited requests**: 100 requests per hour per IP

## Connection Methods

### Method 1: Direct IP Connection (Local Network)

#### For Cursor IDE Users
Create or edit `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "pyrag-jetson": {
      "command": "curl",
      "args": [
        "-X", "POST", 
        "http://10.0.0.95:8000/mcp",
        "-H", "Content-Type: application/json",
        "-d", "{\"query\": \"$QUERY\"}"
      ],
      "env": {}
    }
  }
}
```

#### For Other MCP Clients
```bash
# Test connection
curl -X POST http://10.0.0.95:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"query": "how to use pandas DataFrame"}'
```

### Method 2: API Key Authentication (Optional)

If you enable API keys on the server, clients can use them for enhanced access:

```json
{
  "mcpServers": {
    "pyrag-jetson-secure": {
      "command": "curl",
      "args": [
        "-X", "POST", 
        "http://10.0.0.95:8000/mcp",
        "-H", "Content-Type: application/json",
        "-H", "X-API-Key: YOUR_API_KEY",
        "-d", "{\"query\": \"$QUERY\"}"
      ],
      "env": {}
    }
  }
}
```

## Client Configuration Examples

### Cursor IDE Configuration

1. **Open Cursor IDE**
2. **Go to Settings** → **Extensions** → **MCP**
3. **Add PyRAG server**:
   ```json
   {
     "mcpServers": {
       "pyrag-jetson": {
         "command": "curl",
         "args": [
           "-X", "POST", 
           "http://10.0.0.95:8000/mcp",
           "-H", "Content-Type: application/json",
           "-d", "{\"query\": \"$QUERY\"}"
         ],
         "env": {}
       }
     }
   }
   ```
4. **Restart Cursor**
5. **Test with a Python question**

### VS Code Configuration

1. **Install MCP extension**
2. **Configure in settings.json**:
   ```json
   {
     "mcp.servers": {
       "pyrag-jetson": {
         "command": "curl",
         "args": [
           "-X", "POST", 
           "http://10.0.0.95:8000/mcp",
           "-H", "Content-Type: application/json",
           "-d", "{\"query\": \"$QUERY\"}"
         ]
       }
     }
   }
   ```

### Python Script Client

```python
import requests
import json

class PyRAGClient:
    def __init__(self, server_url="http://10.0.0.95:8000"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def query(self, question: str, library: str = None) -> str:
        """Query the PyRAG MCP server."""
        payload = {
            "query": question,
            "library": library
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "No response")
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to PyRAG: {e}"
    
    def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False

# Usage example
if __name__ == "__main__":
    client = PyRAGClient()
    
    # Check server health
    if client.health_check():
        print("✅ PyRAG server is healthy")
        
        # Ask a question
        response = client.query("How do I use pandas read_csv?")
        print(f"📚 Answer: {response}")
    else:
        print("❌ PyRAG server is not accessible")
```

## Security Best Practices for Clients

### 1. Network Security
- **Use local network**: Connect from devices on the same WiFi network
- **Avoid public WiFi**: Don't connect from public/untrusted networks
- **VPN access**: Consider using a VPN for additional security

### 2. Authentication
- **API keys**: Use API keys if provided by the server admin
- **Rate limiting**: Respect the 100 requests/hour limit
- **Monitoring**: Report any suspicious activity

### 3. Request Security
- **Input validation**: Don't send malicious queries
- **Size limits**: Keep queries under 1MB
- **Timeout handling**: Implement proper timeout handling

## Troubleshooting

### Connection Issues

#### "Connection Refused"
```bash
# Check if server is running
curl http://10.0.0.95:8000/health

# Check if port is open
telnet 10.0.0.95 8000
```

#### "Rate Limit Exceeded"
- Wait for the rate limit window to reset (1 hour)
- Reduce request frequency
- Contact server admin for higher limits

#### "Invalid Request"
- Check request format
- Ensure Content-Type header is set
- Validate query length (under 1MB)

### Performance Issues

#### Slow Responses
- Server may be under load
- Check server health endpoint
- Consider reducing query complexity

#### Timeout Errors
- Increase client timeout settings
- Check network connectivity
- Verify server is responsive

## Monitoring and Logs

### Client-Side Monitoring
```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log all requests
logger.info(f"Querying PyRAG: {question}")
logger.info(f"Response received: {len(response)} characters")
```

### Server Health Monitoring
```bash
# Check server status
curl http://10.0.0.95:8000/health

# Monitor server logs (if you have access)
ssh jetbot@10.0.0.95 "tail -f /opt/pyrag/logs/security.log"
```

## Advanced Configuration

### Custom Rate Limiting
If you need different rate limits, contact the server administrator to adjust:
- `MCP_RATE_LIMIT_REQUESTS`: Number of requests per window
- `MCP_RATE_LIMIT_WINDOW`: Time window in seconds

### IP Whitelisting
For enhanced security, the server can be configured to only allow specific IP addresses:
- `MCP_ENABLE_IP_WHITELIST=true`
- `MCP_ALLOWED_IPS=192.168.1.100,192.168.1.101`

### API Key Management
When enabled, API keys provide:
- Higher rate limits
- Better monitoring
- Access control
- Usage analytics

## Support and Contact

### Getting Help
- **Server issues**: Contact the Jetson Nano administrator
- **Client issues**: Check this guide and troubleshooting section
- **Security concerns**: Report immediately to server admin

### Useful Commands
```bash
# Test basic connectivity
ping 10.0.0.95

# Test HTTP connectivity
curl -v http://10.0.0.95:8000/health

# Check network route
traceroute 10.0.0.95
```

## Conclusion

Your PyRAG MCP server is now securely accessible while protecting your system. Users can connect and ask Python questions without compromising your WiFi or personal data. The security features ensure legitimate usage while preventing abuse.

Remember to:
- ✅ Use the service responsibly
- ✅ Respect rate limits
- ✅ Report any issues
- ✅ Keep your client software updated
- ✅ Use secure network connections
