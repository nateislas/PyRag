#!/bin/bash

# PyRAG Jetson Nano Deployment Script
set -e

echo "🚀 Deploying PyRAG MCP Server on Jetson Nano..."

# Check if running on Jetson Nano
if ! grep -q "jetson" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This script is designed for Jetson Nano. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborting deployment."
        exit 1
    fi
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Found: $python_version"
    echo "Please install Python 3.11+ and try again."
    exit 1
fi

echo "✅ Python version: $python_version"

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    git \
    nginx \
    ufw

# Create project directory
PROJECT_DIR="/opt/pyrag"
echo "📁 Setting up project directory: $PROJECT_DIR"
sudo mkdir -p "$PROJECT_DIR"
sudo chown "$USER:$USER" "$PROJECT_DIR"

# Clone or copy project files
if [ -d ".git" ]; then
    echo "📋 Copying current project to Jetson Nano..."
    cp -r . "$PROJECT_DIR/"
else
    echo "📋 Please copy your PyRAG project files to $PROJECT_DIR"
    echo "Then run this script again from that directory."
    exit 1
fi

cd "$PROJECT_DIR"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -e .[dev]

# Configure firewall with security
echo "🔥 Configuring secure firewall..."
# Only allow SSH and MCP server, block everything else
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp  # SSH only
sudo ufw allow 8000/tcp  # MCP Server only

# Block potentially dangerous ports
sudo ufw deny 8001/tcp  # ChromaDB - internal only
sudo ufw deny 5432/tcp  # PostgreSQL - internal only  
sudo ufw deny 6379/tcp  # Redis - internal only
sudo ufw deny 80/tcp    # HTTP - not needed for MCP
sudo ufw deny 443/tcp   # HTTPS - not needed for MCP

# Enable logging for security monitoring
sudo ufw logging on
sudo ufw --force enable

echo "✅ Firewall configured securely - only SSH (22) and MCP Server (8000) accessible"

# Create secure production .env file
echo "⚙️  Creating secure production .env file..."
cat > .env << EOF
# Environment
ENVIRONMENT=production

# MCP Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8000

# Security Configuration
MCP_ENABLE_RATE_LIMIT=true
MCP_RATE_LIMIT_REQUESTS=100
MCP_RATE_LIMIT_WINDOW=3600
MCP_ENABLE_API_KEYS=false
MCP_API_KEY_SECRET=your-secret-key-change-this-in-production
MCP_MAX_REQUEST_SIZE=1048576
MCP_ENABLE_IP_WHITELIST=false
MCP_ALLOWED_IPS=

# Database (using SQLite for simplicity on Jetson Nano)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pyrag
DB_USER=pyrag
DB_PASSWORD=pyrag

# Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8001

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Jetson Nano specific settings
DEVICE=cpu  # Use CPU for better compatibility
BATCH_SIZE=4  # Smaller batch size for Jetson Nano
EOF

# Create secure systemd service for MCP server
echo "🔧 Creating secure systemd service for MCP server..."
sudo tee /etc/systemd/system/pyrag-mcp.service > /dev/null << EOF
[Unit]
Description=PyRAG MCP Server (Secure)
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=PYTHONPATH=$PROJECT_DIR/src
ExecStart=$PROJECT_DIR/venv/bin/python scripts/run_mcp_server.py
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictSUIDSGID=true

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

# Create secure systemd service for essential services (internal only)
echo "🔧 Creating secure systemd service for essential services..."
sudo tee /etc/systemd/system/pyrag-services.service > /dev/null << EOF
[Unit]
Description=PyRAG Essential Services (Internal Only)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/docker-compose -f docker-compose.simple.yml up -d
ExecStop=$PROJECT_DIR/venv/bin/docker-compose -f docker-compose.simple.yml down
TimeoutStartSec=300

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictSUIDSGID=true

[Install]
WantedBy=multi-user.target
EOF

# Create security monitoring script
echo "🔒 Creating security monitoring script..."
cat > /opt/pyrag/security_monitor.sh << 'EOF'
#!/bin/bash
# Security monitoring script for PyRAG MCP server

LOG_FILE="/opt/pyrag/logs/security.log"
mkdir -p /opt/pyrag/logs

log_security_event() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Monitor failed login attempts
check_failed_logins() {
    failed_ssh=$(grep "Failed password" /var/log/auth.log | wc -l)
    if [ "$failed_ssh" -gt 10 ]; then
        log_security_event "WARNING: High number of failed SSH attempts: $failed_ssh"
    fi
}

# Monitor MCP server access
check_mcp_access() {
    if netstat -tlnp | grep -q ":8000"; then
        log_security_event "INFO: MCP server is running and accessible"
    else
        log_security_event "ERROR: MCP server is not accessible"
    fi
}

# Monitor firewall status
check_firewall() {
    if ufw status | grep -q "Status: active"; then
        log_security_event "INFO: Firewall is active"
    else
        log_security_event "ERROR: Firewall is not active"
    fi
}

# Monitor system resources
check_resources() {
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        log_security_event "WARNING: High memory usage: ${memory_usage}%"
    fi
    
    disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log_security_event "WARNING: High disk usage: ${disk_usage}%"
    fi
}

# Main monitoring loop
main() {
    log_security_event "Security monitoring started"
    
    while true; do
        check_failed_logins
        check_mcp_access
        check_firewall
        check_resources
        
        # Wait 5 minutes before next check
        sleep 300
    done
}

main
EOF

chmod +x /opt/pyrag/security_monitor.sh

# Create systemd service for security monitoring
echo "🔒 Creating security monitoring service..."
sudo tee /etc/systemd/system/pyrag-security-monitor.service > /dev/null << EOF
[Unit]
Description=PyRAG Security Monitor
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/security_monitor.sh
Restart=always
RestartSec=30

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

# Remove nginx configuration (not needed for MCP server)
echo "🗑️  Removing unnecessary nginx configuration..."
sudo systemctl stop nginx
sudo systemctl disable nginx
sudo apt remove -y nginx

# Enable and start secure services
echo "🚀 Enabling and starting secure services..."
sudo systemctl daemon-reload
sudo systemctl enable pyrag-services
sudo systemctl enable pyrag-mcp
sudo systemctl enable pyrag-security-monitor
sudo systemctl start pyrag-services
sudo systemctl start pyrag-mcp
sudo systemctl start pyrag-security-monitor

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Service status:"
sudo systemctl status pyrag-mcp --no-pager -l
sudo systemctl status pyrag-services --no-pager -l
sudo systemctl status pyrag-security-monitor --no-pager -l

echo ""
echo "✅ PyRAG MCP Server deployment complete with security features!"
echo ""
echo "🔒 Security Features Enabled:"
echo "  - Firewall: Only SSH (22) and MCP Server (8000) accessible"
echo "  - Rate limiting: 100 requests per hour per IP"
echo "  - Input sanitization: Prevents injection attacks"
echo "  - Service isolation: Internal services not exposed"
echo "  - Security monitoring: Continuous security checks"
echo ""
echo "🌐 Access Information:"
echo "  - MCP Server: http://$(hostname -I | awk '{print $1}'):8000"
echo "  - Health Check: http://$(hostname -I | awk '{print $1}'):8000/health"
echo ""
echo "🔧 Management Commands:"
echo "  - Start MCP server: sudo systemctl start pyrag-mcp"
echo "  - Stop MCP server: sudo systemctl stop pyrag-mcp"
echo "  - View logs: sudo journalctl -u pyrag-mcp -f"
echo "  - Security logs: tail -f /opt/pyrag/logs/security.log"
echo "  - Restart services: sudo systemctl restart pyrag-mcp"
echo ""
echo "📱 For remote access, configure your MCP client to connect to:"
echo "  $(hostname -I | awk '{print $1}'):8000"
echo ""
echo "⚠️  IMPORTANT: Change the API key secret in .env file for production use!"
