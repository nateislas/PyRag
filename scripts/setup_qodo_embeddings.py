#!/usr/bin/env python3
"""Setup script for Qodo embeddings."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔧 {description}...")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stderr.strip():
            print(f"   Error: {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   PyRAG requires Python 3.11 or higher")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Update pip
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install sentence-transformers
    if not run_command("pip install sentence-transformers>=2.2.0", "Installing sentence-transformers"):
        return False
    
    # Install flash-attn (optional but recommended)
    print("   💡 Installing flash-attn (optional but recommended for performance)...")
    if not run_command("pip install flash-attn>=2.5.6", "Installing flash-attn"):
        print("   ⚠️  flash-attn installation failed, but sentence-transformers will still work")
        print("   💡 You can try installing it later for better performance")
    
    return True


def verify_installation():
    """Verify that the installation works."""
    print("\n🔍 Verifying installation...")
    
    try:
        import sentence_transformers
        print(f"   ✅ sentence-transformers {sentence_transformers.__version__} installed")
        
        # Try to import the model (this will download it)
        print("   📥 Downloading Qodo model (this may take a few minutes)...")
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B")
        print(f"   ✅ Qodo model loaded successfully")
        print(f"   📏 Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Qodo Embedding Setup for PyRAG")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n💥 Setup failed: Python version incompatible")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n💥 Setup failed: Dependency installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n💥 Setup failed: Installation verification failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Run the test script: python scripts/test_qodo_embeddings.py")
    print("   2. Update your .env file if needed")
    print("   3. Run the ingestion pipeline to re-embed existing documents")
    print("\n🔧 Configuration options:")
    print("   - EMBEDDING_MODEL: Qodo/Qodo-Embed-1-1.5B")
    print("   - EMBEDDING_DEVICE: auto, cuda, or cpu")
    print("   - EMBEDDING_MAX_LENGTH: 32768 (max supported)")
    print("   - EMBEDDING_BATCH_SIZE: 8 (adjust based on your GPU memory)")


if __name__ == "__main__":
    main()
