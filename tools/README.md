# PyRAG Tools

This directory contains development and operational tools for PyRAG.

## Directory Structure

### `pipeline/` - Core Pipeline Scripts
Production-ready scripts for running PyRAG pipelines:
- `automated_library_ingestion.py` - Batch processing of multiple libraries
- `check_current_state.py` - System state monitoring and verification
- `setup_qodo_embeddings.py` - Qodo embedding model setup and configuration

### `testing/` - Testing and Validation Scripts
Development and testing scripts for validating PyRAG functionality:
- `test_complete_system.py` - Comprehensive end-to-end system validation
- `test_enhanced_integration.py` - Enhanced processing pipeline testing
- `test_enhanced_processing.py` - Individual processing component testing
- `test_rag_evaluation.py` - RAG quality evaluation system testing
- `test_qodo_embeddings.py` - Qodo embedding model testing

## Usage

### Pipeline Scripts
```bash
# Run automated library ingestion
python tools/pipeline/automated_library_ingestion.py

# Check system state
python tools/pipeline/check_current_state.py

# Setup Qodo embeddings
python tools/pipeline/setup_qodo_embeddings.py
```

### Testing Scripts
```bash
# Run complete system test
python tools/testing/test_complete_system.py

# Test enhanced processing
python tools/testing/test_enhanced_processing.py

# Test RAG evaluation
python tools/testing/test_rag_evaluation.py
```

## Note
These tools are for development and operational use. They are not included in the main PyRAG package distribution.
