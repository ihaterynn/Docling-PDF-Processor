# Docling PDF Processing Project

This project uses Docling to extract and process content from PDF documents, specifically designed to handle research papers with accurate table extraction.

## Features

- **Advanced PDF Processing**: Uses Docling's pipeline for comprehensive document conversion
- **Table Structure Recognition**: Configured for accurate table extraction from research papers
- **Memory Management**: Built-in limits to prevent system overload
- **JSON Export**: Converts processed content to structured JSON format

## Prerequisites

- Python 3.8 or higher
- Docling library installed

## Installation

1. Install Docling:
```bash
pip install docling
```

```bash
cd "c:\Users\User\OneDrive\Desktop\VScode\Docling Test"
```

```bash
python main.py
```

## Manual Cache Clearing 
# Remove all cached models

```bash
rmdir /s "%USERPROFILE%\.cache\docling\models"
```

### Configuration Options
The script is configured with the following settings:

- Max Pages : 30 pages (to prevent overload)
- Max File Size : 15MB limit
- Table Processing : Enabled with accurate mode
- Cell Matching : Disabled to avoid column merge issues

### Interactive Features
When you run the script, it will:

1. Display current cache size and cached models
2. Ask if you want to clear cache before processing
3. Process the PDF document
4. Show updated cache information
5. Ask if you want to clear cache after processing

## Cache Management
### Understanding Docling Cache
Docling stores models in the cache directory: ~/.cache/docling/models/

This cache can grow large over time as it downloads and stores various AI models for document processing.