# Docling Document Processing

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

2. Run the script:
```bash
cd "c:\Users\User\OneDrive\Desktop\VScode\Docling Test"
```

```bash
python main.py
```

## Manual Cache Clearing 
### Remove all cached models

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

## Citation

This project uses the [Docling](https://github.com/docling-project/docling) framework for document processing.  
If you use this project or build upon it, please cite the original Docling work:

**Docling Technical Report**  
Deep Search Team. *Docling Technical Report*. arXiv:2408.09869, August 2024. [https://arxiv.org/abs/2408.09869](https://arxiv.org/abs/2408.09869)

BibTeX:
```bibtex
@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {Docling Technical Report},
  url = {https://arxiv.org/abs/2408.09869},
  eprint = {2408.09869},
  doi = {10.48550/arXiv.2408.09869},
  version = {1.0.0},
  year = {2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
