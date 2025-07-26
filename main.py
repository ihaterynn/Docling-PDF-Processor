from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
import shutil
import os
import json
from pathlib import Path
from datetime import datetime
import fitz  
import re
from PIL import Image
import io

def check_cuda_availability():
    """Check if CUDA is available for GPU acceleration"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"🚀 CUDA available! Using GPU: {device_name} (Devices: {device_count})")
            return True
        else:
            print("⚠️ CUDA not available, using CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed, using CPU (install torch for GPU support)")
        return False
    except Exception as e:
        print(f"⚠️ Error checking CUDA: {e}, using CPU")
        return False

# Cache management functions
def clear_docling_cache():
    """Clear all Docling cached models to free up space"""
    cache_path = Path.home() / ".cache" / "docling" / "models"
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
            print(f"✅ Cleared Docling cache at: {cache_path}")
            return True
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
    else:
        print("ℹ️ No Docling cache found")
        return False

def get_cache_size():
    """Get the size of Docling cache directory"""
    cache_path = Path.home() / ".cache" / "docling" / "models"
    if cache_path.exists():
        total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"📁 Cache size: {size_mb:.2f} MB at {cache_path}")
        return size_mb
    else:
        print("ℹ️ No cache directory found")
        return 0

def list_cached_models():
    """List all cached models in the Docling cache"""
    cache_path = Path.home() / ".cache" / "docling" / "models"
    if cache_path.exists():
        models = [d.name for d in cache_path.iterdir() if d.is_dir()]
        if models:
            print(f"📦 Cached models ({len(models)}):")
            for model in models:
                print(f"  - {model}")
        else:
            print("ℹ️ No cached models found")
        return models
    else:
        print("ℹ️ No cache directory found")
        return []

def create_processed_directory():
    """Create Processed directory if it doesn't exist"""
    processed_dir = Path("Processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for images
    images_dir = processed_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    return processed_dir, images_dir

def generate_filename(source_file, extension):
    """Generate timestamped filename for processed files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = Path(source_file).stem
    return f"{source_name}_{timestamp}.{extension}"

def extract_images_from_pdf(pdf_path, images_dir, timestamp):
    """Extract images from PDF using PyMuPDF and save with metadata"""
    try:
        doc = fitz.open(pdf_path)
        image_metadata = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    # Get image rectangle (coordinates)
                    img_rects = page.get_image_rects(xref)
                    
                    for rect_index, rect in enumerate(img_rects):
                        # Generate image filename
                        img_filename = f"image_p{page_num+1}_{img_index}_{rect_index}_{timestamp}.png"
                        img_path = images_dir / img_filename
                        
                        # Save image
                        if pix.alpha:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(str(img_path))
                        
                        # Store metadata
                        image_info = {
                            "filename": img_filename,
                            "page_number": page_num + 1,
                            "image_index": img_index,
                            "coordinates": {
                                "x0": rect.x0,
                                "y0": rect.y0,
                                "x1": rect.x1,
                                "y1": rect.y1,
                                "width": rect.width,
                                "height": rect.height
                            },
                            "size_bytes": pix.size,
                            "colorspace": pix.colorspace.name if pix.colorspace else "Unknown"
                        }
                        image_metadata.append(image_info)
                        
                        print(f"📸 Extracted: {img_filename} from page {page_num+1}")
                
                pix = None  # Free memory
        
        doc.close()
        return image_metadata
        
    except Exception as e:
        print(f"❌ Error extracting images: {e}")
        return []

def create_image_text_mapping(docling_result, image_metadata, processed_dir, timestamp):
    """Create mapping between images and text passages for chunking"""
    try:
        mapping_data = {
            "document_info": {
                "timestamp": timestamp,
                "total_pages": len(docling_result.document.pages) if hasattr(docling_result.document, 'pages') else 0,
                "total_images": len(image_metadata)
            },
            "image_text_mapping": [],
            "text_chunks": [],
            "images": image_metadata
        }
        
        # Get text content with position info from Docling
        if hasattr(docling_result.document, 'texts'):
            for text_idx, text_element in enumerate(docling_result.document.texts):
                if hasattr(text_element, 'prov') and text_element.prov:
                    for prov in text_element.prov:
                        page_num = prov.page_no
                        bbox = prov.bbox
                        
                        # Find nearby images on the same page
                        nearby_images = []
                        for img in image_metadata:
                            if img['page_number'] == page_num:
                                # Check if image is near this text 
                                img_coords = img['coordinates']
                                text_coords = {
                                    'x0': bbox.l,
                                    'y0': bbox.t,
                                    'x1': bbox.r,
                                    'y1': bbox.b
                                }
                                
                                # Calculate distance between text and image
                                distance = calculate_distance(text_coords, img_coords)
                                if distance < 200:  # Adjust threshold 
                                    nearby_images.append({
                                        'image_filename': img['filename'],
                                        'distance': distance,
                                        'relative_position': get_relative_position(text_coords, img_coords)
                                    })
                        
                        # Create text chunk with image associations
                        chunk_data = {
                            'chunk_id': f'chunk_{text_idx}_{len(mapping_data["text_chunks"])}',
                            'text': text_element.text if hasattr(text_element, 'text') else '',
                            'page_number': page_num,
                            'coordinates': {
                                'x0': bbox.l,
                                'y0': bbox.t,
                                'x1': bbox.r,
                                'y1': bbox.b
                            },
                            'associated_images': nearby_images,
                            'char_span': [prov.charspan[0], prov.charspan[1]] if hasattr(prov, 'charspan') else [0, 0]
                        }
                        mapping_data['text_chunks'].append(chunk_data)
        
        # Save mapping file
        mapping_file = processed_dir / f"image_text_mapping_{timestamp}.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"🗺️ Image-text mapping saved to: {mapping_file}")
        return mapping_data
        
    except Exception as e:
        print(f"❌ Error creating image-text mapping: {e}")
        return None

def calculate_distance(text_coords, img_coords):
    """Calculate distance between text and image bounding boxes"""
    text_center_x = (text_coords['x0'] + text_coords['x1']) / 2
    text_center_y = (text_coords['y0'] + text_coords['y1']) / 2
    img_center_x = (img_coords['x0'] + img_coords['x1']) / 2
    img_center_y = (img_coords['y0'] + img_coords['y1']) / 2
    
    return ((text_center_x - img_center_x) ** 2 + (text_center_y - img_center_y) ** 2) ** 0.5

def get_relative_position(text_coords, img_coords):
    """Determine relative position of image to text"""
    text_center_y = (text_coords['y0'] + text_coords['y1']) / 2
    img_center_y = (img_coords['y0'] + img_coords['y1']) / 2
    
    if img_center_y < text_center_y:
        return "above"
    elif img_center_y > text_center_y:
        return "below"
    else:
        return "inline"

# Check CUDA availability
cuda_available = check_cuda_availability()

# PDF pipeline configuration with GPU support
pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False  # avoid column merge issues
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # more accurate table extraction

# Configure GPU acceleration if available
if cuda_available:
    try:
        # Enable GPU acceleration for OCR and layout detection
        pipeline_options.ocr_options.use_gpu = True
        pipeline_options.table_structure_options.use_gpu = True
        print("✅ GPU acceleration enabled for OCR and table structure")
    except AttributeError:
        # Fallback if GPU options are not available in this version
        print("ℹ️ GPU options not available in this Docling version")

# Document converter
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Use the PDF file from Data directory
source_pdf = "Data/A Lightweight Hybrid Model with Location Preserving ViT for efficient food recognition.pdf"

if __name__ == "__main__":
    # Create Processed directory and images subdirectory
    processed_dir, images_dir = create_processed_directory()
    print(f"📁 Created/Using directory: {processed_dir.absolute()}")
    print(f"🖼️ Images will be saved to: {images_dir.absolute()}")
    
    # Optional: Show cache info before processing
    print("\n=== Cache Management ===")
    get_cache_size()
    list_cached_models()
    
    # Ask user if they want to clear cache before processing
    clear_cache = input("\n🗑️ Clear cache before processing? (y/N): ").lower().strip()
    if clear_cache in ['y', 'yes']:
        clear_docling_cache()
    
    print("\n=== PDF Processing ===")
    # Convert (also limit page/file size to stay within 3GB space)
    result = doc_converter.convert(
        source_pdf,
        max_num_pages=30,          # cutoff to prevent overload
        max_file_size=15_000_000   # 15MB limit
    )
    
    # Generate timestamp for consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filenames with timestamps
    json_filename = generate_filename(source_pdf, "json")
    markdown_filename = generate_filename(source_pdf, "md")
    text_filename = generate_filename(source_pdf, "txt")
    
    # Extract images from PDF
    print("\n=== Image Extraction ===")
    image_metadata = extract_images_from_pdf(source_pdf, images_dir, timestamp)
    
    # Save to JSON file in Processed directory
    json_file_path = processed_dir / json_filename
    try:
        if hasattr(result, 'document'):
            doc_dict = result.document.model_dump()
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
            print(f"📄 JSON saved to: {json_file_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")
    
    # Save to Markdown format in Processed directory
    markdown_file_path = processed_dir / markdown_filename
    try:
        if hasattr(result.document, 'export_to_markdown'):
            markdown_content = result.document.export_to_markdown()
            with open(markdown_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"📝 Markdown saved to: {markdown_file_path}")
    except Exception as e:
        print(f"Note: Markdown export not available: {e}")
    
    # Save extracted text to plain text file (FIXED VERSION)
    text_file_path = processed_dir / text_filename
    try:
        if hasattr(result, 'document'):
            # Try to get text from markdown export and clean it
            if hasattr(result.document, 'export_to_markdown'):
                markdown_content = result.document.export_to_markdown()
                # Simple markdown to text conversion
                # Remove markdown image tags
                text_content = re.sub(r'<!-- image -->', '', markdown_content)
                # Remove markdown headers
                text_content = re.sub(r'^#+\s*', '', text_content, flags=re.MULTILINE)
                # Remove extra whitespace
                text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
                
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content.strip())
                print(f"📝 Text content saved to: {text_file_path}")
            else:
                print("Warning: Could not extract text content")
    except Exception as e:
        print(f"Error saving text: {e}")
    
    # Create image-text mapping for chunking
    print("\n=== Creating Image-Text Mapping ===")
    mapping_data = create_image_text_mapping(result, image_metadata, processed_dir, timestamp)
    
    # Create processing summary
    summary_file = processed_dir / f"processing_summary_{timestamp}.txt"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Processing Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Source PDF: {source_pdf}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CUDA Available: {cuda_available}\n")
            f.write(f"\nGenerated Files:\n")
            f.write(f"- JSON: {json_filename}\n")
            f.write(f"- Markdown: {markdown_filename}\n")
            f.write(f"- Text: {text_filename}\n")
            f.write(f"- Image-Text Mapping: image_text_mapping_{timestamp}.json\n")
            
            if hasattr(result, 'document'):
                page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0
                f.write(f"\nDocument Info:\n")
                f.write(f"- Pages: {page_count}\n")
                f.write(f"- Images Extracted: {len(image_metadata)}\n")
                if mapping_data:
                    f.write(f"- Text Chunks: {len(mapping_data.get('text_chunks', []))}\n")
        
        print(f"📋 Processing summary saved to: {summary_file}")
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    # Show cache info after processing
    print("\n=== Post-Processing Cache Info ===")
    get_cache_size()
    
    # Ask user if they want to clear cache after processing
    clear_cache_after = input("\n🗑️ Clear cache after processing? (y/N): ").lower().strip()
    if clear_cache_after in ['y', 'yes']:
        clear_docling_cache()
    
    print(f"\n✅ All files saved to: {processed_dir.absolute()}")
    print(f"🖼️ Images saved to: {images_dir.absolute()}")
    if image_metadata:
        print(f"📊 Extracted {len(image_metadata)} images with coordinate mapping")
