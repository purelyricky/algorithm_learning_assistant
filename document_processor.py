"""
Document Processing Module for Algorithm Learning Assistant
Handles extraction and processing of various document formats (PDF, DOCX, code files)
"""

import os
import hashlib
import uuid
from pathlib import Path
import chardet
import magic
from datetime import datetime

# PDF processing
import PyPDF2
import pdfplumber

# Word document processing
import docx
import mammoth

# OCR capabilities
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Code analysis
import ast
import keyword

def detect_file_type(file_path):
    """
    Detects file type using python-magic
    Args: file_path - path to the file
    Returns: tuple (mime_type, file_extension, is_supported)
    """
    try:
        mime_type = magic.from_file(file_path, mime=True)
        file_extension = Path(file_path).suffix.lower()
        
        supported_types = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'text/plain': 'txt',
            'text/x-python': 'py',
            'text/x-java-source': 'java',
            'text/markdown': 'md',
            'application/json': 'json'
        }
        
        is_supported = mime_type in supported_types
        return mime_type, file_extension, is_supported
        
    except Exception as e:
        print(f"Error detecting file type: {e}")
        return None, None, False

def calculate_file_hash(file_path):
    """
    Calculates SHA-256 hash of file to detect duplicates
    Args: file_path - path to the file
    Returns: hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating file hash: {e}")
        return None

def extract_text_from_pdf(file_path):
    """
    Extracts text from PDF files using multiple methods
    Args: file_path - path to PDF file
    Returns: tuple (text_content, metadata)
    """
    text_content = ""
    metadata = {"pages": 0, "extraction_method": "unknown"}
    
    try:
        # Method 1: Try pdfplumber first (better for complex layouts)
        with pdfplumber.open(file_path) as pdf:
            metadata["pages"] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text
            
            if text_content.strip():
                metadata["extraction_method"] = "pdfplumber"
                return text_content, metadata
    
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")
    
    try:
        # Method 2: Fallback to PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata["pages"] = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text
            
            if text_content.strip():
                metadata["extraction_method"] = "PyPDF2"
                return text_content, metadata
                
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
    
    # Method 3: OCR as last resort (if available and enabled)
    if OCR_AVAILABLE and not text_content.strip():
        try:
            text_content = extract_pdf_with_ocr(file_path)
            if text_content:
                metadata["extraction_method"] = "OCR"
                return text_content, metadata
        except Exception as e:
            print(f"OCR extraction failed: {e}")
    
    return text_content, metadata

def extract_pdf_with_ocr(file_path):
    """
    Extracts text from PDF using OCR (for scanned documents)
    Args: file_path - path to PDF file
    Returns: extracted text string
    """
    if not OCR_AVAILABLE:
        return ""
    
    try:
        # Convert PDF pages to images and apply OCR
        import fitz  # PyMuPDF for PDF to image conversion
        
        doc = fitz.open(file_path)
        text_content = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Apply OCR to the image
            page_text = pytesseract.image_to_string(img)
            text_content += f"\n--- Page {page_num + 1} (OCR) ---\n"
            text_content += page_text
        
        doc.close()
        return text_content
        
    except Exception as e:
        print(f"OCR processing failed: {e}")
        return ""

def extract_text_from_docx(file_path):
    """
    Extracts text from Word documents
    Args: file_path - path to DOCX file
    Returns: tuple (text_content, metadata)
    """
    text_content = ""
    metadata = {"paragraphs": 0, "extraction_method": "unknown"}
    
    try:
        # Method 1: Using mammoth for better formatting preservation
        with open(file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            text_content = result.value
            metadata["extraction_method"] = "mammoth"
            metadata["paragraphs"] = len(text_content.split('\n'))
            
            if text_content.strip():
                return text_content, metadata
                
    except Exception as e:
        print(f"Mammoth extraction failed: {e}")
    
    try:
        # Method 2: Fallback to python-docx
        doc = docx.Document(file_path)
        paragraphs = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)
        
        text_content = '\n'.join(paragraphs)
        metadata["extraction_method"] = "python-docx"
        metadata["paragraphs"] = len(paragraphs)
        
        return text_content, metadata
        
    except Exception as e:
        print(f"python-docx extraction failed: {e}")
        return "", metadata

def extract_text_from_code(file_path, file_extension):
    """
    Extracts and analyzes code files
    Args: file_path - path to code file, file_extension - file extension
    Returns: tuple (text_content, metadata)
    """
    try:
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result['encoding'] or 'utf-8'
        
        # Read file with detected encoding
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        metadata = {
            "encoding": encoding,
            "lines": len(content.split('\n')),
            "language": file_extension.lstrip('.'),
            "extraction_method": "direct_read"
        }
        
        # Add language-specific analysis
        if file_extension == '.py':
            metadata.update(analyze_python_code(content))
        elif file_extension == '.java':
            metadata.update(analyze_java_code(content))
        
        # Format content with metadata header
        formatted_content = f"""
Language: {metadata['language']}
Lines of code: {metadata['lines']}
Encoding: {metadata['encoding']}

--- Code Content ---
{content}
        """
        
        return formatted_content.strip(), metadata
        
    except Exception as e:
        print(f"Code extraction failed: {e}")
        return "", {"error": str(e)}

def analyze_python_code(content):
    """
    Analyzes Python code for functions, classes, imports
    Args: content - Python source code
    Returns: analysis metadata dictionary
    """
    analysis = {
        "functions": [],
        "classes": [],
        "imports": [],
        "docstrings": []
    }
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis["functions"].append(node.name)
                if ast.get_docstring(node):
                    analysis["docstrings"].append(f"Function {node.name}: {ast.get_docstring(node)[:100]}")
            
            elif isinstance(node, ast.ClassDef):
                analysis["classes"].append(node.name)
                if ast.get_docstring(node):
                    analysis["docstrings"].append(f"Class {node.name}: {ast.get_docstring(node)[:100]}")
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis["imports"].append(node.module)
        
        analysis["function_count"] = len(analysis["functions"])
        analysis["class_count"] = len(analysis["classes"])
        analysis["import_count"] = len(analysis["imports"])
        
    except SyntaxError as e:
        analysis["syntax_error"] = str(e)
    except Exception as e:
        analysis["analysis_error"] = str(e)
    
    return analysis

def analyze_java_code(content):
    """
    Basic Java code analysis (simpler than Python AST)
    Args: content - Java source code
    Returns: analysis metadata dictionary
    """
    analysis = {
        "classes": [],
        "methods": [],
        "imports": []
    }
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Simple pattern matching for Java constructs
        if line.startswith('import '):
            import_name = line.replace('import ', '').replace(';', '').strip()
            analysis["imports"].append(import_name)
        
        elif 'class ' in line and '{' in line:
            # Extract class name
            parts = line.split('class ')
            if len(parts) > 1:
                class_part = parts[1].split('{')[0].split()[0]
                analysis["classes"].append(class_part)
        
        elif ('public ' in line or 'private ' in line) and '(' in line and ')' in line:
            # Likely a method declaration
            if 'class ' not in line:  # Avoid class declarations
                method_part = line.split('(')[0].split()
                if len(method_part) > 0:
                    method_name = method_part[-1]
                    analysis["methods"].append(method_name)
    
    analysis["class_count"] = len(analysis["classes"])
    analysis["method_count"] = len(analysis["methods"])
    analysis["import_count"] = len(analysis["imports"])
    
    return analysis

def chunk_document_content(content, metadata, chunk_size=1000):
    """
    Splits document content into chunks for vector storage
    Args: content - extracted text, metadata - document metadata, chunk_size - target chunk size
    Returns: list of chunk dictionaries
    """
    chunks = []
    
    # Simple sentence-based chunking
    sentences = content.split('. ')
    current_chunk = ""
    chunk_index = 0
    
    for sentence in sentences:
        # Add sentence to current chunk if it fits
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            # Save current chunk and start new one
            if current_chunk.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": current_chunk.strip(),
                    "char_count": len(current_chunk),
                    "metadata": metadata.copy()
                })
                chunk_index += 1
            
            current_chunk = sentence + ". "
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "chunk_index": chunk_index,
            "content": current_chunk.strip(),
            "char_count": len(current_chunk),
            "metadata": metadata.copy()
        })
    
    return chunks

def process_document(file_path):
    """
    Main document processing function
    Args: file_path - path to the document file
    Returns: dictionary with processed document data
    """
    if not os.path.exists(file_path):
        return {"error": "File not found", "success": False}
    
    # Detect file type
    mime_type, file_extension, is_supported = detect_file_type(file_path)
    
    if not is_supported:
        return {
            "error": f"Unsupported file type: {mime_type}",
            "success": False
        }
    
    # Calculate file hash for duplicate detection
    file_hash = calculate_file_hash(file_path)
    if not file_hash:
        return {"error": "Could not calculate file hash", "success": False}
    
    # Extract text based on file type
    text_content = ""
    extraction_metadata = {}
    
    try:
        if file_extension == '.pdf':
            text_content, extraction_metadata = extract_text_from_pdf(file_path)
        
        elif file_extension == '.docx':
            text_content, extraction_metadata = extract_text_from_docx(file_path)
        
        elif file_extension in ['.py', '.java', '.txt', '.md', '.json']:
            text_content, extraction_metadata = extract_text_from_code(file_path, file_extension)
        
        else:
            return {"error": f"No processor for {file_extension}", "success": False}
        
        if not text_content.strip():
            return {"error": "No text content extracted", "success": False}
        
        # Generate chunks
        chunks = chunk_document_content(text_content, extraction_metadata)
        
        # Prepare final document data
        document_data = {
            "success": True,
            "document_id": str(uuid.uuid4()),
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "file_hash": file_hash,
            "file_size": os.path.getsize(file_path),
            "mime_type": mime_type,
            "file_extension": file_extension,
            "processed_at": datetime.now().isoformat(),
            "extraction_metadata": extraction_metadata,
            "total_chunks": len(chunks),
            "total_characters": len(text_content),
            "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
            "chunks": chunks
        }
        
        return document_data
        
    except Exception as e:
        return {
            "error": f"Processing failed: {str(e)}",
            "success": False
        }

def get_supported_file_types():
    """
    Returns list of supported file types and descriptions
    """
    return {
        'pdf': 'PDF documents (with OCR support for scanned docs)',
        'docx': 'Microsoft Word documents',
        'txt': 'Plain text files',
        'md': 'Markdown files',
        'py': 'Python source code files',
        'java': 'Java source code files',
        'json': 'JSON data files'
    }

if __name__ == "__main__":
    # Test document processing
    test_file = input("Enter path to test file: ").strip()
    if test_file:
        result = process_document(test_file)
        
        if result["success"]:
            print(f"✅ Successfully processed: {result['filename']}")
            print(f"📄 File type: {result['file_extension']}")
            print(f"📊 Chunks created: {result['total_chunks']}")
            print(f"📝 Characters: {result['total_characters']}")
            print(f"🔍 Preview: {result['content_preview']}")
        else:
            print(f"❌ Processing failed: {result['error']}")