import os
import pdfplumber
from typing import List, Dict, Any

def save_uploaded_file(uploaded_file, target_dir: str = "data/uploaded_pdfs") -> str:
    """
    Saves an uploaded file from Streamlit to the target directory.
    
    Args:
        uploaded_file: The Streamlit UploadedFile object.
        target_dir: Directory where the file should be saved.
        
    Returns:
        The path to the saved file.
    """
    try:
        # Create directories if they do not exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Preserve original filename
        file_path = os.path.join(target_dir, uploaded_file.name)
        
        # Write the file content
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    except Exception as e:
        raise RuntimeError(f"Error saving file {uploaded_file.name}: {str(e)}")

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF file page by page using pdfplumber.
    
    Args:
        file_path: Absolute or relative path to the PDF file.
        
    Returns:
        A list of dictionaries, one for each page with non-empty text:
        {
            "source": PDF filename,
            "page": page number (1-indexed),
            "text": extracted page text
        }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")
        
    extracted_pages = []
    filename = os.path.basename(file_path)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # Check if PDF has pages
            if not pdf.pages:
                raise ValueError(f"The PDF file '{filename}' has no pages.")
                
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    # Skip pages with empty text
                    if text and text.strip():
                        extracted_pages.append({
                            "source": filename,
                            "page": page_idx,
                            "text": text.strip()
                        })
                except Exception as page_err:
                    # Specific page parsing failed
                    raise RuntimeError(f"Failed to extract text from page {page_idx} of '{filename}': {str(page_err)}")
                    
    except FileNotFoundError as fnf:
        raise fnf
    except ValueError as val_err:
        raise val_err
    except Exception as e:
        # Catch corrupted PDFs and other general PDF open/parse failures
        raise ValueError(f"Failed to open or parse PDF file '{filename}'. It may be corrupted or invalid. Error: {str(e)}")
        
    return extracted_pages
