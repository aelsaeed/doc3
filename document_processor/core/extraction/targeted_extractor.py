"""
Targeted extraction module using doctr for locating keywords and extracting specific fields
"""
import os
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional
import re
import torch
import numpy as np
from PIL import Image

# Import doctr
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

from document_processor.utils.custom_exceptions import (
    FileTypeError, FileReadError, TextExtractionError, EmptyTextError
)
from document_processor.utils.gpu_utils import check_gpu_availability
from document_processor.utils.validation import validate_file

logger = logging.getLogger(__name__)

class TargetedExtractor:
    """
    Extractor that ONLY scans for specific keywords and extracts their values
    """
    
    # Define keywords to look for in documents based on document type
    # The key is the document type, the value is a dict of field name -> search keywords
    DOCUMENT_KEYWORDS = {
        "K1 (Schedule K-1)": {
            "partner_name": ["partner's name", "partner name"],
            "partner_address": ["partner's address", "address"],
            "ein": ["ein", "employer identification number"],
            "ordinary_income": ["ordinary business income", "ordinary income", "net income"],
            "interest_income": ["interest income", "portfolio interest"],
            "dividend_income": ["dividend", "qualified dividends"],
            "royalty_income": ["royalty", "royalties"],
            "capital_gain": ["capital gain", "net short-term capital gain", "net long-term capital gain"],
            "tax_year": ["tax year", "taxable year"],
        },
        "Tax Return": {
            "taxpayer_name": ["taxpayer name", "your first name", "your name"],
            "taxpayer_ssn": ["social security number", "ssn", "taxpayer ssn"],
            "filing_status": ["filing status", "single", "married filing jointly"],
            "total_income": ["total income", "adjusted gross income"],
            "tax_owed": ["total tax", "amount you owe"],
            "tax_year": ["tax year", "form 1040", "u.s. individual income tax return"],
        },
        "Financial Statement": {
            "total_assets": ["total assets", "assets"],
            "total_liabilities": ["total liabilities", "liabilities"],
            "net_worth": ["net worth", "equity", "shareholder equity"],
            "net_income": ["net income", "income"],
            "total_revenue": ["total revenue", "revenue", "sales"],
            "statement_date": ["as of", "period ended", "year ended"],
        },
        "Invoice": {
            "invoice_number": ["invoice number", "invoice #", "invoice no"],
            "invoice_date": ["invoice date", "date", "issued on"],
            "due_date": ["due date", "payment due", "due by"],
            "total_amount": ["total amount", "total", "amount due", "balance due"],
            "customer_name": ["bill to", "customer", "client"],
            "vendor_name": ["from", "vendor", "supplier"],
        },
        "W1 (Form W-1)": {
            "employee_name": ["employee's name", "employee name"],
            "employee_ssn": ["employee's ssn", "social security number"],
            "employer_name": ["employer's name", "employer name"],
            "employer_ein": ["employer's ein", "ein"],
            "wages": ["wages, tips, other compensation", "wages"],
            "federal_tax": ["federal income tax withheld", "federal tax"],
            "tax_year": ["tax year", "year", "20"],
        },
        "W2 (Form W-2)": {
            "employee_name": ["employee's name", "employee name"],
            "employee_ssn": ["employee's ssn", "social security number"],
            "employer_name": ["employer's name", "employer name"],
            "employer_ein": ["employer identification number", "employer's ein", "ein"],
            "wages": ["wages, tips, other compensation", "wages", "box 1"],
            "federal_tax": ["federal income tax withheld", "federal tax", "box 2"],
            "social_security_wages": ["social security wages", "box 3"],
            "social_security_tax": ["social security tax withheld", "box 4"],
            "medicare_wages": ["medicare wages and tips", "box 5"],
            "medicare_tax": ["medicare tax withheld", "box 6"],
            "state_wages": ["state wages", "box 16"],
            "state_tax": ["state income tax", "box 17"],
            "tax_year": ["tax year", "year", "20"],
        }
    }
    
    # Common keywords to look for in any document
    COMMON_KEYWORDS = {
        "date": ["date:", "dated:", "as of:"],
        "total": ["total:", "total amount:", "sum:", "balance:"],
        "account_number": ["account:", "account no:", "account number:"],
        "reference": ["ref:", "reference:", "reference no:"],
        "amount": ["amount:", "payment:", "fee:"],
    }
    
    def __init__(self):
        """Initialize targeted extractor"""
        # Initialize device
        self.device = check_gpu_availability()
        logger.info(f"Targeted extractor initialized with device: {self.device}")
        
        # Initialize doctr if available
        self.doctr_model = None
        if DOCTR_AVAILABLE:
            try:
                logger.info("Initializing doctr OCR model")
                self.doctr_model = ocr_predictor(
                    det_arch='db_resnet50',
                    reco_arch='crnn_vgg16_bn',
                    pretrained=True
                )
                # Move model to device
                self.doctr_model.det_predictor.model = self.doctr_model.det_predictor.model.to(self.device)
                self.doctr_model.reco_predictor.model = self.doctr_model.reco_predictor.model.to(self.device)
                logger.info("doctr OCR model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize doctr OCR model: {str(e)}")
                self.doctr_model = None
        else:
            logger.warning("doctr not available. Install with 'pip install doctr-io'")
    
    def extract(self, file_path: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Extract ONLY specific targeted information from document
        
        Args:
            file_path (str): Path to document file
            doc_type (str, optional): Document type for targeted extraction
            
        Returns:
            Dict[str, Any]: Dictionary with extracted fields
        """
        try:
            # Validate file
            result, error = validate_file(file_path)
            if not result:
                raise error
            
            # Check if we have the required models
            if self.doctr_model is None:
                logger.warning("doctr model not available, using basic extraction")
                return self._extract_using_basic(file_path, doc_type)
            
            # Load document with doctr
            logger.info(f"Loading document with doctr: {file_path}")
            doc = DocumentFile.from_pdf(file_path) if file_path.lower().endswith('.pdf') \
                else DocumentFile.from_images(file_path)
            
            # Process document
            result = self.doctr_model(doc)
            
            # Perform targeted extraction based on document type
            extracted_data = {}
            
            # Get document specific keywords
            if doc_type and doc_type in self.DOCUMENT_KEYWORDS:
                # Use document-specific keywords
                keywords = self.DOCUMENT_KEYWORDS[doc_type]
                extracted_data = self._extract_targeted_data(result, keywords)
                
            # Also look for common keywords in all documents
            common_data = self._extract_targeted_data(result, self.COMMON_KEYWORDS)
            
            # Merge document-specific and common keywords
            for field, value in common_data.items():
                if field not in extracted_data:
                    extracted_data[field] = value
            
            return {
                "extracted_fields": extracted_data,
                "document_type": doc_type
            }
            
        except (FileTypeError, FileReadError) as e:
            # Re-raise known file-related exceptions
            raise
        except Exception as e:
            logger.error(f"Error in targeted extraction for {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise TextExtractionError(file_path, f"Targeted extraction failed: {str(e)}")
    
    def _extract_using_basic(self, file_path: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Extract using basic regex patterns when doctr is not available
        
        Args:
            file_path (str): Path to document file
            doc_type (str, optional): Document type for targeted extraction
            
        Returns:
            Dict[str, Any]: Dictionary with extracted fields
        """
        try:
            import fitz  # PyMuPDF
            
            # Extract text
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            
            if not text.strip():
                raise EmptyTextError(file_path)
            
            # Extract using regex patterns
            extracted_data = {}
            
            # Get document specific keywords
            if doc_type and doc_type in self.DOCUMENT_KEYWORDS:
                keywords = self.DOCUMENT_KEYWORDS[doc_type]
                for field, search_terms in keywords.items():
                    for term in search_terms:
                        # Look for the term followed by a value
                        pattern = re.compile(f"{re.escape(term)}\\s*[:\\-]?\\s*([^\\n\\r:]*)", re.IGNORECASE)
                        matches = pattern.search(text)
                        if matches and matches.group(1).strip():
                            extracted_data[field] = matches.group(1).strip()
                            break
            
            # Also look for common keywords
            for field, search_terms in self.COMMON_KEYWORDS.items():
                if field not in extracted_data:  # Only add if not already found
                    for term in search_terms:
                        pattern = re.compile(f"{re.escape(term)}\\s*[:\\-]?\\s*([^\\n\\r:]*)", re.IGNORECASE)
                        matches = pattern.search(text)
                        if matches and matches.group(1).strip():
                            extracted_data[field] = matches.group(1).strip()
                            break
            
            return {
                "extracted_fields": extracted_data,
                "document_type": doc_type
            }
                
        except Exception as e:
            logger.error(f"Error in basic extraction for {file_path}: {str(e)}")
            raise TextExtractionError(file_path, f"Basic extraction failed: {str(e)}")
    
    def _extract_targeted_data(self, result, keywords: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Extract targeted data based on keywords
        
        Args:
            result: doctr OCR result
            keywords (Dict[str, List[str]]): Dictionary mapping field names to keyword lists
            
        Returns:
            Dict[str, str]: Dictionary with extracted data
        """
        extracted_data = {}
        
        # Process each page
        for page_idx, page in enumerate(result.pages):
            logger.debug(f"--- Processing Page {page_idx + 1}/{len(result.pages)} ---")
            page_text = ""
            
            # Get all text blocks with their positions
            blocks_with_positions = []
            for block_idx, block in enumerate(page.blocks):
                block_text = ""
                block_coords = []
                
                for line in block.lines:
                    line_text = " ".join(word.value for word in line.words)
                    block_text += line_text + " "
                    
                    # Store coordinates of first and last word in line
                    if line.words:
                        first_word = line.words[0]
                        last_word = line.words[-1]
                        block_coords.append(first_word.geometry)
                        block_coords.append(last_word.geometry)
                
                # Calculate block bbox from all word bboxes
                if block_coords:
                    x_min = min(coord[0][0] for coord in block_coords)
                    y_min = min(coord[0][1] for coord in block_coords)
                    x_max = max(coord[1][0] for coord in block_coords)
                    y_max = max(coord[1][1] for coord in block_coords)
                    bbox = ((x_min, y_min), (x_max, y_max))
                else:
                    bbox = ((0, 0), (1, 1))  # Default if no coords
                
                blocks_with_positions.append({
                    "text": block_text.strip(),
                    "bbox": bbox,
                    "index": block_idx
                })
                
                page_text += block_text + "\n"
            
            # For each keyword set, find matching blocks and extract data
            for field_name, keyword_list in keywords.items():
                if field_name in extracted_data:
                    continue  # Already found this field
                    
                for keyword in keyword_list:
                    logger.debug(f"  Searching for keyword 	'{keyword}	' for field 	'{field_name}	'")
                    matched_blocks = self._find_blocks_with_keyword(blocks_with_positions, keyword)
                    
                    if matched_blocks:
                        logger.debug(f"    Found {len(matched_blocks)} block(s) matching keyword 	'{keyword}'")
                        # Sort by confidence/relevance if needed
                        matched_blocks.sort(key=lambda x: x["relevance"], reverse=True)
                        best_match = matched_blocks[0]
                        best_match_text_preview = best_match["text"][:100]
                        logger.debug(f"      Best match block text: \t\"{best_match_text_preview}...\"")
                        
                        # Extract the value (usually in the same or next block)
                        value = self._extract_value_from_context(
                            blocks_with_positions, 
                            best_match,
                            field_name
                        )
                        
                        if value and field_name not in extracted_data:
                            logger.debug(f"        Extracted value 	\"{value}	\" for field 	\"{field_name}	\"")
                            extracted_data[field_name] = value
                            break  # Found a value for this field, move to next field
                        else:
                            logger.debug(f"        Could not extract value for field 	\"{field_name}	\" from keyword 	\"{keyword}	\"")
        
        return extracted_data
    
    def _find_blocks_with_keyword(self, blocks, keyword):
        """
        Find blocks containing a keyword
        
        Args:
            blocks (List[Dict]): List of blocks with text and position
            keyword (str): Keyword to search for
            
        Returns:
            List[Dict]: List of matched blocks with relevance score
        """
        matches = []
        keyword = keyword.lower()
        
        for block in blocks:
            text = block["text"].lower()
            if keyword in text:
                # Calculate relevance based on position of keyword in text
                position = text.find(keyword)
                length = len(text)
                # Higher relevance if keyword is at beginning of text
                relevance = 1.0 - (position / max(length, 1))
                
                matches.append({
                    **block,
                    "relevance": relevance,
                    "keyword": keyword,
                    "keyword_pos": position
                })
        
        return matches
    
    def _extract_value_from_context(self, blocks, matched_block, field_name):
        """
        Extract value from context based on matched keyword block
        
        Args:
            blocks (List[Dict]): All blocks in the page
            matched_block (Dict): Block containing the keyword
            field_name (str): Name of the field to extract
            
        Returns:
            str: Extracted value or None if not found
        """
        block_text = matched_block["text"]
        keyword = matched_block["keyword"]
        keyword_pos = matched_block["keyword_pos"]
        
        # Method 1: Extract value from the same block (after keyword)
        after_keyword = block_text[keyword_pos + len(keyword):].strip()
        if after_keyword:
            # Try to extract from text after keyword in same block
            value = self._extract_value_by_field_type(after_keyword, field_name)
            if value:
                return value
        
        # Method 2: Look for value in the next block or nearby blocks
        block_idx = matched_block["index"]
        for i in range(1, 3):  # Check next 2 blocks
            next_idx = block_idx + i
            if next_idx < len(blocks):
                next_block = blocks[next_idx]
                value = self._extract_value_by_field_type(next_block["text"], field_name)
                if value:
                    return value
        
        # Method 3: For some fields, look for patterns in the text
        # This is a simplified approach - in a real implementation, you would
        # use more sophisticated NER or pattern matching based on field type
        return None
    
    def _extract_value_by_field_type(self, text, field_name):
        """
        Extract value based on field type using regex or other methods
        
        Args:
            text (str): Text to extract from
            field_name (str): Type of field to extract
            
        Returns:
            str: Extracted value or None
        """
        # Strip punctuation from beginning
        text = re.sub(r'^[:\s]*', '', text)
        
        # Different extraction methods based on field type
        if "amount" in field_name or "income" in field_name or "revenue" in field_name or "assets" in field_name:
            # Look for currency values
            matches = re.search(r'[$]?[\d,]+\.?\d*', text)
            if matches:
                return matches.group(0)
                
        elif "date" in field_name:
            # Look for dates in various formats
            date_patterns = [
                r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',  # MM/DD/YYYY or similar
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',  # Month DD, YYYY
                r'\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}'  # YYYY-MM-DD
            ]
            
            for pattern in date_patterns:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches:
                    return matches.group(0)
        
        elif "name" in field_name:
            # Just take the first line, assuming it's a name
            name = text.split('\n')[0].strip()
            return name if name else None
            
        elif "address" in field_name:
            # More restrictive address extraction to prevent full text capture
            # Look for common address patterns and limit length
            lines = text.split('\n')
            # Take only first 3 lines and limit total length to 150 chars
            address_lines = []
            for i in range(min(3, len(lines))):
                line = lines[i].strip()
                # Skip if line is too long (likely not an address component)
                if len(line) > 80:
                    continue
                # Skip if line contains too many words (likely not an address component)
                if len(line.split()) > 12:
                    continue
                address_lines.append(line)
            
            address = '\n'.join(address_lines).strip()
            # Limit total address length
            address = address[:150] if len(address) > 150 else address
            return address if address else None
            
        elif "ein" in field_name or "ssn" in field_name:
            # Look for EIN or SSN patterns
            patterns = [
                r'\d{2}-\d{7}',  # EIN: XX-XXXXXXX
                r'\d{3}-\d{2}-\d{4}'  # SSN: XXX-XX-XXXX
            ]
            
            for pattern in patterns:
                matches = re.search(pattern, text)
                if matches:
                    return matches.group(0)
        
        # For other types, just take the first chunk of text
        if text:
            # Take first sentence or up to 50 chars
            value = text.split('.')[0].strip()
            return value[:50] if len(value) > 50 else value
        
        return None