"""
Document processor service that coordinates the document processing pipeline
"""
import os
import logging
import traceback
import threading
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

from document_processor.core.classification.classifier import DocumentClassifier
from document_processor.core.extraction.text_extractor import DocumentExtractorFactory, extract_document_content
from document_processor.core.extraction.targeted_extractor import TargetedExtractor
from document_processor.core.information.financial_extractor import FinancialEntityExtractor
from document_processor.utils.custom_exceptions import ProcessingError, DocumentNotSupportedError
from document_processor.utils.file_utils import get_file_extension, is_valid_document
from document_processor.core.processing_modes import ProcessingMode
from document_processor.utils.validation import validate_file

logger = logging.getLogger(__name__)

class ProcessorService:
    """
    Service for coordinating the document processing pipeline,
    including classification, text extraction, and entity extraction.
    """
    
    def __init__(self, config):
        """
        Initialize the processor service
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.upload_folder = config.get('UPLOAD_FOLDER', 'uploads')
        self.models_folder = config.get('MODELS_FOLDER', 'models')
        
        # Initialize bulk extraction components
        self.classifier = DocumentClassifier()
        self.text_extractor = DocumentExtractorFactory  # This is a factory class, not an instance
        self.financial_extractor = FinancialEntityExtractor()
        
        # Initialize targeted extraction components
        self.targeted_extractor = TargetedExtractor()
        
        logger.info("Document processor service initialized")
    
    def process_document(self, file_path, mode=ProcessingMode.BULK):
        """
        Process a document through the pipeline
        
        Args:
            file_path (str): Path to the document
            mode (ProcessingMode): Processing mode to use (bulk or targeted)
            
        Returns:
            dict: Processing results including classification, extracted text, and entities
        """
        try:
            logger.info(f"Processing document {file_path} using {mode.value} mode")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise DocumentNotSupportedError(f"File does not exist: {file_path}")
            
            # Validate document
            result, error = validate_file(file_path)
            if not result:
                raise error
            
            # First classify the document (needed for both modes)
            # We need to extract some text for this
            extractor = self.text_extractor.get_extractor(file_path)
            extraction_result = extractor.extract(file_path)
            text = extraction_result.get('text', '')
            doc_type, confidence = self.classifier.classify(text)
            
            # Process document based on mode
            if mode == ProcessingMode.TARGETED:
                return self._process_with_targeted(file_path, doc_type, confidence)
            else:
                return self._process_with_bulk(file_path, text, doc_type, confidence, extraction_result)
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise ProcessingError(f"Document processing failed: {str(e)}")
    
    def _process_with_bulk(self, file_path, text, doc_type, confidence, extraction_result):
        """
        Process document using bulk extraction (docling + PyMuPDF)
        
        Args:
            file_path (str): Path to the document
            text (str): Already extracted text
            doc_type (str): Document type
            confidence (float): Classification confidence
            extraction_result (dict): Result of text extraction
            
        Returns:
            dict: Processing results
        """
        # Extract financial entities
        entities = self.financial_extractor.extract_entities(text)
        
        # Get tables from extraction result
        tables = extraction_result.get('tables', [])
        
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "doc_type": doc_type,
            "classification_confidence": confidence,
            "text": text,  # Include full text in bulk mode
            "entities": entities,
            "tables": tables,
            "processing_mode": ProcessingMode.BULK.value
        }
    
    def _process_with_targeted(self, file_path, doc_type, confidence):
        """
        Process document using targeted extraction (doctr)
        Only extracts specific fields, does NOT include full text
        
        Args:
            file_path (str): Path to the document
            doc_type (str): Document type
            confidence (float): Classification confidence
            
        Returns:
            dict: Processing results with only targeted fields
        """
        # Use targeted extraction based on document type
        targeted_results = self.targeted_extractor.extract(file_path, doc_type)
        
        # Format entities for display
        entities = []
        for field, value in targeted_results.get('extracted_fields', {}).items():
            if value:
                # Determine entity type based on field name
                entity_type = self._map_field_to_entity_type(field)
                entities.append({
                    "type": entity_type,
                    "text": value,
                    "field": field
                })
        
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "doc_type": doc_type,
            "classification_confidence": confidence,
            "text": "",  # No full text in targeted mode
            "entities": entities,
            "tables": [],  # No tables in targeted mode
            "extracted_fields": targeted_results.get('extracted_fields', {}),
            "processing_mode": ProcessingMode.TARGETED.value
        }
    
    def _map_field_to_entity_type(self, field_name):
        """
        Map field name to entity type
        
        Args:
            field_name (str): Field name from targeted extraction
            
        Returns:
            str: Entity type
        """
        # Map common field names to entity types
        field_to_entity = {
            # Financial fields
            'net_income': 'AMOUNT',
            'total_income': 'AMOUNT',
            'ordinary_income': 'AMOUNT',
            'interest_income': 'AMOUNT',
            'dividend_income': 'AMOUNT',
            'royalty_income': 'AMOUNT',
            'capital_gain': 'AMOUNT',
            'total_amount': 'AMOUNT',
            'total_assets': 'AMOUNT',
            'total_liabilities': 'AMOUNT',
            'net_worth': 'AMOUNT',
            'total_revenue': 'AMOUNT',
            'wages': 'AMOUNT',
            'federal_tax': 'AMOUNT',
            'tax_owed': 'AMOUNT',
            'amount': 'AMOUNT',
            'total': 'AMOUNT',
            
            # Date fields
            'tax_year': 'DATE',
            'statement_date': 'DATE',
            'invoice_date': 'DATE',
            'due_date': 'DATE',
            'date': 'DATE',
            
            # Name/entity fields
            'partner_name': 'ENTITY',
            'taxpayer_name': 'ENTITY',
            'employee_name': 'ENTITY',
            'employer_name': 'ENTITY',
            'customer_name': 'ENTITY',
            'vendor_name': 'ENTITY',
            
            # Identifier fields
            'ein': 'ID',
            'taxpayer_ssn': 'ID',
            'employee_ssn': 'ID',
            'employer_ein': 'ID',
            'invoice_number': 'ID',
            'account_number': 'ID',
            'reference': 'ID',
            
            # Address fields
            'partner_address': 'ADDRESS',
            'recipient_address': 'ADDRESS',
        }
        
        # Default to OTHER if not found
        return field_to_entity.get(field_name, 'OTHER')