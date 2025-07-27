import fitz  # PyMuPDF
import json
import pandas as pd
import re
from pathlib import Path
import time
import numpy as np
from utils import setup_logging, clean_text

class PDFProcessor:
    def __init__(self, config=None):
        self.logger = setup_logging()
        self.config = config or {}
        self.similarity_threshold = self.config.get('processing', {}).get('text_similarity_threshold', 0.25)
    
    def extract_pdf_features(self, pdf_path):
        """Extract text blocks with enhanced reconstruction to fix fragmentation"""
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            all_text_blocks = []
            
            for page_num, page in enumerate(doc):
                if page_num >= 50:
                    break
                
                # Multiple extraction strategies
                structured_blocks = self._extract_structured_blocks(page, page_num + 1)
                line_blocks = self._extract_line_based_blocks(page, page_num + 1)
                reconstructed_blocks = self._reconstruct_complete_text(structured_blocks, line_blocks, page)
                
                all_text_blocks.extend(reconstructed_blocks)
            
            doc.close()
            
            processing_time = time.time() - start_time
            self.logger.info(f"Enhanced extraction: {len(all_text_blocks)} blocks from {pdf_path} in {processing_time:.2f}s")
            
            return all_text_blocks
            
        except Exception as e:
            self.logger.error(f"Enhanced extraction failed: {str(e)}")
            return self._fallback_extraction(pdf_path)
    
    def _extract_structured_blocks(self, page, page_num):
        """Extract structured text blocks with span combination"""
        blocks = page.get_text("dict")
        structured = []
        
        for block in blocks["blocks"]:
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                line_text = ""
                font_info = []
                bbox_info = []
                
                for span in line["spans"]:
                    text = span['text']
                    if text.strip():
                        line_text += text
                        font_info.append({
                            'size': span['size'],
                            'flags': span['flags'],
                            'font': span['font']
                        })
                        bbox_info.append(span['bbox'])
                
                if line_text.strip() and bbox_info:
                    # Calculate combined bbox
                    min_x = min(bbox[0] for bbox in bbox_info)
                    min_y = min(bbox[1] for bbox in bbox_info)
                    max_x = max(bbox[2] for bbox in bbox_info)
                    max_y = max(bbox[3] for bbox in bbox_info)
                    
                    structured.append({
                        'text': line_text.strip(),
                        'font_info': font_info,
                        'bbox': [min_x, min_y, max_x, max_y],
                        'page': page_num,
                        'type': 'structured'
                    })
        
        return structured
    
    def _extract_line_based_blocks(self, page, page_num):
        """Extract text using line-based reconstruction"""
        try:
            # Get text blocks with better positioning
            text_dict = page.get_text("dict")
            line_blocks = []
            
            # Group text by Y position (lines)
            y_groups = {}
            
            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_bbox = line["bbox"]
                    y_center = (line_bbox[1] + line_bbox[3]) / 2
                    y_key = round(y_center / 3) * 3  # Group nearby lines
                    
                    if y_key not in y_groups:
                        y_groups[y_key] = []
                    
                    # Collect all spans in this line
                    spans_in_line = []
                    for span in line["spans"]:
                        if span["text"].strip():
                            spans_in_line.append({
                                'text': span["text"],
                                'bbox': span["bbox"],
                                'x': span["bbox"][0]
                            })
                    
                    if spans_in_line:
                        y_groups[y_key].extend(spans_in_line)
            
            # Reconstruct complete lines
            for y_key in sorted(y_groups.keys()):
                spans = sorted(y_groups[y_key], key=lambda x: x['x'])
                
                combined_text = ""
                combined_bbox = None
                
                for span in spans:
                    if combined_text:
                        combined_text += " " + span['text'].strip()
                    else:
                        combined_text = span['text'].strip()
                    
                    if combined_bbox is None:
                        combined_bbox = list(span['bbox'])
                    else:
                        combined_bbox[0] = min(combined_bbox[0], span['bbox'][0])
                        combined_bbox[1] = min(combined_bbox[1], span['bbox'][1])
                        combined_bbox[2] = max(combined_bbox[2], span['bbox'][2])
                        combined_bbox[3] = max(combined_bbox[3], span['bbox'][3])
                
                if combined_text.strip():
                    line_blocks.append({
                        'text': combined_text.strip(),
                        'bbox': combined_bbox,
                        'page': page_num,
                        'type': 'line_based'
                    })
            
            return line_blocks
            
        except Exception as e:
            self.logger.error(f"Line-based extraction failed: {str(e)}")
            return []
    
    def _reconstruct_complete_text(self, structured_blocks, line_blocks, page):
        """Choose the best text reconstruction for each position"""
        final_blocks = []
        page_height = page.rect.height
        page_width = page.rect.width
        
        # Combine all blocks with position info
        all_blocks = structured_blocks + line_blocks
        
        # Group by approximate Y position
        position_groups = {}
        for block in all_blocks:
            y_pos = round(block['bbox'][1] / 5) * 5
            if y_pos not in position_groups:
                position_groups[y_pos] = []
            position_groups[y_pos].append(block)
        
        # Choose best text for each position
        for y_pos in sorted(position_groups.keys()):
            blocks_at_pos = position_groups[y_pos]
            
            if len(blocks_at_pos) == 1:
                best_block = blocks_at_pos[0]
            else:
                # Choose the most complete text
                best_block = max(blocks_at_pos, key=lambda x: self._text_quality_score(x['text']))
            
            # Convert to final format
            text = best_block['text']
            if len(text.strip()) > 0:
                # Get font properties
                font_size = 12
                font_flags = 0
                font_name = "default"
                
                if 'font_info' in best_block and best_block['font_info']:
                    font_info = best_block['font_info'][0]
                    font_size = font_info['size']
                    font_flags = font_info['flags']
                    font_name = font_info['font']
                
                final_blocks.append({
                    'text': text,
                    'font_size': font_size,
                    'font_flags': font_flags,
                    'page': best_block['page'],
                    'bbox': best_block['bbox'],
                    'font_name': font_name,
                    'page_height': page_height,
                    'page_width': page_width
                })
        
        return final_blocks
    
    def _text_quality_score(self, text):
        """Calculate text quality score (higher = better)"""
        score = len(text)  # Base score is length
        
        # Bonus for complete words
        words = text.split()
        complete_words = [w for w in words if len(w) >= 3 and w.isalnum()]
        score += len(complete_words) * 5
        
        # Penalty for obvious fragments
        if re.search(r'[a-z][A-Z]', text):  # Mixed case breaks
            score -= 10
        
        if text.endswith(('f', 'r', 't', 'n', 'd')) and len(text) < 20:
            score -= 15
        
        # Bonus for proper sentence structure
        if text[0].isupper() and not text.endswith((',', ':')):
            score += 5
        
        return score
    
    def _fallback_extraction(self, pdf_path):
        """Fallback extraction method"""
        try:
            doc = fitz.open(pdf_path)
            all_text_blocks = []
            
            for page_num, page in enumerate(doc):
                if page_num >= 50:
                    break
                
                text_dict = page.get_text("dict")
                page_height = page.rect.height
                page_width = page.rect.width
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = clean_text(span['text'])
                                if len(text.strip()) > 0:
                                    all_text_blocks.append({
                                        'text': text,
                                        'font_size': span['size'],
                                        'font_flags': span['flags'],
                                        'page': page_num + 1,
                                        'bbox': span['bbox'],
                                        'font_name': span['font'],
                                        'page_height': page_height,
                                        'page_width': page_width
                                    })
            
            doc.close()
            return all_text_blocks
            
        except Exception as e:
            self.logger.error(f"Fallback extraction failed: {str(e)}")
            return []
    
    # Keep all your existing methods (parse_ground_truth, create_training_data, etc.)
    def parse_ground_truth(self, json_path):
        """Parse JSON ground truth labels"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            title = data.get('title', '').strip()
            headings = []
            
            for item in data.get('outline', []):
                headings.append({
                    'text': clean_text(item.get('text', '')),
                    'level': item.get('level', ''),
                    'page': item.get('page', 0) + 1
                })
            
            return title, headings
            
        except Exception as e:
            self.logger.error(f"Error parsing JSON {json_path}: {str(e)}")
            return "", []
    
    def create_training_data(self, pdf_path, json_path):
        """Create training data with ultra-flexible matching"""
        pdf_name = Path(pdf_path).name
        self.logger.info(f"Processing training pair: {pdf_name}")
        
        text_blocks = self.extract_pdf_features(pdf_path)
        if not text_blocks:
            return []
        
        title, headings = self.parse_ground_truth(json_path)
        
        training_data = []
        title_matches = 0
        heading_matches = 0
        
        heading_lookup = {}
        for heading in headings:
            page = heading['page']
            if page not in heading_lookup:
                heading_lookup[page] = []
            heading_lookup[page].append(heading)
        
        for block in text_blocks:
            label = 'BODY'
            block_text = block['text'].strip()
            
            if len(block_text) < 2:
                continue
            
            # Check title match
            if title:
                title_sim = self._ultra_flexible_similarity(block_text, title)
                if title_sim >= self.similarity_threshold:
                    label = 'TITLE'
                    title_matches += 1
            
            # Check heading matches
            block_page = block['page']
            if block_page in heading_lookup:
                for heading in heading_lookup[block_page]:
                    heading_sim = self._ultra_flexible_similarity(block_text, heading['text'])
                    if heading_sim >= self.similarity_threshold:
                        label = heading['level']
                        heading_matches += 1
                        break
            
            features = self._extract_block_features(block)
            features['label'] = label
            training_data.append(features)
        
        self.logger.info(f"Created {len(training_data)} samples: {title_matches} titles, {heading_matches} headings")
        return training_data
    
    def _ultra_flexible_similarity(self, text1, text2):
        """Ultra-flexible text similarity"""
        if not text1 or not text2:
            return 0.0
        
        str1 = str(text1).strip()
        str2 = str(text2).strip()
        
        if not str1 or not str2:
            return 0.0
        
        # Exact match
        if str1.lower() == str2.lower():
            return 1.0
        
        # Substring match
        if str1.lower() in str2.lower() or str2.lower() in str1.lower():
            return 0.9
        
        # Normalized comparison
        def normalize_ultra(text):
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\b\d+\b', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            return text
        
        norm1 = normalize_ultra(str1)
        norm2 = normalize_ultra(str2)
        
        if norm1 == norm2:
            return 0.95
        
        if norm1 in norm2 or norm2 in norm1:
            return 0.85
        
        # Word-level matching
        words1 = set(w for w in norm1.split() if len(w) > 1)
        words2 = set(w for w in norm2.split() if len(w) > 1)
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        if intersection > 0:
            jaccard = min(jaccard + 0.2, 1.0)
        
        if len(words1) == 1 and len(words2) == 1 and intersection == 1:
            return 0.8
        
        return jaccard
    
    def _extract_block_features(self, block):
        """Extract numerical features from text block"""
        text = block['text']
        bbox = block['bbox']
        
        return {
            'text': text,
            'font_size': float(block['font_size']),
            'is_bold': bool(block['font_flags'] & 2**4),
            'is_italic': bool(block['font_flags'] & 2**1),
            'page': int(block['page']),
            'text_length': len(text),
            'bbox_x': float(bbox[0]),
            'bbox_y': float(bbox[1]),
            'bbox_width': float(bbox[2] - bbox[0]),
            'bbox_height': float(bbox[3] - bbox[1]),
            'page_height': float(block['page_height']),
            'page_width': float(block['page_width']),
            'font_name': str(block['font_name'])
        }
    
    def process_multiple_pdfs(self, pdf_json_pairs, max_files=None):
        """Process multiple PDF-JSON pairs for training"""
        all_training_data = []
        
        pairs_to_process = pdf_json_pairs[:max_files] if max_files else pdf_json_pairs
        
        for i, (pdf_path, json_path) in enumerate(pairs_to_process):
            self.logger.info(f"Processing file {i+1}/{len(pairs_to_process)}: {Path(pdf_path).name}")
            
            try:
                training_data = self.create_training_data(pdf_path, json_path)
                all_training_data.extend(training_data)
                
            except Exception as e:
                self.logger.error(f"Error processing {Path(pdf_path).name}: {str(e)}")
                continue
        
        self.logger.info(f"Total training samples created: {len(all_training_data)}")
        
        if all_training_data:
            labels = [item['label'] for item in all_training_data]
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            self.logger.info("Final label distribution:")
            for label, count in sorted(label_counts.items()):
                percentage = (count / len(labels) * 100)
                self.logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        return all_training_data
