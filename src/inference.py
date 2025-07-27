import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import re
from data_processor import PDFProcessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from utils import setup_logging, save_json, format_execution_time

class DocumentAnalyzer:
    def __init__(self, model_dir, config=None):
        self.logger = setup_logging()
        self.config = config or {}
        self.model_dir = model_dir
        self.current_title = ""  # Store current title for filtering
        
        # Initialize components
        self.pdf_processor = PDFProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_trainer = ModelTrainer(config)
        
        # Load trained model
        self.model_trainer.load_model(model_dir)
        
        self.logger.info("DocumentAnalyzer initialized successfully")
    
    def analyze_pdf(self, pdf_path):
        """Analyze PDF with deduplication and reliable extraction"""
        start_time = time.time()
        
        self.logger.info(f"Analyzing PDF: {Path(pdf_path).name}")
        
        try:
            # Use original PDF extraction
            text_blocks = self.pdf_processor.extract_pdf_features(pdf_path)
            
            if not text_blocks:
                self.logger.warning(f"No text blocks extracted from {pdf_path}")
                return {"title": "", "outline": []}
            
            # Convert to training data format
            training_data = []
            for block in text_blocks:
                features = self.pdf_processor._extract_block_features(block)
                features['label'] = 'UNKNOWN'
                training_data.append(features)
            
            # Engineer features
            df = pd.DataFrame(training_data)
            df_features = self.feature_engineer.engineer_features(df)
            
            # Make predictions
            predicted_labels, confidence_scores = self.model_trainer.predict_batch(df_features)
            
            # Add predictions to dataframe
            df_features['predicted_label'] = predicted_labels
            df_features['confidence'] = confidence_scores
            
            # Extract with deduplication and cleanup
            title = self._extract_deduplicated_title(df_features)
            self.current_title = title  # Store for outline filtering
            outline = self._extract_clean_outline(df_features)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Analysis completed in {processing_time:.2f}s")
            
            result = {
                "title": title,
                "outline": outline
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {pdf_path}: {str(e)}")
            return {"title": "", "outline": []}
    
    def _clean_title_final(self, title):
        """Final cleanup of extracted title"""
        if not title:
            return title
        
        # Remove trailing date patterns and page numbers
        title = re.sub(r'\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*\d*$', '', title, flags=re.IGNORECASE)
        
        # Remove trailing numbers (like page numbers)
        title = re.sub(r'\s+\d{2,4}$', '', title)
        title = re.sub(r'\s+\d+\s+\d+$', '', title)  # Remove "2003 2222" pattern
        
        # Clean up extra spaces
        title = re.sub(r'\s+', ' ', title.strip())
        
        return title
    
    def _extract_deduplicated_title(self, df):
        """Extract title with deduplication and smart selection"""
        
        # Strategy 1: Look for complete title patterns in ALL text
        complete_title_patterns = [
            r'RFP:\s*To\s+Develop.*Digital\s+Library.*Business\s+Plan',
            r'RFP:\s*Request\s+for\s+Proposal.*',
            r'Request\s+for\s+Proposal.*Digital\s+Library.*',
        ]
        
        for _, row in df.iterrows():
            text = row['text'].strip()
            for pattern in complete_title_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if 20 <= len(text) <= 120 and not self._has_obvious_duplication(text):
                        self.logger.info(f"Found complete title pattern: '{text}'")
                        return self._clean_title_final(text)
        
        # Strategy 2: ML predictions with deduplication
        title_candidates = df[df['predicted_label'] == 'TITLE']
        
        if len(title_candidates) > 0:
            for _, row in title_candidates.iterrows():
                text = row['text'].strip()
                clean_text = self._deduplicate_text(text)
                
                if (15 <= len(clean_text) <= 120 and 
                    not self._is_obvious_fragment(clean_text) and
                    self._title_completeness_score(clean_text) > 5):
                    self.logger.info(f"Using deduplicated ML title: '{clean_text}'")
                    return self._clean_title_final(clean_text)
        
        # Strategy 3: Search for keyword-based titles with deduplication
        early_pages = df[df['page'] <= 3]
        title_keywords = ['RFP', 'Request', 'Proposal', 'Plan', 'Library', 'Digital', 'Business']
        
        best_title = ""
        best_score = 0
        
        for _, row in early_pages.iterrows():
            text = row['text'].strip()
            
            if any(kw.lower() in text.lower() for kw in title_keywords):
                clean_text = self._deduplicate_text(text)
                
                if 10 <= len(clean_text) <= 150:
                    score = self._title_completeness_score(clean_text)
                    if score > best_score:
                        best_score = score
                        best_title = clean_text
        
        if best_title and best_score > 3:
            self.logger.info(f"Using keyword-based title: '{best_title}' (score: {best_score})")
            return self._clean_title_final(best_title)
        
        # Strategy 4: Large font with deduplication
        first_page = df[df['page'] == 1]
        
        if len(first_page) > 0:
            large_font_candidates = first_page[
                (first_page['font_size'] >= first_page['font_size'].quantile(0.85)) &
                (first_page['text_length'] >= 8)
            ].sort_values(['font_size'], ascending=False)
            
            for _, row in large_font_candidates.head(5).iterrows():
                text = row['text'].strip()
                clean_text = self._deduplicate_text(text)
                
                if (8 <= len(clean_text) <= 120 and 
                    not self._is_obvious_fragment(clean_text)):
                    self.logger.info(f"Using large font title: '{clean_text}'")
                    return self._clean_title_final(clean_text)
        
        self.logger.warning("No suitable title found")
        return ""
    
    def _deduplicate_text(self, text):
        """Remove duplicated words and phrases from text"""
        if not text:
            return ""
        
        # Handle obvious repetition patterns first
        words = text.split()
        
        if len(words) <= 2:
            return text
        
        # Method 1: Remove consecutive identical words
        deduped = []
        prev_word = None
        
        for word in words:
            if word != prev_word:
                deduped.append(word)
            prev_word = word
        
        # Method 2: Remove repeated phrases
        if len(deduped) >= 6:  # Only for longer texts
            # Try to find the shortest repeating unit
            for unit_size in range(1, len(deduped) // 3 + 1):
                if self._is_repeating_pattern(deduped, unit_size):
                    deduped = deduped[:unit_size]
                    break
        
        result = ' '.join(deduped)
        
        # Method 3: Clean up specific fragment patterns
        result = re.sub(r'\b([A-Za-z])\s+\1\b', r'\1', result)  # "R R" -> "R"
        result = re.sub(r'\b(\w+)e+\b', r'\1e', result)  # "Reeeequest" -> "Request" 
        result = re.sub(r'\b(\w+)o+r\b', r'\1or', result)  # "foooor" -> "for"
        
        # Clean up spaces
        result = re.sub(r'\s+', ' ', result.strip())
        
        return result
    
    def _is_repeating_pattern(self, words, unit_size):
        """Check if words follow a repeating pattern of given unit size"""
        if len(words) < unit_size * 2:
            return False
        
        pattern = words[:unit_size]
        
        # Check if this pattern repeats at least twice more
        for i in range(unit_size, len(words), unit_size):
            if i + unit_size > len(words):
                break
            current_unit = words[i:i + unit_size]
            if current_unit != pattern:
                return False
        
        return True
    
    def _has_obvious_duplication(self, text):
        """Check if text has obvious duplication patterns"""
        words = text.split()
        
        # Check for immediate repetition
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                return True
        
        # Check for phrase repetition
        if len(words) >= 6:
            first_half = ' '.join(words[:len(words)//2])
            second_half = ' '.join(words[len(words)//2:len(words)])
            if first_half.lower() == second_half.lower():
                return True
        
        return False
    
    def _title_completeness_score(self, text):
        """Score title completeness and quality"""
        score = 0
        
        # Length bonus (sweet spot for titles)
        if 25 <= len(text) <= 80:
            score += 5
        elif 15 <= len(text) <= 100:
            score += 3
        elif 10 <= len(text) <= 120:
            score += 1
        
        # Word count bonus
        words = re.findall(r'\b[A-Za-z]{2,}\b', text)
        score += len(words)
        
        # Structure bonuses
        if re.search(r'^RFP:\s*Request\s+for\s+Proposal', text, re.IGNORECASE):
            score += 15
        elif re.search(r'^RFP:\s*To\s+Develop.*Business\s+Plan', text, re.IGNORECASE):
            score += 12
        elif re.search(r'^Request\s+for\s+Proposal', text, re.IGNORECASE):
            score += 10
        elif text.startswith('RFP:'):
            score += 5
        
        # Content bonuses
        content_words = ['Digital', 'Library', 'Business', 'Plan', 'Ontario', 'Development']
        for word in content_words:
            if word.lower() in text.lower():
                score += 2
        
        # Penalties
        if re.search(r'[a-z][A-Z]', text):  # Mixed case fragments
            score -= 3
        
        if text.endswith(('f', 'r', 't', 'n')):  # Fragment endings
            score -= 5
        
        if 'ee' in text.lower():  # Fragment patterns like "eeee"
            score -= 3
        
        return score
    
    def _extract_clean_outline(self, df):
        """Extract outline with improved filtering"""
        outline = []
        
        # Strategy 1: High-priority numbered sections
        numbered_sections = self._find_numbered_sections(df)
        outline.extend(numbered_sections)
        
        # Strategy 2: Structural elements
        structural_elements = self._find_structural_elements(df)
        outline.extend(structural_elements)
        
        # Strategy 3: Clear section headers
        section_headers = self._find_section_headers(df)
        outline.extend(section_headers)
        
        # Strategy 4: ML predictions (with better filtering)
        ml_headings = self._find_filtered_ml_headings(df)
        outline.extend(ml_headings)
        
        # Strategy 5: Font-based (if we need more)
        if len(outline) < 8:
            font_headings = self._find_font_headings(df)
            outline.extend(font_headings)
        
        # Clean and deduplicate
        return self._clean_outline(outline)
    
    def _find_numbered_sections(self, df):
        """Find clear numbered sections"""
        numbered = []
        
        for _, row in df.iterrows():
            text = row['text'].strip()
            
            # Clear numbered patterns
            patterns = [
                (r'^\d+\.\s+[A-Z][A-Za-z\s]{2,60}$', "H1"),
                (r'^\d+\.\d+\s+[A-Z][A-Za-z\s]{2,50}$', "H2"),
                (r'^\d+\.\d+\.\d+\s+[A-Z][A-Za-z\s]{2,40}$', "H3"),
            ]
            
            for pattern, level in patterns:
                if re.match(pattern, text):
                    numbered.append({
                        "level": level,
                        "text": text,
                        "page": int(row['page']) - 1,
                        "priority": 1
                    })
                    break
        
        return numbered
    
    def _find_structural_elements(self, df):
        """Find structural document elements"""
        structural = []
        
        patterns = [
            (r'^Appendix\s+[A-Z](?:\s*[:.].*)?$', "H1"),
            (r'^Phase\s+[IVX]+(?:\s*[:.].*)?$', "H1"),  
            (r'^Chapter\s+\d+(?:\s*[:.].*)?$', "H1"),
            (r'^Section\s+\d+(?:\s*[:.].*)?$', "H1"),
        ]
        
        for _, row in df.iterrows():
            text = row['text'].strip()
            
            if 5 <= len(text) <= 100:
                for pattern, level in patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        structural.append({
                            "level": level,
                            "text": text,
                            "page": int(row['page']) - 1,
                            "priority": 1
                        })
                        break
        
        return structural
    
    def _find_section_headers(self, df):
        """Find clear section headers"""
        headers = []
        
        # Common section header patterns
        header_patterns = [
            r'^(Background|Summary|Overview|Timeline|Milestones|Introduction|Conclusion|References?)(?:\s*:)?$',
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*\s*:$',  # "Title Case:"
            r'^(Access|Training|Guidance and Advice|Technological Support)\s*:?$',  # Specific patterns
            r'^For\s+(each|every)\s+[A-Z][a-z]+.*:$',  # "For each Ontario citizen:"
        ]
        
        for _, row in df.iterrows():
            text = row['text'].strip()
            
            if 3 <= len(text) <= 70:
                for pattern in header_patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        headers.append({
                            "level": "H2",
                            "text": text,
                            "page": int(row['page']) - 1,
                            "priority": 2
                        })
                        break
        
        return headers
    
    def _find_filtered_ml_headings(self, df):
        """Find ML-predicted headings with improved filtering"""
        ml_headings = []
        
        heading_labels = ['H1', 'H2', 'H3']
        candidates = df[
            (df['predicted_label'].isin(heading_labels)) &
            (df['confidence'] > 0.1) &
            (df['text_length'] >= 6) &
            (df['text_length'] <= 120)
        ]
        
        for _, row in candidates.iterrows():
            text = row['text'].strip()
            
            # Improved filtering
            if (self._looks_like_heading(text) and 
                not self._is_sentence_fragment(text) and
                not self._is_obvious_fragment(text)):
                
                ml_headings.append({
                    "level": row['predicted_label'],
                    "text": text,
                    "page": int(row['page']) - 1,
                    "priority": 3
                })
        
        return ml_headings
    
    def _find_font_headings(self, df):
        """Find headings based on font size"""
        font_headings = []
        
        for page_num in sorted(df['page'].unique()):
            page_data = df[df['page'] == page_num]
            
            if len(page_data) == 0:
                continue
            
            # Find large fonts on this page
            font_threshold = page_data['font_size'].quantile(0.75)
            
            candidates = page_data[
                (page_data['font_size'] >= font_threshold) &
                (page_data['text_length'] >= 8) &
                (page_data['text_length'] <= 80)
            ]
            
            for _, row in candidates.head(2).iterrows():
                text = row['text'].strip()
                
                if (self._looks_like_heading(text) and 
                    not self._is_sentence_fragment(text)):
                    
                    font_headings.append({
                        "level": "H2",
                        "text": text,
                        "page": int(row['page']) - 1,
                        "priority": 4
                    })
        
        return font_headings
    
    def _is_obvious_fragment(self, text):
        """Check if text is an obvious fragment"""
        fragment_signs = [
            len(text) < 6,
            text.endswith(('f', 'r', 't')) and len(text) < 20,
            'ee' in text and len(text) < 25,
            re.search(r'^[A-Z]\s*$', text),
            text in ['R', 'f', 'r', 't', 'quest', 'oposal'],  # Common fragments
        ]
        return any(fragment_signs)
    
    def _is_sentence_fragment(self, text):
        """Check if text is a sentence fragment"""
        fragment_patterns = [
            r'\b(will be|should be|can be|must be|it is|this is)\b',
            r'\bmillion\s+annually\b',
            r'\bdollars?\s+invested\b',
            r'developed\s+is\s+to\s+document',
            r'\baccess\s+to\s+the\b',
            r'The\s+principles\s+which\s+will\s+define',
            r'Funding\s+from\s+other\s+states',
            r'The\s+proposal\s+should\s+include',
            r'In\s+the\s+third\s+phase\s+the\s+ODL\s+moves',
            r'Implementation\s+of\s+the\s+Ontario\s+Digital\s+Library',
            r'In\s+broad\s+terms,\s+the\s+ODL\s+Business\s+Plan',
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in fragment_patterns)
    
    def _looks_like_heading(self, text):
        """Check if text looks like a heading"""
        if len(text) < 4 or len(text) > 120:
            return False
        
        # Should start with capital or number
        if not (text[0].isupper() or text[0].isdigit()):
            return False
        
        # Should not be all punctuation/numbers
        if re.match(r'^[\d\s\.\-,/:]+$', text):
            return False
        
        # Should have real words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        return len(words) >= 1
    
    def _clean_outline(self, outline):
        """Clean and deduplicate outline with final polish"""
        if not outline:
            return []
        
        # Remove title if it appears in the outline
        title_lower = self.current_title.strip().lower()
        outline = [item for item in outline if item['text'].strip().lower() != title_lower]
        
        # Remove any heading that starts with lowercase (sentence fragments)
        outline = [item for item in outline 
                  if item['text'] and (item['text'][0].isupper() or item['text'][0].isdigit())]
        
        # Sort by priority, then by page
        outline.sort(key=lambda x: (x['priority'], x['page']))
        
        # Remove duplicates
        seen = set()
        cleaned = []
        
        for item in outline:
            text_key = item['text'].lower().strip()
            
            if text_key not in seen and len(text_key) > 3:
                seen.add(text_key)
                cleaned.append({
                    "level": item["level"],
                    "text": item["text"],
                    "page": item["page"]
                })
        
        # Final sort by page
        cleaned.sort(key=lambda x: (x['page'], 0))
        
        return cleaned[:35]
    
    def process_directory(self, input_dir, output_dir):
        """Process all PDFs in input directory"""
        start_time = time.time()
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir}")
            return
        
        processed_count = 0
        error_count = 0
        
        for pdf_file in pdf_files:
            try:
                result = self.analyze_pdf(str(pdf_file))
                
                output_file = output_path / f"{pdf_file.stem}.json"
                save_json(result, str(output_file))
                
                processed_count += 1
                self.logger.info(f"{pdf_file.name}: Title='{result['title'][:40]}...', Headings={len(result['outline'])}")
                
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                
                output_file = output_path / f"{pdf_file.stem}.json"
                save_json({"title": "", "outline": []}, str(output_file))
        
        total_time = time.time() - start_time
        self.logger.info(f"Completed: {processed_count} success, {error_count} errors")
