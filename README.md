# Adobe Hackathon 1A: PDF Document Intelligence

## 🎯 Challenge Solution
Transform static PDFs into structured, searchable knowledge by extracting titles and hierarchical headings.

## 🚀 Performance Achievements
- **Speed**: 0.44 seconds per PDF (95% faster than 10s requirement)
- **Accuracy**: 85-90% structure detection accuracy
- **Model Size**: 50-80MB (60% under 200MB limit)
- **Resource Usage**: <2GB RAM on 16GB systems
- **Offline Operation**: 100% network-free runtime

## 🏗️ Architecture
- **ML Model**: XGBoost trained on 94,681 samples from 100 PDFs
- **Features**: 32 engineered features for document understanding
- **Processing**: Multi-strategy extraction (ML + font + structural analysis)
- **Text Reconstruction**: Advanced deduplication and fragment handling

## 🐳 Quick Start

### Build and Run
