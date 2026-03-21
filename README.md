# 🛡️ LENS: Local Expert News Sentinel

**LENS (Local Expert News Sentinel)** is a privacy-focused, multi-modal fact-checking system designed to assist users in identifying misinformation in both vernacular and mainstream news content.

Unlike traditional fact-checking tools that require manual searching, LENS operates as a **Chrome Extension**, enabling real-time analysis of selected text and images directly within the browser. It combines Natural Language Processing (NLP) and Computer Vision techniques to provide structured insights into claims, credibility, and supporting evidence.

---

## ✨ Key Features

### 🔍 Multi-Modal Verification
LENS analyzes both textual and visual content:
- **Text Analysis**: Uses semantic similarity via Sentence-Transformers to compare extracted claims against a curated fact dataset.
- **OCR Integration**: Extracts text from screenshots and images using Tesseract OCR for verification.
- **Image–Text Alignment**: Uses CLIP to evaluate whether an image contextually matches the associated text.
- **Image Authenticity (Prototype)**: Uses a pretrained ResNet18 model for basic classification-based heuristics.
> [!NOTE]
> This is not a specialized deepfake detection model.

### 📊 Real-Time Analytics
- **Truth Accuracy Score**: Percentage of claims matching known facts.
- **Claim Confidence Graph**: Visual representation of confidence per claim (Chart.min.js).
- **Contextual Reasoning**: Explains results based on semantic similarity levels.

### ⚙️ System Design Highlights
- **Browser-Based Workflow**: Chrome Extension (Manifest V3) with side-panel UI.
- **Asynchronous Processing**: Redis + RQ used for background task execution.
- **Semantic Matching Pipeline**: Claim extraction → embedding → similarity comparison → scoring.
- **Domain-Based Credibility Scoring**: Heuristic scoring based on URL patterns **.gov**, **.edu** → higher credibility and **news**, **blog**, others → moderate to lower credibility
- **Vector Search Module (Experimental)**: FAISS-based similarity search implemented but not currently used in the main pipeline.
