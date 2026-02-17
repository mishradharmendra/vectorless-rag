# Vectorless RAG with PageIndex

A production-grade implementation of Vectorless RAG for analyzing technical manuals, SOPs, and SEC financial filings.

## What is Vectorless RAG?

Unlike traditional RAG systems that rely on vector embeddings and semantic similarity search, **PageIndex** uses a hierarchical, tree-based document structure with LLM-driven reasoning to navigate and extract precise information.

### Key Differences from Traditional RAG

| Aspect | Traditional RAG | Vectorless RAG (PageIndex) |
|--------|-----------------|---------------------------|
| Retrieval | Vector similarity search | LLM-driven tree navigation |
| Structure | Flat chunks | Hierarchical sections |
| Precision | Approximate matching | Exact section targeting |
| Context | Fixed chunk windows | Full section context |
| Reasoning | Post-retrieval only | Throughout retrieval |
| Cross-refs | Often lost | Automatically followed |

## Use Cases

### Technical Manuals & Policies (Primary Focus)
- Safety procedures and SOPs
- Engineering specifications
- Regulatory compliance documents
- Equipment maintenance manuals

### Other Supported Documents
- SEC 10-K/10-Q filings
- Legal contracts
- Academic research papers
- Medical documentation

## Features

- 🌲 **Hierarchical Document Indexing**: Preserves document structure (Sections, Subsections, Appendices)
- 🧭 **LLM-Powered Navigation**: Uses reasoning to find relevant sections
- 🎯 **Precise Extraction**: Targets exact information, not similar-looking text
- 🔗 **Cross-Reference Following**: Automatically navigates to referenced sections
- 📊 **Transparent Reasoning**: Full navigation trace for auditability
- 🔄 **Backtracking Support**: Can correct navigation mistakes

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. Run the Demo

```bash
python app.py
```

## Usage

### Document Type Selection

The application supports two document types:

1. **Technical Manual / SOP** (High Voltage Safety)
2. **SEC 10-K Filing** (Financial Report)

### Demo Mode

Run predefined queries against sample documents:

```bash
python app.py
# Select document type (1 for Technical Manual)
# Select mode 1 for demo queries
```

### Interactive Mode

Ask your own questions:

```bash
python app.py
# Select document type
# Select mode 2 for interactive
```

### Example Queries (Technical Manual)

- "What PPE is required for work where incident energy is between 8 and 25 cal/cm²?"
- "What are the minimum approach distances for working near 36kV-46kV equipment?"
- "What is the complete LOTO verification procedure for high voltage work?"
- "What training is required to work on high voltage equipment?"
- "What are the emergency procedures if someone contacts energized HV equipment?"

### Example Queries (SEC Filing)

- "What is ACME Corporation's total revenue and growth rate?"
- "What are the main cybersecurity risks and insurance coverage?"
- "What is the company's debt structure and covenant compliance?"

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  PageIndex Navigator                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │  LLM Reasoning Engine                               ││
│  │  - Analyze query intent                             ││
│  │  - Evaluate section relevance                       ││
│  │  - Decide: DESCEND / EXTRACT / BACKTRACK / COMPLETE ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Document Index                         │
│                                                         │
│   📋 HV Safety SOP v3.2                                 │
│   ├── 📁 Section 1: Purpose and Scope                   │
│   ├── 📁 Section 2: Definitions                         │
│   ├── 📁 Section 3: Safety Requirements                 │
│   │   ├── 📁 3.1: Approach Distances                    │
│   │   ├── 📁 3.2: PPE Requirements                      │
│   │   │   ├── 📄 3.2.1: PPE Category 1                  │
│   │   │   ├── 📄 3.2.2: PPE Category 2                  │
│   │   │   ├── 📄 3.2.3: PPE Category 3                  │
│   │   │   └── 📄 3.2.4: PPE Category 4                  │
│   │   └── 📁 3.3: LOTO Procedures                       │
│   └── 📁 Appendices                                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Query Result                           │
│  - Synthesized answer                                   │
│  - Source sections                                      │
│  - Navigation path                                      │
│  - Confidence score                                     │
│  - Full reasoning trace                                 │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag/
├── app.py                                  # Main CLI application
├── pageindex.py                            # Core PageIndex implementation
├── requirements.txt                        # Python dependencies
├── sample_data/
│   ├── technical_manual_hv_safety.json     # Sample HV Safety SOP
│   └── sec_10k_sample.json                 # Sample SEC 10-K filing
├── README.md                               # This file
└── medium-article.md                       # Detailed article with examples
```

## Sample Output

```
 Query: What PPE is required for work where incident energy is between 8 and 25 cal/cm²?

 Navigation Trace:
Step 1: DESCEND - Query is about PPE requirements → Section 3: Safety Requirements
Step 2: DESCEND - Section 3.2 covers PPE → 3.2: Personal Protective Equipment
Step 3: DESCEND - 8-25 cal/cm² is Category 3 → 3.2.3: PPE Category 3
Step 4: EXTRACT - Found specific PPE requirements for Category 3
Step 5: BACKTRACK - Check additional glove requirements → 3.2
Step 6: DESCEND - Voltage-rated gloves apply → 3.2.5: Voltage-Rated Gloves
Step 7: EXTRACT - Found glove inspection requirements
Step 8: COMPLETE - Have comprehensive PPE requirements

Answer:
For work where incident energy is between 8 and 25 cal/cm², PPE Category 3 
is required. Equipment includes:
• Arc flash suit jacket and bib overalls (minimum 25 cal/cm²)
• Arc flash suit hood
• Arc-rated gloves  
• Safety glasses
• Hearing protection
• Hard hat (if not in hood)
• Leather work shoes

Additionally, gloves must be inspected before each use and tested every 
6 months. Leather protectors are required over rubber gloves.

Sources: 3.2.3: PPE Category 3 (25 cal/cm²), 3.2.5: Voltage-Rated Gloves
Navigation Path: root → Section 3 → 3.2 → 3.2.3 → ↑3.2 → 3.2.5
Confidence: 95%
```

## Extending for Production

### Adding New Document Types

1. Create a parser for your document format
2. Implement a `from_*` class method in `DocumentIndex`
3. Ensure hierarchical structure is preserved

```python
@classmethod
def from_your_document_type(cls, data: dict) -> 'DocumentIndex':
    root = DocumentNode(id="root", title="Your Document", ...)
    # Build your tree structure
    return cls(document_id=..., metadata=..., root=root)
```

### Customizing Navigation

Modify the prompts in `PageIndexNavigator`:
- `NAVIGATOR_SYSTEM_PROMPT`: Overall navigation behavior
- `NAVIGATION_PROMPT`: Per-step decision making

### Production API

See [medium-article.md](medium-article.md#deployment-guide) for:
- FastAPI deployment
- Docker containerization
- Kubernetes configuration

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o recommended)
- See requirements.txt for dependencies

## License

MIT
