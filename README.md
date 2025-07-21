# 🧠 Document-to-Structured-JSON API

This FastAPI-powered system converts unstructured documents (resumes, GitHub Actions, etc.) into structured JSON, strictly following a provided schema. Powered by Google’s Gemini model via LangChain.

## 🚀 Features

* ✅ Supports multiple file types: `.pdf`, `.docx`, `.txt`, `.csv`, `.md`, `.yaml`, `.json`
* 🔍 Auto-detects document type (resume or GitHub Action)
* 🤖 Uses Gemini-2.0 via LangChain tool calling for structured extraction
* 🧠 Adapts to schema complexity (supports deeply nested schemas with 100+ fields)
* 📊 Provides metadata and validation summary
* 🔌 Exposed as an easy-to-use FastAPI service

---

## 🛠️ How It Works

### Method Used

This system uses **LangChain's tool calling capabilities** with the **Gemini 2.0 Flash** model. Pydantic models are used to define the schema tools. The model is prompted to extract structured data chunk-wise and invoke tools per schema specification.

Depending on the schema complexity, the system:

* Automatically chunks large inputs
* Binds tools dynamically based on document type
* Calls the Gemini model adaptively with validation

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document-structured-extraction.git
cd document-structured-extraction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:

```
GEMINI_API_KEY=your_google_gemini_api_key
GOOGLE_CLOUD_PROJECT=your_project_id
```

---

## ▶️ Running the API

```bash
uvicorn main:app --reload
```

The API will be available at: [http://localhost:8000](http://localhost:8000)

---

## 📄 API Endpoint

### `POST /process-file/`

**Description**: Process a file with a schema and return structured JSON.

#### ✅ Request Body

```json
{
  "file_path": "path/to/document.pdf",
  "schema_path": "path/to/schema.json",
  "output_path": "optional/output/directory"
}
```

* `file_path`: Path to the input document.
* `schema_path`: Path to the JSON schema.
* `output_path`: (Optional) Directory to save the output file.

#### 🔁 Response Format

```json
{
  "status": "success",
  "message": "Document processed successfully",
  "document_type": "resume",
  "data": { /* structured JSON */ },
  "validation": { /* section-level validation */ },
  "metadata": {
    "file_type": "pdf",
    "original_length": 4587,
    "token_count": 996,
    "processing_timestamp": "2025-07-21T08:05:12",
    "schema_complexity": {
      "nested_objects": 27,
      "max_nesting_level": 5,
      "schema_token_count": 4589,
      "complexity": "medium",
      "recommended_chunks": 2
    }
  },
  "output_path": "output/processed_document.json"
}
```

---

## ✅ Health Checks

### `GET /`

Returns API status.

### `GET /health`

Returns `{ "status": "healthy" }`

---

## 📁 Example Files

* `sample_resume.pdf`: A typical input resume
* `resume_schema.json`: Matching schema for resume
* `github_action.md`: Sample GitHub Action description
* `github_schema.json`: Matching schema for GitHub actions

---

## 🥪 Test Cases Completed

You’ve already validated this solution on:

* ✔️ Resume extraction from PDF (`sample_resume.pdf`)
* ✔️ GitHub Action extraction from Markdown (`github_action.md`)

---

## 🧡 Trade-Offs Made

* **Speed vs. Accuracy**: Gemini-Flash was used for its speed. For more complex extraction, Gemini-Pro or chunk fine-tuning can be considered.
* **Schema Adaptability**: Rather than training for each schema, we leverage prompt-tool calling to maximize flexibility.
* **LLM Cost**: LLM calls are made chunk-wise depending on schema complexity and token length.

---

## 🧠 Future Improvements

* Add streaming mode for large docs
* Add confidence scores per field
* Support additional document types (e.g., invoices, contracts)
