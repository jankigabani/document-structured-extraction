import os
import json
import tiktoken
import mimetypes
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
from pathlib import Path
import fitz  # PyMuPDF
import docx2txt
import pandas as pd
import markdown
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, ValidationError
import yaml
import getpass
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

class BasicInfo(BaseModel):
    """Extract basic personal information from resume."""
    name: Optional[str] = Field(default="", description="Full name of the person")
    email: Optional[str] = Field(default="", description="Email address")
    phone: Optional[str] = Field(default="", description="Phone number")
    location: Optional[str] = Field(default="", description="Location/Address (city, state, country)")
    summary: Optional[str] = Field(default="", description="Professional summary or objective")
    label: Optional[str] = Field(default="", description="Professional title or role (e.g., Software Engineer)")
    url: Optional[str] = Field(default="", description="Personal website or portfolio URL")

class Profile(BaseModel):
    """Extract social media profiles."""
    network: str = Field(description="Social network name (e.g., LinkedIn, GitHub, Twitter)")
    username: Optional[str] = Field(default="", description="Username on the platform")
    url: str = Field(description="URL to the profile")

class WorkExperience(BaseModel):
    """Extract work experience information."""
    name: str = Field(description="Company name")
    position: str = Field(description="Job title/position")
    location: Optional[str] = Field(default="", description="Work location (city, state)")
    startDate: Optional[str] = Field(default="", description="Start date in YYYY-MM format or YYYY")
    endDate: Optional[str] = Field(default="", description="End date in YYYY-MM format, YYYY, or 'Present'")
    summary: Optional[str] = Field(default="", description="Job description and main responsibilities")
    highlights: List[str] = Field(default=[], description="Key achievements, accomplishments, or bullet points")
    url: Optional[str] = Field(default="", description="Company website URL")

class Education(BaseModel):
    """Extract education information."""
    institution: str = Field(description="Name of educational institution")
    studyType: Optional[str] = Field(default="", description="Degree type (Bachelor, Master, PhD, etc.)")
    area: Optional[str] = Field(default="", description="Field of study or major")
    startDate: Optional[str] = Field(default="", description="Start date in YYYY format")
    endDate: Optional[str] = Field(default="", description="End date in YYYY format")
    score: Optional[str] = Field(default="", description="GPA, percentage, or grade")
    url: Optional[str] = Field(default="", description="Institution website URL")
    courses: List[str] = Field(default=[], description="Notable courses or subjects")

class Skills(BaseModel):
    """Extract skills information."""
    name: str = Field(description="Skill category name (e.g., Programming Languages, Web Development)")
    level: Optional[str] = Field(default="", description="Proficiency level (Beginner, Intermediate, Advanced, Expert)")
    keywords: List[str] = Field(description="List of specific skills in this category")

class Projects(BaseModel):
    """Extract project information."""
    name: str = Field(description="Project name or title")
    description: Optional[str] = Field(default="", description="Project description and overview")
    keywords: List[str] = Field(default=[], description="Technologies, programming languages, or tools used")
    highlights: List[str] = Field(default=[], description="Key features, achievements, or accomplishments")
    startDate: Optional[str] = Field(default="", description="Project start date")
    endDate: Optional[str] = Field(default="", description="Project end date")
    url: Optional[str] = Field(default="", description="Project URL, demo, or repository link")
    roles: List[str] = Field(default=[], description="Your role(s) in the project")
    entity: Optional[str] = Field(default="", description="Associated company, organization, or entity")
    type: Optional[str] = Field(default="", description="Project type (personal, work, academic, etc.)")

class Languages(BaseModel):
    """Extract language proficiency information."""
    language: str = Field(description="Language name")
    fluency: str = Field(description="Fluency level (Native, Fluent, Intermediate, Basic)")

class Awards(BaseModel):
    """Extract awards and achievements."""
    title: str = Field(description="Award or achievement title")
    date: Optional[str] = Field(default="", description="Date received (YYYY-MM-DD format)")
    awarder: Optional[str] = Field(default="", description="Organization or entity that gave the award")
    summary: Optional[str] = Field(default="", description="Description of the award or achievement")

class Certificates(BaseModel):
    """Extract certification information."""
    name: str = Field(description="Certificate name")
    issuer: str = Field(description="Issuing organization")
    date: Optional[str] = Field(default="", description="Date obtained")
    url: Optional[str] = Field(default="", description="Certificate verification URL")

class ActionInputOutput(BaseModel):
    """Extract input or output definition."""
    name: str = Field(description="Name of the input/output parameter")
    description: str = Field(description="Description of the parameter")
    required: Optional[bool] = Field(description="Whether the parameter is required")
    default_value: Optional[str] = Field(description="Default value if any")

class ActionStep(BaseModel):
    """Extract action step information."""
    id: Optional[str] = Field(description="Step ID")
    name: str = Field(description="Step name or description")
    action_type: str = Field(description="Type: 'uses' for external action or 'run' for command")
    uses: Optional[str] = Field(description="External action being used (e.g., actions/checkout@v4)")
    run_command: Optional[str] = Field(description="Command being executed")
    shell: Optional[str] = Field(description="Shell type (bash, powershell, etc.)")
    with_parameters: Optional[Dict[str, str]] = Field(description="Parameters passed to the action")
    condition: Optional[str] = Field(description="Conditional execution (if clause)")

class ActionBranding(BaseModel):
    """Extract branding information."""
    icon: str = Field(description="Icon name")
    color: str = Field(description="Color theme")

class GitHubActionInfo(BaseModel):
    """Extract complete GitHub Action information."""
    name: str = Field(description="Action name")
    author: Optional[str] = Field(description="Action author")
    description: str = Field(description="Action description")
    purpose: Optional[str] = Field(description="Purpose or use case of the action")

tools = [BasicInfo, Profile, WorkExperience, Education, Skills, Projects, Languages, Awards, Certificates]
github_action_tools = [ActionInputOutput, ActionStep, ActionBranding, GitHubActionInfo]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    google_api_key=GEMINI_API_KEY,  
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

def identify_file_type(file_path):
    """Identify file type based on extension."""
    if file_path.endswith(".pdf"):
        return "pdf"
    elif file_path.endswith(".docx"):
        return "docx"
    elif file_path.endswith(".csv"):
        return "csv"
    elif file_path.endswith(".md"):
        return "md"
    elif file_path.endswith(".txt"):
        return "txt"
    elif file_path.endswith(".json"):
        return "json"
    elif file_path.endswith((".yml", ".yaml")):
        return "yaml"
    else:
        raise ValueError("Unsupported file type")

def parse_file(file_path, file_type):
    """Parse various file types and extract text content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_type == "pdf":
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    elif file_type == "docx":
        return docx2txt.process(file_path).strip()
    elif file_type == "csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    elif file_type == "md":
        with open(file_path, "r", encoding="utf-8") as f:
            return markdown.markdown(f.read())
    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    elif file_type == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    elif file_type == "yaml":
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return yaml.dump(data, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))
    except:
        # Fallback estimation
        return int(len(text.split()) * 1.3)

def chunk_text(text, chunk_size=15000, chunk_overlap=500):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens
    )
    return splitter.split_text(text)

def analyze_schema_complexity(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze JSON schema complexity to determine processing strategy."""

    def count_nested_objects(obj, level=0):
        count = 0
        max_depth = level

        if isinstance(obj, dict):
            if obj.get('type') == 'object':
                count += 1

            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    nested_count, nested_depth = count_nested_objects(value, level + 1)
                    count += nested_count
                    max_depth = max(max_depth, nested_depth)

        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    nested_count, nested_depth = count_nested_objects(item, level + 1)
                    count += nested_count
                    max_depth = max(max_depth, nested_depth)

        return count, max_depth

    nested_objects, max_nesting_level = count_nested_objects(schema)
    schema_str = json.dumps(schema)
    schema_token_count = count_tokens(schema_str)

    if nested_objects < 20 and max_nesting_level < 4:
        complexity = "simple"
        recommended_chunks = 1
    elif nested_objects < 50 and max_nesting_level < 6:
        complexity = "medium"
        recommended_chunks = 2
    else:
        complexity = "complex"
        recommended_chunks = 3

    return {
        "nested_objects": nested_objects,
        "max_nesting_level": max_nesting_level,
        "schema_token_count": schema_token_count,
        "complexity": complexity,
        "recommended_chunks": recommended_chunks
    }

def convert_text_to_json(text: str, schema: Dict[str, Any], confidence_threshold: float = 0.7):
    """
    Convert unstructured text to structured JSON following the given schema.
    """
    complexity_analysis = analyze_schema_complexity(schema)
    print(f"Processing with {complexity_analysis['complexity']} complexity...")

    text_tokens = count_tokens(text)
    print(f"Input text tokens: {text_tokens}")

    max_context = 25000  

    if text_tokens > max_context:
        print(f"Text too large ({text_tokens} tokens), chunking into smaller parts...")
        chunks = chunk_text(text, chunk_size=15000, chunk_overlap=500)
        print(f"Created {len(chunks)} chunks")
    else:
        chunks = [text]
        print("Processing as single chunk")

    all_results = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}...")

        extraction_prompt = f"""
        You are an expert at extracting structured information from resumes and CVs.

        Extract information from the following resume/CV text and structure it using the provided tools.

        IMPORTANT INSTRUCTIONS:
        1. Extract ALL relevant information you can find from the resume
        2. Use the exact field names specified in the tools
        3. For missing information, use empty strings ("") or empty arrays ([]) as appropriate
        4. Format dates consistently: use YYYY-MM-DD, YYYY-MM, or YYYY format
        5. Be thorough - extract every piece of information available
        6. For work experience: extract company name, position, dates, location, summary, and all achievements/highlights
        7. For education: extract institution, degree type, field of study, dates, and GPA/scores
        8. For skills: group skills into logical categories (e.g., "Programming Languages", "Web Technologies")
        9. For projects: extract name, description, technologies used, and key achievements
        10. For contact info: extract name, email, phone, location, summary, professional title
        11. Look for social media profiles (LinkedIn, GitHub, Twitter, etc.)
        12. Extract languages, awards, certificates if mentioned
        13. Call each tool multiple times if you find multiple instances

        Resume/CV text to extract from:
        {chunk}

        Use ALL the provided tools to extract information systematically. Make sure to call each tool type multiple times if you find multiple entries of that type.
        """

        messages = [HumanMessage(extraction_prompt)]

        try:
            ai_response = llm_with_tools.invoke(messages)

            if hasattr(ai_response, 'tool_calls') and ai_response.tool_calls:
                print(f"Found {len(ai_response.tool_calls)} tool calls")
                chunk_results = {}

                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    if tool_name not in chunk_results:
                        chunk_results[tool_name] = []
                    chunk_results[tool_name].append(tool_args)

                all_results.append(chunk_results)
                
                for tool_name, results in chunk_results.items():
                    print(f"  - {tool_name}: {len(results)} entries extracted")
            else:
                print("No structured data extracted from this chunk")
                print("AI Response:", ai_response.content if hasattr(ai_response, 'content') else str(ai_response))

        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        final_result = merge_chunk_results(all_results)
        print(f"\nFinal extraction summary:")
        print(f"  - Basic info fields: {len([k for k, v in final_result['basics'].items() if v])}")
        print(f"  - Work experiences: {len(final_result['work'])}")
        print(f"  - Education entries: {len(final_result['education'])}")
        print(f"  - Skill categories: {len(final_result['skills'])}")
        print(f"  - Projects: {len(final_result['projects'])}")
        return final_result
    else:
        print("No results extracted from any chunks")
        return {
            "basics": {},
            "work": [],
            "education": [],
            "skills": [],
            "projects": [],
            "languages": [],
            "awards": [],
            "certificates": []
        }

def merge_chunk_results(chunk_results: List[Dict]) -> Dict:
    """Merge results from multiple chunks into final structured format."""

    merged = {
        "basics": {},
        "work": [],
        "education": [],
        "skills": [],
        "projects": [],
        "languages": [],
        "awards": [],
        "certificates": [],
        "volunteer": [],
        "publications": [],
        "interests": [],
        "references": []
    }

    profiles = []

    for chunk_result in chunk_results:
        if "BasicInfo" in chunk_result:
            for basic_info in chunk_result["BasicInfo"]:
                for key, value in basic_info.items():
                    if value and str(value).strip():  
                        merged["basics"][key] = value

        if "Profile" in chunk_result:
            for profile in chunk_result["Profile"]:
                if profile.get("network") and profile.get("url"):
                    profiles.append({
                        "network": profile.get("network", ""),
                        "username": profile.get("username", ""),
                        "url": profile.get("url", "")
                    })

        if "WorkExperience" in chunk_result:
            for work in chunk_result["WorkExperience"]:
                if work.get("name") or work.get("position"): 
                    merged["work"].append({
                        "name": work.get("name", ""),
                        "position": work.get("position", ""),
                        "location": work.get("location", ""),
                        "startDate": work.get("startDate", ""),
                        "endDate": work.get("endDate", ""),
                        "summary": work.get("summary", ""),
                        "highlights": work.get("highlights", []),
                        "url": work.get("url", "")
                    })

        if "Education" in chunk_result:
            for edu in chunk_result["Education"]:
                if edu.get("institution"):  
                    merged["education"].append({
                        "institution": edu.get("institution", ""),
                        "studyType": edu.get("studyType", ""),
                        "area": edu.get("area", ""),
                        "startDate": edu.get("startDate", ""),
                        "endDate": edu.get("endDate", ""),
                        "score": edu.get("score", ""),
                        "url": edu.get("url", ""),
                        "courses": edu.get("courses", [])
                    })

        if "Skills" in chunk_result:
            for skill in chunk_result["Skills"]:
                if skill.get("name") and skill.get("keywords"): 
                    merged["skills"].append({
                        "name": skill.get("name", ""),
                        "level": skill.get("level", ""),
                        "keywords": skill.get("keywords", [])
                    })

        if "Projects" in chunk_result:
            for project in chunk_result["Projects"]:
                if project.get("name"): 
                    merged["projects"].append({
                        "name": project.get("name", ""),
                        "description": project.get("description", ""),
                        "keywords": project.get("keywords", []),
                        "highlights": project.get("highlights", []),
                        "startDate": project.get("startDate", ""),
                        "endDate": project.get("endDate", ""),
                        "url": project.get("url", ""),
                        "roles": project.get("roles", []),
                        "entity": project.get("entity", ""),
                        "type": project.get("type", "")
                    })

        if "Languages" in chunk_result:
            for lang in chunk_result["Languages"]:
                if lang.get("language"):
                    merged["languages"].append({
                        "language": lang.get("language", ""),
                        "fluency": lang.get("fluency", "")
                    })

        if "Awards" in chunk_result:
            for award in chunk_result["Awards"]:
                if award.get("title"):
                    merged["awards"].append({
                        "title": award.get("title", ""),
                        "date": award.get("date", ""),
                        "awarder": award.get("awarder", ""),
                        "summary": award.get("summary", "")
                    })

        if "Certificates" in chunk_result:
            for cert in chunk_result["Certificates"]:
                if cert.get("name"):
                    merged["certificates"].append({
                        "name": cert.get("name", ""),
                        "issuer": cert.get("issuer", ""),
                        "date": cert.get("date", ""),
                        "url": cert.get("url", "")
                    })

    if profiles:
        merged["basics"]["profiles"] = profiles

    return merged

def determine_document_type_and_schema(file_path, file_type):
    """Determine document type and return appropriate schema and tools."""
    if file_type in ["yaml"]:
        return "github_action", github_action_tools
    elif file_type in ["pdf", "docx", "txt"] and "resume" in file_path.lower():
        return "resume", tools
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        if any(keyword in content for keyword in ["action", "workflow", "github", "steps", "inputs", "outputs"]):
            return "github_action", github_action_tools
        elif any(keyword in content for keyword in ["experience", "education", "skills", "resume", "cv"]):
            return "resume", tools
        else:
            return "resume", tools

def merge_github_action_results(chunk_results: List[Dict]) -> Dict:
    """Merge GitHub Action results from multiple chunks."""
    merged = {
        "name": "",
        "author": "",
        "description": "",
        "branding": {
            "icon": "",
            "color": ""
        },
        "inputs": {},
        "outputs": {},
        "runs": {
            "using": "composite",
            "steps": []
        }
    }

    for chunk_result in chunk_results:
        if "GitHubActionInfo" in chunk_result:
            for action_info in chunk_result["GitHubActionInfo"]:
                for key, value in action_info.items():
                    if value and value.strip():
                        if key == "purpose":
                            if merged["description"] and value not in merged["description"]:
                                merged["description"] += f" {value}"
                            elif not merged["description"]:
                                merged["description"] = value
                        else:
                            merged[key] = value

        if "ActionBranding" in chunk_result:
            for branding in chunk_result["ActionBranding"]:
                merged["branding"]["icon"] = branding.get("icon", "")
                merged["branding"]["color"] = branding.get("color", "")

        if "ActionInputOutput" in chunk_result:
            for param in chunk_result["ActionInputOutput"]:
                param_name = param.get("name", "")
                if param_name:
                    param_data = {
                        "description": param.get("description", ""),
                        "required": param.get("required", False)
                    }
                    if param.get("default_value"):
                        param_data["default"] = param.get("default_value")
        
                    if param_name == "page-url":
                        merged["outputs"][param_name] = {
                            "description": param.get("description", ""),
                            "value": param.get("default_value", "")
                        }
                    else:
                        merged["inputs"][param_name] = param_data

        if "ActionStep" in chunk_result:
            for step in chunk_result["ActionStep"]:
                step_data = {
                    "name": step.get("name", "")
                }
                if step.get("id"):
                    step_data["id"] = step.get("id")
                if step.get("uses"):
                    step_data["uses"] = step.get("uses")
                if step.get("run_command"):
                    step_data["run"] = step.get("run_command")
                if step.get("shell"):
                    step_data["shell"] = step.get("shell")
                if step.get("with_parameters"):
                    step_data["with"] = step.get("with_parameters")
                if step.get("condition"):
                    step_data["if"] = step.get("condition")
                merged["runs"]["steps"].append(step_data)

    return merged

def convert_document_to_json(text: str, doc_type: str, schema: Dict[str, Any], processing_tools: List, confidence_threshold: float = 0.7):
    """
    Universal function to convert any document type to structured JSON.
    Args:
        text: Input text to convert
        doc_type: Type of document (resume, github_action, etc.)
        schema: JSON schema to follow
        processing_tools: Pydantic tools for extraction
        confidence_threshold: Minimum confidence for field acceptance
    Returns:
        Dict containing the structured data
    """
    complexity_analysis = analyze_schema_complexity(schema)
    print(f"Processing {doc_type} with {complexity_analysis['complexity']} complexity...")

    text_tokens = count_tokens(text)
    print(f"Input text tokens: {text_tokens}")

    max_context = 25000  
    if text_tokens > max_context:
        print(f"Text too large ({text_tokens} tokens), chunking into smaller parts...")
        chunks = chunk_text(text, chunk_size=15000, chunk_overlap=500)
        print(f"Created {len(chunks)} chunks")
    else:
        chunks = [text]
        print("Processing as single chunk")

    llm_with_doc_tools = llm.bind_tools(processing_tools)

    all_results = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}...")

        if doc_type == "github_action":
            extraction_prompt = f"""
            You are an expert at extracting GitHub Action workflow information from text descriptions.
            Extract information from the following text and structure it according to the GitHub Action schema.
            IMPORTANT INSTRUCTIONS:
            1. Extract action name, description, and purpose
            2. Identify all inputs with their descriptions, requirements, and defaults
            3. Identify all outputs with their descriptions and values, including the page-url
            4. Extract all workflow steps with their actions, commands, and parameters
            5. Look for branding information (icon, color)
            6. Format step conditions and shell specifications correctly
            Schema to follow:
            {json.dumps(schema, indent=2)}
            Text to extract from:
            {chunk}
            Use the provided tools to extract information systematically.
            """
        else:  
            extraction_prompt = f"""
            You are an expert at extracting structured information from unstructured text.
            Extract information from the following text and structure it according to the provided schema.
            IMPORTANT INSTRUCTIONS:
            1. Extract ALL relevant information you can find
            2. Use the exact field names from the schema
            3. Format dates as YYYY-MM-DD or YYYY-MM or YYYY
            4. If information is not available, use null or empty string
            5. Be thorough and accurate
            Schema to follow:
            {json.dumps(schema, indent=2)}
            Text to extract from:
            {chunk}
            Use the provided tools to extract information systematically.
            """

        messages = [HumanMessage(extraction_prompt)]

        ai_response = llm_with_doc_tools.invoke(messages)

        if hasattr(ai_response, 'tool_calls') and ai_response.tool_calls:
            print(f"Found {len(ai_response.tool_calls)} tool calls")
            chunk_results = {}

            for tool_call in ai_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                print(f"Extracted {tool_name}: {tool_args}")

                if tool_name not in chunk_results:
                    chunk_results[tool_name] = []
                chunk_results[tool_name].append(tool_args)

            all_results.append(chunk_results)
        else:
            print("No structured data extracted from this chunk")

    if doc_type == "github_action":
        final_result = merge_github_action_results(all_results)
    else:
        final_result = merge_chunk_results(all_results)  

    return final_result

def process_document_universal(file_path: str, schema_path: str) -> Dict[str, Any]:
    """
    Universal pipeline to process any supported document type.
    Args:
        file_path: Path to the document file
        schema_path: Path to the schema file
    Returns:
        Structured JSON data following the appropriate schema
    """
    print(f"Processing document: {file_path}")
    print("=" * 50)

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        print("Analyzing document...")
        file_type = identify_file_type(file_path)
        doc_type, processing_tools = determine_document_type_and_schema(file_path, file_type)
        print(f"Document analyzed successfully!")
        print(f"   - File type: {file_type}")
        print(f"   - Document type: {doc_type}")
        print(f"   - Schema sections: {len(schema.get('properties', {}))}")

        print("\nParsing file content...")
        text = parse_file(file_path, file_type)
        print(f"File parsed successfully!")
        print(f"   - Text length: {len(text)} characters")
        print(f"   - Token count: {count_tokens(text)}")

        print(f"\nConverting {doc_type} to structured JSON...")
        
        if doc_type == "resume":
            structured_data = convert_text_to_json(text, schema)
        else:
            structured_data = convert_document_to_json(
                text, doc_type, schema, processing_tools
            )
        
        print("\nConversion completed!")

        print("\nBasic validation...")
        if doc_type == "github_action":
            required_sections = ["name", "description", "runs"]
        else:  
            required_sections = ["basics", "work", "education", "skills", "projects"]

        validation_results = {}
        for section in required_sections:
            if section in structured_data and structured_data[section]:
                if isinstance(structured_data[section], dict):
                    has_content = any(v for v in structured_data[section].values() if v)
                    validation_results[section] = "Present" if has_content else "Empty"
                elif isinstance(structured_data[section], list):
                    validation_results[section] = "Present" if structured_data[section] else "Empty"
                else:
                    validation_results[section] = "Present" if structured_data[section] else "Empty"
            else:
                validation_results[section] = "Missing"

        print("Validation Results:")
        for section, status in validation_results.items():
            print(f"   - {section}: {status}")

        return {
            "status": "success",
            "document_type": doc_type,
            "data": structured_data,
            "validation": validation_results,
            "metadata": {
                "file_type": file_type,
                "original_length": len(text),
                "token_count": count_tokens(text),
                "processing_timestamp": datetime.now().isoformat(),
                "schema_complexity": analyze_schema_complexity(schema)
            }
        }
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "data": None
        }