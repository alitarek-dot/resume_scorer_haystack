import os
from typing import List
from fastapi import FastAPI, HTTPException, Form, APIRouter
from pydantic import BaseModel, HttpUrl
from haystack import Pipeline
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.converters import PyPDFToDocument
import tempfile
import requests

resume_scorer_router = APIRouter()

# Function to dynamically build the prompt template based on user input
def create_dynamic_prompt(job_title, skills, experience, education, job_description):
    prompt_template = f"""
    Analyze the following resume and provide a conclusion about whether the candidate is a good fit for the position of {job_title} based on the criteria below:
    1. Job description: {job_description}
    2. Skills: {', '.join(skills)}
    3. Years of experience: {experience}
    4. Educational background: {education}

    Resume:
    {{% for doc in documents %}}
        {{ doc.content }}
    {{% endfor %}}

    Conclusion:
    """
    return prompt_template

# Retrieve your OpenAI API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

# Set up the OpenAI generator
llm = OpenAIGenerator(model="gpt-4o-mini", api_key=Secret.from_token(openai_key))

class JobInput(BaseModel):
    resume_url: HttpUrl
    job_title: str
    skills: str
    experience: str
    education: str
    job_description: str

@resume_scorer_router.post("/analyze_resume")
async def analyze_resume(job_input: JobInput):
    try:
        # Download the PDF from the provided URL
        response = requests.get(str(job_input.resume_url))
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download the resume")

        # Create a temporary file to store the downloaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Initialize the document store
        document_store = InMemoryDocumentStore()

        # Parse the PDF file and convert it to text using PyPDFToDocument
        converter = PyPDFToDocument()
        docs = converter.run(sources=[temp_file_path])

        # Wrap the text content in Document objects and write it to the document store
        document_store.write_documents([
            doc for doc in docs["documents"]
        ])

        # Set up the retriever using the InMemoryBM25Retriever
        retriever = InMemoryBM25Retriever(document_store=document_store)

        # Generate the dynamic prompt template
        skills_list = [skill.strip() for skill in job_input.skills.split(',')]
        dynamic_prompt_template = create_dynamic_prompt(
            job_input.job_title,
            skills_list,
            job_input.experience,
            job_input.education,
            job_input.job_description
        )

        # Initialize the prompt builder with the dynamic template
        prompt_builder = PromptBuilder(template=dynamic_prompt_template)

        # Build the pipeline and add components in the right sequence
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)

        # Connect nodes in the pipeline
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Ask the analysis question
        question = "Analyze the resume and provide a conclusion about the candidate's fit for the job."
        results = rag_pipeline.run(
            {
                "retriever": {"query": question}
            }
        )

        # Clean up the temporary file
        os.unlink(temp_file_path)

        # Return the response from the OpenAI generator
        return {"analysis": results["llm"]["replies"][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ResumeInput(BaseModel):
    resume_url: HttpUrl

@resume_scorer_router.post("/candidate_info")
async def get_candidate_info(resume_input: ResumeInput):
    try:
        # Download the PDF from the provided URL
        response = requests.get(str(resume_input.resume_url))
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download the resume")

        # Create a temporary file to store the downloaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Initialize the document store
        document_store = InMemoryDocumentStore()

        # Parse the PDF file and convert it to text using PyPDFToDocument
        converter = PyPDFToDocument()
        docs = converter.run(sources=[temp_file_path])

        # Wrap the text content in Document objects and write it to the document store
        document_store.write_documents([
            doc for doc in docs["documents"]
        ])

        # Set up the retriever
        retriever = InMemoryBM25Retriever(document_store=document_store)

        candidate_info = "Provide basic candidate info for example: Name, Phone number, Location, etc."
        candidate_info_prompt_template = """
        Given these documents, {candidate_info} The output should be similar to the following example: Name, Phone Number, Location, etc....
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """

        prompt_builder = PromptBuilder(template=candidate_info_prompt_template)
        llm = OpenAIGenerator(model="gpt-4o-mini", api_key=Secret.from_token(openai_key))

        query_pipeline = Pipeline()
        query_pipeline.add_component("retriever", retriever)
        query_pipeline.add_component("prompt_builder", prompt_builder)
        query_pipeline.add_component("llm", llm)

        query_pipeline.connect("retriever", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder", "llm")

        info = query_pipeline.run(
            {
                "retriever": {"query": candidate_info}
            }
        )
        processed_info = info["llm"]["replies"][0]

        processed_info = processed_info.replace("\n", ", ").replace("  ", " ").strip()
        
        if processed_info.endswith(", "):
            processed_info = processed_info[:-2]

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return {"candidate_info": processed_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(resume_scorer_router, host="0.0.0.0", port=8000)