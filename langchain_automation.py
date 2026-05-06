"""
LangChain Dual-Path Automation Workflow
- Path 1: In-depth research for outreach
- Path 2: Research data for UI prompt building in Stitch
- Uses OpenAI API with extended thinking capability
"""

import json
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM with extended thinking
# Note: Use latest available model (gpt-4o or o1-preview for thinking capability)
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=1,  # Required for extended thinking (o1/o3 models)
)

# Extended thinking LLM (for deeper analysis)
llm_thinking = ChatOpenAI(
    model="o1-preview",  # or o3 when available - these have extended thinking
    api_key=OPENAI_API_KEY,
    temperature=1,  # Must be 1 for reasoning models
)

# ============================================================================
# Output Models (Pydantic)
# ============================================================================

class OutreachResearch(BaseModel):
    """Data structure for outreach research path"""
    company_name: str = Field(..., description="Company name from input")
    key_insights: List[str] = Field(..., description="Key insights for outreach")
    decision_makers: List[str] = Field(..., description="Identified decision makers")
    pain_points: List[str] = Field(..., description="Business pain points")
    personalized_messaging: str = Field(..., description="Personalized outreach message")
    next_steps: List[str] = Field(..., description="Recommended next steps")


class UIPromptResearch(BaseModel):
    """Data structure for UI prompt research path"""
    company_profile: str = Field(..., description="Company profile summary")
    ui_requirements: Dict[str, Any] = Field(..., description="UI requirements from data")
    data_fields: List[str] = Field(..., description="Key data fields needed in UI")
    user_personas: List[str] = Field(..., description="Target user personas")
    functionality_needs: List[str] = Field(..., description="Required UI functionality")
    design_recommendations: str = Field(..., description="Design recommendations for Stitch")


# ============================================================================
# Path 1: In-Depth Research for Outreach
# ============================================================================

def path_1_outreach_research(input_data: Dict[str, Any]) -> OutreachResearch:
    """
    Path 1: Deep research for personalized outreach
    Uses extended thinking for complex analysis
    
    Args:
        input_data: JSON data containing company/prospect information
    
    Returns:
        OutreachResearch: Structured outreach research data
    """
    
    prompt_template = PromptTemplate(
        input_variables=["company_info"],
        template="""
        You are an expert sales researcher specializing in personalized outreach.
        
        Analyze this company/prospect data deeply:
        {company_info}
        
        Provide comprehensive research for outreach including:
        1. Key business insights that make them a good prospect
        2. Identified decision makers and stakeholders
        3. Specific pain points relevant to your solution
        4. Personalized messaging approach
        5. Recommended next steps for outreach
        
        Think through each aspect carefully, considering market dynamics and their business context.
        """
    )
    
    # Use extended thinking LLM for deeper analysis
    chain = LLMChain(
        llm=llm_thinking,
        prompt=prompt_template
    )
    
    # Format input
    company_info_str = json.dumps(input_data, indent=2)
    
    # Get response
    response = chain.run(company_info=company_info_str)
    
    # Parse into structured format
    parser = JsonOutputParser(pydantic_object=OutreachResearch)
    
    # Add structured parsing prompt
    parse_prompt = PromptTemplate(
        input_variables=["analysis"],
        template="""
        Based on this analysis, extract and structure the data as JSON:
        {analysis}
        
        Return ONLY valid JSON matching this structure:
        {{
            "company_name": "...",
            "key_insights": [...],
            "decision_makers": [...],
            "pain_points": [...],
            "personalized_messaging": "...",
            "next_steps": [...]
        }}
        """
    )
    
    parse_chain = LLMChain(llm=llm, prompt=parse_prompt)
    structured_response = parse_chain.run(analysis=response)
    
    # Parse JSON
    structured_data = json.loads(structured_response)
    
    return OutreachResearch(**structured_data)


# ============================================================================
# Path 2: Research Data for UI Prompt in Stitch
# ============================================================================

def path_2_ui_prompt_research(input_data: Dict[str, Any]) -> UIPromptResearch:
    """
    Path 2: Extract and structure data for UI prompt generation in Stitch
    Creates data that feeds directly into Stitch UI builder prompts
    
    Args:
        input_data: JSON data containing product/feature requirements
    
    Returns:
        UIPromptResearch: Structured data for Stitch UI building
    """
    
    prompt_template = PromptTemplate(
        input_variables=["product_info"],
        template="""
        You are a UI/UX specialist and product architect.
        
        Analyze this product/feature data for building AI-generated UIs:
        {product_info}
        
        Extract and structure:
        1. Company profile relevant to UI design
        2. Specific UI requirements and constraints
        3. Key data fields that need to be displayed
        4. User personas who will interact with the UI
        5. Required functionality and interactions
        6. Design recommendations for implementation
        
        Focus on aspects that will help an AI generate the perfect UI prompt.
        Consider data relationships, user flows, and business logic.
        """
    )
    
    # Use extended thinking for comprehensive UI requirements analysis
    chain = LLMChain(
        llm=llm_thinking,
        prompt=prompt_template
    )
    
    product_info_str = json.dumps(input_data, indent=2)
    response = chain.run(product_info=product_info_str)
    
    # Parse into structured format for Stitch
    parse_prompt = PromptTemplate(
        input_variables=["analysis"],
        template="""
        Based on this UI/product analysis, extract and structure the data as JSON for Stitch:
        {analysis}
        
        Return ONLY valid JSON matching this structure:
        {{
            "company_profile": "...",
            "ui_requirements": {{}},
            "data_fields": [...],
            "user_personas": [...],
            "functionality_needs": [...],
            "design_recommendations": "..."
        }}
        """
    )
    
    parse_chain = LLMChain(llm=llm, prompt=parse_prompt)
    structured_response = parse_chain.run(analysis=response)
    
    structured_data = json.loads(structured_response)
    
    return UIPromptResearch(**structured_data)


# ============================================================================
# Main Orchestration Function
# ============================================================================

def process_dual_path_workflow(input_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main workflow orchestrator
    Processes JSON data through both paths concurrently
    
    Args:
        input_json: Input JSON data
    
    Returns:
        Dictionary with results from both paths
    """
    
    print("🚀 Starting Dual-Path LangChain Automation")
    print("=" * 60)
    
    # Path 1: Outreach Research
    print("\n📊 Path 1: Running In-Depth Outreach Research...")
    outreach_results = path_1_outreach_research(input_json)
    print("✅ Outreach research complete")
    
    # Path 2: UI Prompt Research
    print("\n🎨 Path 2: Extracting UI Prompt Data...")
    ui_results = path_2_ui_prompt_research(input_json)
    print("✅ UI prompt data extraction complete")
    
    # Combine results
    combined_results = {
        "timestamp": str(__import__("datetime").datetime.now()),
        "outreach_path": outreach_results.model_dump(),
        "ui_path": ui_results.model_dump(),
        "next_manual_steps": {
            "outreach": "Review outreach_path results and send personalized messages",
            "ui_building": "Use ui_path data to create detailed prompt for Stitch UI builder"
        }
    }
    
    return combined_results


# ============================================================================
# Helper: Generate Stitch UI Prompt
# ============================================================================

def generate_stitch_prompt(ui_research: UIPromptResearch) -> str:
    """
    Generate a comprehensive prompt for Stitch UI builder
    Based on the research data from Path 2
    
    Args:
        ui_research: UIPromptResearch data
    
    Returns:
        String prompt ready for Stitch
    """
    
    prompt_template = PromptTemplate(
        input_variables=["ui_data"],
        template="""
        You are an expert Stitch UI builder. Using this research data, generate a detailed UI:
        
        {ui_data}
        
        Create a comprehensive, user-friendly UI that:
        - Displays all required data fields
        - Serves the identified user personas
        - Implements all required functionality
        - Follows the design recommendations
        - Is visually appealing and intuitive
        
        Generate the complete Stitch UI configuration.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    ui_data_str = json.dumps(ui_research.model_dump(), indent=2)
    
    stitch_prompt = chain.run(ui_data=ui_data_str)
    
    return stitch_prompt


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Example input JSON
    sample_input = {
        "company": "TechStartup Inc",
        "industry": "SaaS",
        "size": "50-100 employees",
        "current_tools": ["Salesforce", "HubSpot"],
        "pain_points": [
            "Manual data entry takes 20% of team time",
            "Data silos between sales and marketing",
            "Real-time reporting is slow"
        ],
        "budget_range": "$50k-$100k annually",
        "decision_timeline": "Q2 2024",
        "target_features": {
            "real_time_dashboards": True,
            "data_integration": True,
            "automated_workflows": True,
            "team_collaboration": True
        }
    }
    
    # Run the dual-path workflow
    results = process_dual_path_workflow(sample_input)
    
    # Display results
    print("\n" + "=" * 60)
    print("📋 WORKFLOW RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    
    # Save results to file
    output_file = "workflow_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")
    
    # Generate Stitch prompt from UI research
    print("\n" + "=" * 60)
    print("🎨 GENERATING STITCH UI PROMPT")
    print("=" * 60)
    stitch_prompt = generate_stitch_prompt(
        UIPromptResearch(**results["ui_path"])
    )
    print(stitch_prompt)
    
    # Save Stitch prompt
    stitch_file = "stitch_ui_prompt.txt"
    with open(stitch_file, "w") as f:
        f.write(stitch_prompt)
    print(f"\n💾 Stitch prompt saved to {stitch_file}")
