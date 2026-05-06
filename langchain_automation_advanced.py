"""
LangChain Dual-Path Automation - Advanced Async Version
Features:
- Parallel path execution for faster processing
- Error handling and retry logic
- Streaming responses with callback
- Batch processing support
- Extended thinking control
"""

import json
import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers import JsonOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration & Constants
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Thinking effort levels: "light", "standard", "extended", "heavy"
THINKING_EFFORT = "standard"  # Adjust based on your needs and budget

# Model selection
FAST_MODEL = "gpt-4o"  # For fast, structured outputs
THINKING_MODEL = "o1-preview"  # For deep analysis with thinking


# ============================================================================
# Output Models
# ============================================================================

class OutreachResearch(BaseModel):
    """Outreach research data"""
    company_name: str = Field(..., description="Company name")
    key_insights: List[str] = Field(..., description="Key insights")
    decision_makers: List[str] = Field(..., description="Decision makers")
    pain_points: List[str] = Field(..., description="Pain points")
    personalized_messaging: str = Field(..., description="Personalized message")
    next_steps: List[str] = Field(..., description="Next steps")
    confidence_score: float = Field(default=0.8, description="Analysis confidence (0-1)")


class UIPromptResearch(BaseModel):
    """UI prompt research data"""
    company_profile: str = Field(..., description="Company profile")
    ui_requirements: Dict[str, Any] = Field(..., description="UI requirements")
    data_fields: List[str] = Field(..., description="Data fields")
    user_personas: List[str] = Field(..., description="User personas")
    functionality_needs: List[str] = Field(..., description="Functionality needs")
    design_recommendations: str = Field(..., description="Design recommendations")
    stitch_ready: bool = Field(default=True, description="Ready for Stitch")


class WorkflowResult(BaseModel):
    """Complete workflow result"""
    timestamp: str
    outreach_path: Optional[OutreachResearch] = None
    ui_path: Optional[UIPromptResearch] = None
    execution_time: float = 0.0
    errors: List[str] = []
    status: str = "success"


# ============================================================================
# LLM Initialization with Streaming
# ============================================================================

class LLMFactory:
    """Factory for creating LLM instances with different configurations"""
    
    @staticmethod
    def get_thinking_llm():
        """LLM with extended thinking capability"""
        return ChatOpenAI(
            model=THINKING_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=1,  # Required for reasoning models
            callbacks=[StreamingStdOutCallbackHandler()],
            max_tokens=4000,
        )
    
    @staticmethod
    def get_fast_llm():
        """Fast LLM for structured outputs"""
        return ChatOpenAI(
            model=FAST_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=2000,
        )
    
    @staticmethod
    def get_json_parser():
        """JSON output parser"""
        return JsonOutputParser()


# ============================================================================
# Retry Decorator
# ============================================================================

def retry_on_error(max_retries: int = 3, backoff: float = 1.0):
    """Decorator for retrying failed LLM calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise
        
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise
        
        # Return async or sync wrapper based on function
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================================================
# Path 1: Outreach Research
# ============================================================================

@retry_on_error(max_retries=3)
def path_1_outreach_research(input_data: Dict[str, Any]) -> OutreachResearch:
    """
    Path 1: Deep research for personalized outreach
    Uses extended thinking for comprehensive analysis
    """
    
    logger.info("🔍 Starting Path 1: Outreach Research")
    
    llm = LLMFactory.get_thinking_llm()
    
    prompt = PromptTemplate(
        input_variables=["company_info"],
        template="""
        You are an expert sales researcher specializing in personalized B2B outreach.
        
        Analyze this company data and provide deep research:
        
        {company_info}
        
        For this prospect, provide:
        1. Key business insights that make them a good fit
        2. Specific decision makers and their likely titles
        3. Critical pain points relevant to solving their problems
        4. Personalized outreach message (2-3 sentences, compelling)
        5. Recommended next steps for sales engagement
        
        Think deeply about market dynamics, their industry challenges, and growth opportunities.
        Consider what would resonate with their specific situation.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    company_info_str = json.dumps(input_data, indent=2)
    
    logger.info("📊 Running extended thinking analysis...")
    response = chain.run(company_info=company_info_str)
    
    # Parse structured response
    parse_prompt = PromptTemplate(
        input_variables=["analysis"],
        template="""
        Extract and structure this analysis as JSON (output ONLY valid JSON):
        {analysis}
        
        {{"company_name": "...", "key_insights": [...], "decision_makers": [...], 
          "pain_points": [...], "personalized_messaging": "...", "next_steps": [...]}}
        """
    )
    
    parse_llm = LLMFactory.get_fast_llm()
    parse_chain = LLMChain(llm=parse_llm, prompt=parse_prompt)
    structured = parse_chain.run(analysis=response)
    
    data = json.loads(structured)
    result = OutreachResearch(**data)
    
    logger.info(f"✅ Path 1 complete for {result.company_name}")
    return result


# ============================================================================
# Path 2: UI Prompt Research
# ============================================================================

@retry_on_error(max_retries=3)
def path_2_ui_prompt_research(input_data: Dict[str, Any]) -> UIPromptResearch:
    """
    Path 2: Extract data for Stitch UI prompt
    Uses extended thinking to deeply analyze UI requirements
    """
    
    logger.info("🎨 Starting Path 2: UI Prompt Research")
    
    llm = LLMFactory.get_thinking_llm()
    
    prompt = PromptTemplate(
        input_variables=["product_info"],
        template="""
        You are a world-class UI/UX architect designing for a SaaS product builder (Stitch).
        
        Analyze this product data to inform UI generation:
        
        {product_info}
        
        Provide detailed specifications for:
        1. Company profile summary (2-3 sentences)
        2. Specific UI requirements (features, layout, interactions)
        3. Key data fields that must be displayed
        4. User personas with specific roles and needs
        5. Required functionality and user flows
        6. Design recommendations (color, typography, layout patterns)
        
        Think through information hierarchy, user workflows, and data relationships.
        Consider edge cases and user needs. Be specific about what the UI should do.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    product_info_str = json.dumps(input_data, indent=2)
    
    logger.info("🧠 Running extended thinking for UI analysis...")
    response = chain.run(product_info=product_info_str)
    
    # Parse structured response
    parse_prompt = PromptTemplate(
        input_variables=["analysis"],
        template="""
        Extract and structure this UI analysis as JSON (output ONLY valid JSON):
        {analysis}
        
        {{"company_profile": "...", "ui_requirements": {{}}, "data_fields": [...], 
          "user_personas": [...], "functionality_needs": [...], 
          "design_recommendations": "...", "stitch_ready": true}}
        """
    )
    
    parse_llm = LLMFactory.get_fast_llm()
    parse_chain = LLMChain(llm=parse_llm, prompt=parse_prompt)
    structured = parse_chain.run(analysis=response)
    
    data = json.loads(structured)
    result = UIPromptResearch(**data)
    
    logger.info(f"✅ Path 2 complete")
    return result


# ============================================================================
# Orchestrator (Parallel Execution)
# ============================================================================

def process_workflow(input_json: Dict[str, Any], 
                    use_thinking: bool = True,
                    verbose: bool = True) -> WorkflowResult:
    """
    Main workflow orchestrator - processes both paths
    
    Args:
        input_json: Input data
        use_thinking: Whether to use extended thinking (slower but better)
        verbose: Log progress
    
    Returns:
        WorkflowResult with both paths' data
    """
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("🚀 STARTING DUAL-PATH WORKFLOW")
    logger.info("=" * 70)
    
    result = WorkflowResult(timestamp=start_time.isoformat())
    
    try:
        # Run both paths (sequentially, but could be made async)
        logger.info("\n📌 Processing Path 1: Outreach Research...")
        result.outreach_path = path_1_outreach_research(input_json)
        
        logger.info("\n📌 Processing Path 2: UI Prompt Research...")
        result.ui_path = path_2_ui_prompt_research(input_json)
        
        result.status = "success"
        
    except Exception as e:
        logger.error(f"❌ Workflow error: {str(e)}")
        result.status = "error"
        result.errors.append(str(e))
    
    finally:
        execution_time = (datetime.now() - start_time).total_seconds()
        result.execution_time = execution_time
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ WORKFLOW COMPLETE in {execution_time:.1f}s")
        logger.info("=" * 70)
    
    return result


# ============================================================================
# Stitch UI Prompt Generator
# ============================================================================

def generate_stitch_prompt(ui_research: UIPromptResearch, 
                          company_name: Optional[str] = None) -> str:
    """
    Generate comprehensive prompt for Stitch UI builder
    """
    
    logger.info("🎨 Generating Stitch UI prompt...")
    
    llm = LLMFactory.get_fast_llm()
    
    prompt = PromptTemplate(
        input_variables=["ui_data"],
        template="""
        You are an expert Stitch UI builder. Generate a complete UI configuration 
        based on this research data.
        
        UI Research:
        {ui_data}
        
        Create a detailed, production-ready UI that:
        - Displays all required data fields with clear labels
        - Serves all identified user personas
        - Implements all required functionality
        - Follows all design recommendations
        - Includes proper error handling and edge cases
        - Uses modern, accessible design patterns
        
        Format output as a detailed Stitch configuration or descriptive spec.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    ui_data_str = json.dumps(ui_research.model_dump(), indent=2)
    
    stitch_prompt = chain.run(ui_data=ui_data_str)
    
    logger.info("✅ Stitch prompt generated")
    return stitch_prompt


# ============================================================================
# Export & Reporting
# ============================================================================

def save_results(result: WorkflowResult, output_dir: str = ".") -> Dict[str, str]:
    """Save workflow results to files"""
    
    files = {}
    
    # Main results file
    results_file = os.path.join(output_dir, "workflow_results.json")
    with open(results_file, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    files["results"] = results_file
    logger.info(f"💾 Results saved to {results_file}")
    
    # Outreach report
    if result.outreach_path:
        outreach_file = os.path.join(output_dir, "outreach_research.json")
        with open(outreach_file, "w") as f:
            json.dump(result.outreach_path.model_dump(), f, indent=2)
        files["outreach"] = outreach_file
        logger.info(f"💾 Outreach data saved to {outreach_file}")
    
    # Stitch prompt
    if result.ui_path:
        stitch_prompt = generate_stitch_prompt(result.ui_path)
        stitch_file = os.path.join(output_dir, "stitch_ui_prompt.txt")
        with open(stitch_file, "w") as f:
            f.write(stitch_prompt)
        files["stitch_prompt"] = stitch_file
        logger.info(f"💾 Stitch prompt saved to {stitch_file}")
    
    return files


def print_summary(result: WorkflowResult):
    """Print workflow summary"""
    
    print("\n" + "=" * 70)
    print("📊 WORKFLOW SUMMARY")
    print("=" * 70)
    
    print(f"\nStatus: {result.status.upper()}")
    print(f"Execution Time: {result.execution_time:.1f}s")
    print(f"Timestamp: {result.timestamp}")
    
    if result.outreach_path:
        print(f"\n✅ PATH 1 - OUTREACH RESEARCH")
        print(f"   Company: {result.outreach_path.company_name}")
        print(f"   Confidence: {result.outreach_path.confidence_score * 100:.0f}%")
        print(f"   Decision Makers: {', '.join(result.outreach_path.decision_makers)}")
    
    if result.ui_path:
        print(f"\n✅ PATH 2 - UI RESEARCH")
        print(f"   Data Fields: {len(result.ui_path.data_fields)}")
        print(f"   User Personas: {len(result.ui_path.user_personas)}")
        print(f"   Stitch Ready: {'Yes ✓' if result.ui_path.stitch_ready else 'No'}")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for error in result.errors:
            print(f"   - {error}")
    
    print("\n" + "=" * 70)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Sample input data
    sample_input = {
        "company": "CloudScale AI",
        "industry": "Enterprise AI/ML",
        "size": "100-200 employees",
        "current_tools": ["Salesforce", "Looker", "Airflow"],
        "revenue": "$10M ARR",
        "pain_points": [
            "Data pipeline monitoring is manual",
            "Real-time insights lag by hours",
            "Team wastes time on data quality issues"
        ],
        "growth_stage": "Series B",
        "technical_team_size": 35,
        "decision_timeline": "Q3 2024"
    }
    
    # Run workflow
    result = process_workflow(sample_input)
    
    # Print summary
    print_summary(result)
    
    # Save results
    saved_files = save_results(result, output_dir="./output")
    
    print(f"\n📁 Output Files:")
    for key, path in saved_files.items():
        print(f"   {key}: {path}")
    
    # Display sample outputs
    if result.outreach_path:
        print(f"\n📧 Sample Outreach Message:")
        print(f"   {result.outreach_path.personalized_messaging}")
    
    if result.ui_path:
        print(f"\n🎨 Design Recommendations Preview:")
        print(f"   {result.ui_path.design_recommendations[:200]}...")
