# LangChain Dual-Path Automation Workflow Guide

## 📋 Overview

This automation processes JSON input data through **two independent paths simultaneously**:
- **Path 1**: In-depth research for personalized outreach (uses extended thinking)
- **Path 2**: Extract research data for UI prompt building in Stitch

## 🧠 Extended Thinking Capability

**What is Extended Thinking?**

OpenAI's extended thinking allows models to spend more time reasoning on complex problems, using a router that decides based on conversation type, complexity, and explicit intent.

**In This Implementation:**

The automation uses **OpenAI's reasoning models** (o1-preview, o3) which have extended thinking capability:

```
User Input JSON
       ↓
    [THINKING PHASE]
    - Analyze context deeply
    - Reason through complexities
    - Consider multiple angles
    - Build mental model
       ↓
    [RESPONSE PHASE]
    - Provide structured output
    - Based on deep analysis
```

**How to Enable Extended Thinking:**

1. Use `o1-preview` or `o3` models (they have thinking built-in)
2. Set `temperature=1` (required for reasoning models)
3. The model automatically allocates thinking time based on task complexity
4. You can control thinking depth with options like Light, Standard, Extended, or Heavy

**Cost & Latency Trade-off:**
- ✅ More accurate results, especially for complex analysis
- ⏱️ Longer response times (thinking takes time)
- 💰 Higher per-token cost
- Best for: Research, analysis, complex reasoning tasks

## 🏗️ Architecture

```
INPUT JSON
    ↓
┌───────────────────────────────────────────┐
│   LangChain Workflow Orchestrator         │
└───────────────────────────────────────────┘
    ↓
    ├─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
┌──────────────────────┐              ┌──────────────────────┐
│ PATH 1: OUTREACH     │              │ PATH 2: UI RESEARCH  │
│ Research (Extended   │              │ (Extended Thinking)  │
│ Thinking)            │              │                      │
└──────────────────────┘              └──────────────────────┘
    ↓                                     ↓
    ├─ Key Insights                      ├─ Company Profile
    ├─ Decision Makers                   ├─ UI Requirements
    ├─ Pain Points                       ├─ Data Fields
    ├─ Personalization                   ├─ User Personas
    └─ Next Steps                        ├─ Functionality
                                         └─ Design Recs
    ↓                                     ↓
┌─────────────────────────────────────────────────┐
│   Structured Output (JSON)                      │
│   + Auto-generated Stitch UI Prompt             │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│   MANUAL STEP 1:                                │
│   Review & send personalized outreach           │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│   MANUAL STEP 2:                                │
│   Paste prompt into Stitch to build UI          │
└─────────────────────────────────────────────────┘
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file in project root:

```
OPENAI_API_KEY=sk-your-api-key-here
```

**Important Notes on Models:**

- `gpt-4o`: Latest general-purpose model (supports all features)
- `o1-preview`: Reasoning model with extended thinking (slower, more expensive, better for analysis)
- `o3`: Latest reasoning model (when available, recommended for extended thinking)

**Model Selection Guide:**

```python
# For extended thinking (recommended for research paths):
llm_thinking = ChatOpenAI(model="o1-preview", temperature=1)

# For fast structured outputs:
llm_fast = ChatOpenAI(model="gpt-4o", temperature=0.7)
```

### 3. Run the Automation

```bash
python langchain_automation.py
```

This will:
1. Process input JSON through both paths
2. Generate structured research data
3. Create Stitch UI prompt
4. Save results to `workflow_results.json`
5. Save Stitch prompt to `stitch_ui_prompt.txt`

## 📊 Path Outputs

### Path 1: Outreach Research Output

```json
{
  "company_name": "TechStartup Inc",
  "key_insights": [
    "Growing SaaS company with clear data integration needs",
    "Team is tech-savvy and ready for automation"
  ],
  "decision_makers": ["VP Sales", "CTO", "Head of Marketing"],
  "pain_points": [
    "Manual data entry overhead",
    "Data silos between departments",
    "Slow reporting cycles"
  ],
  "personalized_messaging": "Hi [Name], We noticed TechStartup is tackling...",
  "next_steps": [
    "Send personalized email with case study",
    "Schedule discovery call",
    "Prepare ROI analysis"
  ]
}
```

### Path 2: UI Research Output

```json
{
  "company_profile": "SaaS company, 50-100 people, focus on data...",
  "ui_requirements": {
    "real_time_dashboards": true,
    "data_integration": true
  },
  "data_fields": ["user_id", "transaction_amount", "timestamp", ...],
  "user_personas": ["Sales Manager", "Data Analyst", "Executive"],
  "functionality_needs": [
    "Real-time data refresh",
    "Custom filters",
    "Export functionality"
  ],
  "design_recommendations": "Clean, minimal UI with focus on KPIs..."
}
```

## 🎨 Using the Output in Stitch

After the automation runs, use the generated `stitch_ui_prompt.txt`:

```
1. Open Stitch UI Builder
2. Click "New UI"
3. Select "AI-Generated"
4. Paste entire content from stitch_ui_prompt.txt
5. Review generated UI
6. Customize as needed
7. Deploy
```

## 🔄 Workflow Customization

### Modify Input Schema

Edit the `sample_input` dictionary to include your specific fields:

```python
sample_input = {
    "company": "Your Company",
    "custom_field_1": "value",
    "custom_field_2": ["list", "of", "values"],
    # Add more fields as needed
}
```

### Adjust Thinking Depth

Control how much time the model spends thinking:

```python
# Current: Auto (model decides based on complexity)
# To increase thinking time:

llm_thinking = ChatOpenAI(
    model="o1-preview",
    temperature=1,
    # Add if API supports it in future:
    # max_thinking_length=10000  # More thinking
)
```

### Add Custom Parsing

Extend the Pydantic models for additional fields:

```python
class OutreachResearch(BaseModel):
    company_name: str
    # ... existing fields ...
    custom_field: str = Field(..., description="Your custom field")
```

## 💡 Best Practices

### When to Use Extended Thinking

✅ **Use for:**
- Complex analysis and research
- Strategic decision-making
- Multi-step reasoning problems
- High-stakes content (outreach, proposals)

❌ **Don't use for:**
- Simple data extraction
- Quick transformations
- Real-time API responses
- High-volume, low-complexity tasks

### Optimizing Prompts

1. **Be Specific**: Clearly state what you want
2. **Provide Context**: Include industry, company size, goals
3. **Structure Output**: Use JSON output parsers
4. **Iterate**: Refine prompts based on results

### Cost Optimization

```python
# Hybrid approach: Use thinking only when needed
if task_complexity == "high":
    llm = llm_thinking  # Use o1/o3
else:
    llm = llm_fast      # Use gpt-4o
```

## 🔧 Troubleshooting

### Issue: High token costs

**Solution:**
- Use `gpt-4o` for simple tasks
- Reduce prompt verbosity
- Cache common prompts using LangChain caching

### Issue: Slow responses

**Solution:**
- Use `gpt-4o` instead of reasoning models
- Reduce thinking time (if API supports)
- Run paths in parallel (add async)

### Issue: Inconsistent output

**Solution:**
- Fix `temperature=1` for reasoning models (required)
- Use stricter Pydantic validation
- Add explicit output format instructions

## 📚 Model Comparison

| Model | Speed | Cost | Thinking | Best For |
|-------|-------|------|----------|----------|
| GPT-4o | ⭐⭐⭐⭐⭐ | $ | ❌ | Fast responses, general tasks |
| o1-preview | ⭐⭐ | $$$ | ✅ | Complex analysis, reasoning |
| o3 | ⭐⭐⭐ | $$ | ✅ | Balanced reasoning (when available) |
| o3-mini | ⭐⭐⭐⭐ | $ | ✅ | Cost-effective reasoning |

## 🚀 Future Enhancements

1. **Async Processing**: Run both paths concurrently for faster execution
2. **Real-time Streaming**: Stream thinking process to UI
3. **Multi-Model Orchestration**: Different models for different tasks
4. **Caching Layer**: Cache repeated analysis
5. **Feedback Loop**: Improve prompts based on results
6. **Integration with APIs**: Direct Salesforce/HubSpot integration
7. **Batch Processing**: Process multiple companies simultaneously

## 📖 OpenAI Extended Thinking References

- GPT-5 includes extended reasoning models for harder problems with automatic routing based on complexity
- OpenAI o3 and o4-mini are reasoning models that think longer before responding
- GPT-5.2 Thinking outperforms human experts on professional knowledge work tasks

## 🎯 Next Steps

1. **Setup**: Install dependencies and configure API key
2. **Test**: Run with sample_input to understand output
3. **Customize**: Modify for your specific use case
4. **Integrate**: Connect to your data sources
5. **Scale**: Add async and batch processing
6. **Monitor**: Track costs and performance

---

**Questions?** Check OpenAI docs: https://platform.openai.com/docs/guides/reasoning
