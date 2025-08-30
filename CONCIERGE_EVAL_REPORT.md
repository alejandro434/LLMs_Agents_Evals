# Concierge Workflow Evaluation Report

## Executive Summary

This report documents the comprehensive evaluation of the Concierge Workflow system, focusing on its conversational capabilities and job-search assistance effectiveness. The evaluation tested the system with three hypothetical job seekers through realistic, informal conversations.

**Overall Score: 48.1% (FAIL)**
- Initial Baseline: 28.4%
- Final Score After Improvements: 48.1%
- Target Score: 70%

## Evaluation Methodology

### Test Framework
- **Script**: `src/graphs/concierge_eval.py`
- **Test Users**: 3 hypothetical job seekers with different profiles
- **Conversation Turns**: 9 per user (27 total interactions)
- **Metrics Evaluated**:
  - Naturalness (conversational tone)
  - Helpfulness (actionable responses)
  - Task Completion (successful job search execution)

### Test User Profiles

1. **Jake Thompson** - Recent College Graduate
   - Location: Baltimore, MD
   - Status: Unemployed, just graduated
   - Experience: Internship at TechStartup
   - Seeking: Entry-level tech positions, flexible on location

2. **Maria Santos** - Career Changer
   - Location: Richmond, VA
   - Status: Employed (Sales Manager at RetailChain)
   - Seeking: Project management or operations roles, hybrid preferred

3. **Robert Chen** - Experienced Professional
   - Location: Norfolk, VA
   - Status: Unemployed (laid off)
   - Experience: Senior Engineer at DefenseContractor
   - Seeking: Senior engineering roles in defense/aerospace

## Key Findings

### Success Rate by User
| User | Profile Collection | Job Searches Handled | Status |
|------|-------------------|---------------------|---------|
| Jake Thompson | ✅ Complete | 5/5 | ✅ SUCCESS |
| Maria Santos | ❌ Incomplete | 0/5 | ❌ FAIL |
| Robert Chen | ✅ Complete | 4/5 | ✅ SUCCESS |

### Performance Metrics
- **Successful User Journeys**: 2/3 (66.7%)
- **Natural Responses**: 12/27 (44.4%)
- **Helpful Responses**: 9/27 (33.3%)
- **Average Response Quality**: 48.1%

## Critical Issues Identified

### 1. Profile Information Extraction Failure (CRITICAL)
**Affected User**: Maria Santos

**Issue**: The receptionist agent fails to properly extract job information when provided in the format "I'm a [Job Title] at [Company]"

**Example**:
```
User: "currently I'm a Sales Manager at RetailChain here in Richmond"
System: [Fails to extract both job title and company, keeps asking for the same information]
```

**Impact**: Complete failure for 1/3 of test users, preventing any job search assistance

**Root Cause**: Likely parsing issue in the receptionist's information extraction logic

### 2. Conversation Loop Problem
**Severity**: HIGH

**Description**: Agent gets stuck in repetitive questioning loops, asking for information already provided

**Observed Patterns**:
- Agent asks for job title and company repeatedly despite user providing it
- Agent doesn't recognize preferences when mixed with questions
- Agent fails to progress to handoff even with complete information

**Example**:
```
Turn 5: User provides job info
Turn 6: User provides preferences
Turn 7-9: Agent keeps asking for the same information
```

### 3. Inconsistent Preference Recognition
**Severity**: MEDIUM

**Issue**: System inconsistently recognizes job preferences

**Working Examples**:
- "entry level tech stuff" ✅
- "senior engineering, defense preferred" ✅

**Failing Examples**:
- "project management or operations roles, hybrid would be nice" ❌
- Mixed preference statements with questions ❌

### 4. Naturalness Deficiencies
**Severity**: MEDIUM

**Score**: 44.4% natural responses

**Issues**:
- Overly formal language in many responses
- Repetitive phrasing patterns
- Lack of empathy in some situations
- Robotic acknowledgments

### 5. Limited Helpfulness
**Severity**: MEDIUM

**Score**: 33.3% helpful responses

**Issues**:
- Many responses don't provide actionable next steps
- Generic responses instead of specific guidance
- Missed opportunities to be proactive

## Successful Behaviors

### 1. Recent Graduate Handling ✅
- Successfully extracted profile from informal conversation
- Handled vague preferences ("pretty flexible")
- Completed all 5 job search requests
- Provided relevant entry-level opportunities

### 2. Experienced Professional Support ✅
- Recognized unemployment situation with empathy
- Successfully searched for defense sector jobs
- Handled security clearance requirements
- Provided specific company recommendations

### 3. Job Search Execution
When profile collection succeeds, the system demonstrates:
- Effective web search for opportunities
- Relevant results for user requirements
- Multiple search strategies
- Specific company and role recommendations

## Improvement Areas

### Immediate Fixes Needed

1. **Fix Information Extraction**
   - Improve parsing of "Job Title at Company" format
   - Add validation to ensure all fields are captured
   - Implement fallback extraction strategies

2. **Prevent Conversation Loops**
   - Add state tracking to avoid re-asking answered questions
   - Implement maximum question limits
   - Force progression after certain thresholds

3. **Enhance Preference Recognition**
   - Expand pattern matching for preferences
   - Handle compound preference statements
   - Recognize preferences embedded in questions

### Medium-term Improvements

1. **Conversational Quality**
   - Add more varied response templates
   - Implement dynamic tone adjustment
   - Increase empathy and encouragement

2. **Proactive Assistance**
   - Anticipate user needs based on profile
   - Offer relevant suggestions unprompted
   - Provide industry-specific insights

3. **Error Recovery**
   - Implement graceful handling of parsing failures
   - Add clarification strategies
   - Provide alternative question formats

## Technical Observations

### System Architecture
- **Receptionist Subgraph**: Primary bottleneck for failures
- **ReAct Subgraph**: Performs well when given proper input
- **Handoff Mechanism**: Works when triggered but trigger conditions are problematic

### Configuration Files Modified
1. `receptionist_subgraph/system_prompt.yml` - Improved but insufficient
2. `receptionist_subgraph/fewshots.yml` - Added examples but parsing still fails
3. `ReAct_subgraph/prompts/system_prompt.yml` - Successfully improved
4. `ReAct_subgraph/prompts/fewshots.yml` - Good improvements

## Recommendations

### Priority 1 (Critical)
1. **Fix the "Job Title at Company" parsing bug**
2. **Implement robust information extraction validation**
3. **Add loop detection and prevention**

### Priority 2 (High)
1. **Expand preference recognition patterns**
2. **Improve conversation state management**
3. **Add comprehensive testing for edge cases**

### Priority 3 (Medium)
1. **Enhance conversational naturalness**
2. **Increase response variety**
3. **Add domain-specific knowledge**

## Conclusion

While the system shows promise with a 70% improvement from baseline (28.4% to 48.1%), it falls short of production readiness. The critical failure with career changers (33% of users) and overall poor conversation quality indicate significant work is needed.

**Current State**: NOT PRODUCTION READY
**Recommendation**: Address Priority 1 issues before any deployment
**Estimated Effort**: 2-3 development sprints for full remediation

## Appendix: Detailed Test Results

### Successful Interactions
- Jake Thompson: 5/5 job searches completed
- Robert Chen: 4/5 job searches completed
- Response time: Generally acceptable
- Search quality: Good when executed

### Failed Interactions
- Maria Santos: 0/5 job searches (complete failure)
- 15/27 interactions resulted in repetitive loops
- 18/27 interactions lacked helpful content

### Performance Trajectory
1. Baseline: 28.4% (Complete system failure)
2. After prompt improvements: 48.1% (Partial success)
3. Target: 70% (Not achieved)
4. Production ready: 80%+ (Far from target)

---

*Report Generated: December 2024*
*Evaluation Framework: `src/graphs/concierge_eval.py`*
*System Version: BuildWithin Evals v0.1.0*
