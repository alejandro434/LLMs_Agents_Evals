"""Qualifier System Prompt Review.

A comprehensive analysis of coherence, clarity, consistency, and ZIP code
accuracy in the qualifier subgraph prompts.
"""

# %%
# Qualifier System Prompt Quality Review

## Executive Summary

The qualifier system prompts have several critical issues that need immediate
attention:

1. **Missing Washington D.C. Support**: Despite prompts claiming D.C. residents
   are qualified, the implementation doesn't handle D.C. ZIP codes
2. **Inaccurate ZIP Code Ranges**: The documented ranges don't match actual ZIP
   code allocations
3. **Inconsistent Messaging**: Different parts of the system claim different
   qualifying states
4. **Clarity Issues**: Ambiguous language and incomplete instructions

## Critical Issues

### 1. Washington D.C. ZIP Code Support Missing

**Problem**: The prompts state that D.C. residents are qualified, but the
implementation (`chains.py`) has no logic to handle D.C. ZIP codes.

- Prompts claim: "Only residents of Maryland (MD), Virginia (VA), and
  Washington, D.C. (DC) are qualified"
- Prompts show D.C. ZIP: "Washington, D.C. (DC): 200xx"
- Implementation: NO handling for ZIP codes 20000-20099

**Impact**: D.C. residents will be incorrectly rejected even though prompts say
they qualify.

### 2. Incorrect ZIP Code Ranges

**Current Prompt Ranges**:
- Maryland: 206xx–219xx
- Virginia: 201xx–246xx
- D.C.: 200xx

**Actual Implementation** (`chains.py`):
- Maryland: 20600–21999 ✓ (matches prompt)
- Virginia: 20100–20199 and 22000–24699 ✗ (doesn't match "201xx–246xx")

**Real ZIP Code Allocations**:
- D.C.: 20001-20099, 20201-20599 (missing from implementation)
- Maryland: 20600-21999 (correct)
- Virginia: 20100-20199, 22000-24699 (implementation correct, prompt wrong)

### 3. Inconsistent State Qualification Rules

**UserInfoOutputSchema** (schemas.py line 36):
> "ONLY states of Maryland and Virginia are qualified."

**QualifierSystemPrompt** (system_prompt.yml line 20):
> "Only residents of Maryland (MD), Virginia (VA), and Washington, D.C. (DC)
> are qualified"

This creates confusion about whether D.C. residents qualify.

## Detailed Analysis

### Coherence Issues

1. **Conflicting Instructions**: The prompts mention D.C. as qualified, but the
   schema documentation excludes it
2. **Incomplete Examples**: The ZIP→state examples are marked as
   "not exhaustive" but miss critical D.C. ranges
3. **Mixed Terminology**: Uses both full state names and abbreviations
   inconsistently

### Clarity Issues

1. **Ambiguous ZIP Notation**: "206xx–219xx" could mean 20600-21999 or
   20600-21900
2. **Missing Edge Cases**: No guidance for:
   - Invalid ZIP codes
   - ZIP codes outside the three jurisdictions
   - Partial ZIP codes
3. **Unclear Priority**: "Prefer the ZIP-derived state" - what happens if ZIP
   can't be mapped?

### Consistency Issues

1. **Schema vs Prompt Mismatch**: Different qualifying states listed
2. **Implementation vs Documentation Gap**: Code doesn't match prompt specs
3. **Response Format**: UserInfoCollectionSystemPrompt requires
   `direct_response_to_the_user` but doesn't explain when/why

## Recommendations

### Immediate Fixes Required

1. **Add D.C. ZIP Code Support** in `chains.py`:

```python
def _infer_state_from_zip(zip5: int) -> str | None:
    """Map a 5-digit ZIP code to a US state.

    Returns the full state name if recognized, else None.
    Ranges (inclusive):
      - Washington D.C.: 20001–20099, 20201–20599
      - Maryland: 20600–21999
      - Virginia: 20100–20199, 22000–24699
      - California: 90000–96199
      - New York: 10000–14999
      - Texas: 75000–79999, 88500–88599
    """
    # Washington D.C.
    if 20001 <= zip5 <= 20099 or 20201 <= zip5 <= 20599:
        return "District of Columbia"
    # Maryland
    if 20600 <= zip5 <= 21999:
        return "Maryland"
    # Virginia (excluding D.C. ranges)
    if 20100 <= zip5 <= 20200 or 22000 <= zip5 <= 24699:
        return "Virginia"
    # ... rest of states
```

2. **Update Prompts** with accurate ranges:

```yaml
UserInfoCollectionSystemPrompt: |
  You are a helpful assistant that collects user information: age, state,
  and US ZIP code. If the user provides a ZIP code, infer the state from
  the ZIP and populate "state" accordingly. Prefer the ZIP-derived state if
  it conflicts with a textual state mention. If state is missing but a ZIP is
  present, infer the state. If neither state nor ZIP is present, ask a concise
  follow-up question.

  ZIP code to state mapping (use these exact ranges):
    - Washington, D.C. (DC): 20001–20099, 20201–20599
    - Maryland (MD): 20600–21999
    - Virginia (VA): 20100–20200, 22000–24699

  If ZIP code cannot be mapped, keep any textual state mention.

  IMPORTANT: YOU MUST ALWAYS write a direct_response_to_the_user field.
```

3. **Fix Schema Documentation**:

Update `schemas.py` line 36:
```python
description=(
    "The U.S. state where the user is located. If a ZIP code is"
    " provided, infer the state from the ZIP and prefer it over any"
    " conflicting textual state. ONLY residents of Maryland, Virginia,"
    " and Washington D.C. are qualified."
),
```

### Additional Improvements

1. **Add Validation Tests** for D.C. ZIP codes
2. **Clarify Edge Case Handling** in prompts
3. **Use Consistent State Naming** (full names or abbreviations, not both)
4. **Add Examples** for D.C. residents in fewshots.yml
5. **Document Response Field Requirements** more clearly

## Testing Recommendations

Add these test cases:

```python
# D.C. resident tests
assert _infer_state_from_zip(20001) == "District of Columbia"  # D.C. start
assert _infer_state_from_zip(20099) == "District of Columbia"  # D.C. end
assert _infer_state_from_zip(20201) == "District of Columbia"  # D.C. second range
assert _infer_state_from_zip(20599) == "District of Columbia"  # D.C. second range end

# Boundary tests
assert _infer_state_from_zip(20000) is None  # Before D.C.
assert _infer_state_from_zip(20100) == "Virginia"  # VA Northern
assert _infer_state_from_zip(20200) == "Virginia"  # VA Northern end
assert _infer_state_from_zip(20600) == "Maryland"  # MD start
```

## Conclusion

The qualifier prompts have significant accuracy and consistency issues that will
cause incorrect user qualification decisions. The most critical issue is the
missing D.C. support despite claims that D.C. residents qualify. These issues
should be fixed immediately to ensure the system works as intended.


if __name__ == "__main__":
    print("Qualifier Prompt Review Complete")
    print("Critical Issues Found: 3")
    print("Recommendations: 8")
    print("Please review and implement fixes immediately.")
