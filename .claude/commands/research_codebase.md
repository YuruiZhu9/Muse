# Research Codebase

## Context Check
{{#cursor}}Before starting, check context usage with /context command{{/cursor}}

## Objective
Research the codebase to understand the current state before implementing changes.

## Inputs
- Feature/bug description: {{featureDescription}}
- Research scope: {{scope}}
- Focus areas (optional): {{focusAreas}}

## Process

### Step 1: Parallel Research (4 Agents)
Launch 4 parallel research threads:
1. **Code Pattern Searcher** - Find relevant files and code patterns
2. **Architecture Analyzer** - Understand current system design
3. **Edge Case Hunter** - Identify potential edge cases
4. **Dependency Mapper** - Map dependencies and integrations

### Step 2: Gather Findings
Each agent must report:
- Discovery summary
- **file:line references** (critical for Plan phase)
- Architecture insights
- Unanswered questions

### Step 3: Consolidate
Synthesize findings into a single research document.

## Output Format
Save to: `thoughts/shared/research/{{YYYY-MM-DD}}-{{slug}}.md`

```markdown
# Research: {{featureTitle}}

**Date:** {{date}}
**Scope:** {{scope}}

## Key Findings

### Files Identified
| File | Lines | Purpose |
|------|-------|---------|
| src/auth/login.ts:45-67 | Login handler | Current OAuth2 flow |
| ... | ... | ... |

### Architecture Insights
- [Bullet points on current architecture]
- [Integration points]

### Edge Cases Discovered
1. [Case 1]
2. [Case 2]

### Open Questions
- [? Question 1]
- [? Question 2]

## References
- {{file:line}} citations throughout
```

## Context Management
⚠️ **Personalized:** Follows `.claude/user-preferences.md`
- At 60-80%: Ask user before clearing
- At >80%: Auto-clear and reload
- Git operations: Skip if not in git repo

## Commands
```bash
# Check context
/context

# User will be prompted if 60-80%
# Auto-clear if >80%

# Check git status (if in git repo)
git status 2>/dev/null && echo "In git repo" || echo "Not in git repo"
```

---

**Note:** See `.claude/user-preferences.md` for full customization.
