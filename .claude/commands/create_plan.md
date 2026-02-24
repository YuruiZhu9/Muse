# Create Implementation Plan

## Context Check
{{#cursor}}Check /context before starting{{/cursor}}
⚠️ Follows `.claude/user-preferences.md` - will prompt at 60-80% threshold

## Objective
Create a detailed implementation plan based on research findings.

## Inputs
- Research document: `thoughts/shared/research/{{YYYY-MM-DD}}-{{slug}}.md`
- Feature requirements: {{requirements}}
- Success criteria: {{successCriteria}}

## Process

### Step 1: Load Research
Read the research document and understand:
- Current architecture
- Files to modify
- Edge cases to handle

### Step 2: Create Plan Structure
Divide work into **phases** (3-5 phases typical):
- Phase 1: Setup/infrastructure
- Phase 2: Core implementation
- Phase 3: Integration
- Phase 4: Testing/validation

### Step 3: For Each Phase, Specify

#### What to Change
| File | Path | Changes | Lines |
|------|------|---------|-------|
| Component | src/auth/oauth.ts | Add token refresh | 120-145 |

#### Code Snippets
```typescript
// Example: src/auth/oauth.ts:120-145
async function refreshToken() {
  // Implementation here
}
```

#### Verification
**Auto-validation:**
```bash
npm test -- auth.test.ts
npm run lint
```

**Manual validation:**
- [ ] Test token refresh in dev environment
- [ ] Verify token expiry handling

**Success criteria:**
- Token refresh works without user action
- Failed refresh triggers re-authentication

## Output Format
Save to: `thoughts/shared/plans/{{YYYY-MM-DD}}-{{slug}}.md`

```markdown
# Implementation Plan: {{featureTitle}}

**Date:** {{date}}
**Research:** Based on `research/{{YYYY-MM-DD}}-{{slug}}.md`
**Estimated Phases:** {{n}}

---

## Phase {{n}}: {{Phase Name}}

### Objective
[Brief goal statement]

### Files to Modify
| File | Lines | Change Type | Description |
|------|-------|-------------|-------------|
| ... | ... | ... | ... |

### Implementation Details

#### File: {{path}}
**Location:** Lines {{start}}-{{end}}
**Change:** {{description}}

```{{language}}
{{codeSnippet}}
```

### Verification
**Auto:**
```bash
{{command}}
```

**Manual:**
- [ ] {{check1}}
- [ ] {{check2}}

**Success:**
- {{criteria}}

---

## Overall Success Criteria
{{overallCriteria}}

## Risk Mitigation
| Risk | Mitigation |
|------|------------|
| ... | ... |
```

## Iteration Guidelines
- **Minimum iterations:** 5
- **Optimal iterations:** 5-6
- **Time investment:** 30-45 minutes
- Each iteration should refine:
  - Phase boundaries
  - File locations (verify with {{Glob}})
  - Verification commands
  - Success criteria

## Context Management
- After 2-3 iterations, check /context
- If approaching 60%, /clear and reload plan doc
- Use references like "see Phase 2, src/auth/oauth.ts:120-145"
- NOT full code snippets in every iteration

## Commands
```bash
# Load research
cat thoughts/shared/research/{{date}}-{{slug}}.md

# Verify file exists
ls -la src/auth/oauth.ts

# Check context
/context
```
