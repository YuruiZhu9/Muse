# Validate Implementation

## Context Check
{{#cursor}}Check /context before starting{{/cursor}}
⚠️ Follows `.claude/user-preferences.md` - prompts before clearing at 60-80%

## Objective
Generate comprehensive validation report comparing plan vs actual implementation.

## Inputs
- Plan document: `thoughts/shared/plans/{{YYYY-MM-DD}}-{{slug}}.md`
- Feature name: {{featureName}}

## Process

### Step 1: Load Plan
Read the complete implementation plan.

### Step 2: Compare Plan vs Reality
For EACH phase in the plan:

#### Check Implementation
- [ ] Files modified as specified
- [ ] Code matches planned snippets
- [ ] Lines modified within specified ranges

#### Run Verification
- [ ] Execute all auto-validation commands
- [ ] Check test results
- [ ] Run linting/format checks

#### Document Deviations
```markdown
## Phase {{n}} Analysis

**Implementation Status:** ✅ COMPLETE / ⚠️ PARTIAL / ❌ MISSING

**Matches Plan:** Yes / No

**Deviations:**
- {{description of deviation}}

**Test Results:**
- Auto: {{pass/fail}} - {{details}}
- Manual: {{pending/passed/failed}}

**Recommendations:**
- {{action items}}
```

### Step 3: Generate Validation Report
Create comprehensive report covering:

#### 1. Implementation Coverage
| Phase | Planned | Implemented | Match % | Notes |
|-------|---------|-------------|---------|-------|
| Phase 1 | 3 files | 3 files | 100% | - |
| Phase 2 | 2 files | 2 files | 100% | Minor deviation |
| Phase 3 | 5 files | 4 files | 80% | Missing X |
| Phase 4 | - | - | - | Not started |

#### 2. Quality Checks
- Code quality: {{analysis}}
- Test coverage: {{coverage}}%
- Linting: {{clean / issues found}}
- Security: {{no concerns / concerns}}

#### 3. Functional Verification
| Feature | Planned | Actual | Status |
|---------|---------|--------|--------|
| Token refresh | Automatic | Works | ✅ |
| Error handling | Comprehensive | Partial | ⚠️ |
| ... | ... | ... | ... |

#### 4. Outstanding Items
**Must Fix:**
- [ ] {{critical item 1}}
- [ ] {{critical item 2}}

**Should Fix:**
- [ ] {{important item 1}}
- [ ] {{important item 2}}

**Nice to Have:**
- [ ] {{enhancement 1}}

#### 5. Recommendations
1. {{recommendation 1}}
2. {{recommendation 2}}
3. {{recommendation 3}}

## Output Format
Save to: `thoughts/shared/validation/{{YYYY-MM-DD}}-{{slug}}.md`

```markdown
# Validation Report: {{featureTitle}}

**Date:** {{date}}
**Plan:** `plans/{{YYYY-MM-DD}}-{{slug}}.md`

---

## Executive Summary

**Overall Status:** ✅ PASS / ⚠️ PASS WITH ISSUES / ❌ FAIL

**Completion:** {{n}}/{{total}} phases complete

**Critical Issues:** {{count}}
**Recommendations:** {{count}}

---

## Detailed Analysis

[Per-phase analysis as above]

---

## Test Results Summary

```bash
# Auto-validation output
npm test
{{output}}

npm run lint
{{output}}
```

---

## Next Steps

1. {{immediate action 1}}
2. {{immediate action 2}}
3. {{immediate action 3}}
```

## Validation Checklist Template

```markdown
## Pre-Release Checklist

### Code Quality
- [ ] All phases implemented per plan
- [ ] Code follows project conventions
- [ ] No TODO comments left in production code
- [ ] Linting passes with no errors

### Testing
- [ ] Unit tests pass ({{coverage}}% coverage)
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Edge cases tested

### Documentation
- [ ] Code is self-documenting
- [ ] Complex logic has comments
- [ ] API changes documented (if applicable)

### Security
- [ ] No hardcoded credentials
- [ ] Input validation in place
- [ ] Error messages don't leak info

### Performance
- [ ] No obvious performance regressions
- [ ] Database queries optimized (if applicable)
- [ ] Memory usage reasonable

### Deployment Readiness
- [ ] Environment variables documented
- [ ] Migration scripts ready (if DB changes)
- [ ] Rollback plan documented
```

## Context Management
- Validation can be context-intensive
- Check /context after each phase analysis
- Save intermediate results to file
- /clear and reload if needed

## Commands
```bash
# Run all tests
npm test

# Run specific test suite
npm test -- {{testPattern}}

# Lint check
npm run lint

# Type check
npm run type-check

# Build verification
npm run build

# Context check
/context

# Clear if needed
/clear
```

## Report Categories

### ✅ PASS Criteria
- All phases implemented
- All tests passing
- No critical issues
- Manual verification complete

### ⚠️ PASS WITH ISSUES Criteria
- All phases implemented
- Minor deviations from plan
- Some non-critical issues
- Manual verification complete

### ❌ FAIL Criteria
- Missing critical functionality
- Tests failing
- Security concerns
- Incomplete implementation
