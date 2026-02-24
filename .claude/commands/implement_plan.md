# Implement Plan Phase

## Context Check
{{#cursor}}Check /context before starting{{/cursor}}
{{#cursor}}/context
{{/cursor}}

⚠️ **Personalized:** Will prompt at 60-80% usage before clearing

## Objective
Execute a single phase of the implementation plan.

## Inputs
- Plan document: `thoughts/shared/plans/{{YYYY-MM-DD}}-{{slug}}.md`
- Phase number: {{phaseNumber}}
- Current progress: (tracked in plan doc)

## Critical Rules
1. **ONE PHASE AT A TIME** - Never implement multiple phases together
2. **Verify before proceeding** - Complete validation of Phase N before Phase N+1
3. **Document progress** - Update plan doc with completion status

## Process

### Step 1: Load Plan
Read the plan document and locate Phase {{phaseNumber}}.

### Step 2: Confirm Scope
Echo the changes for this phase:
```
Phase {{phaseNumber}}: {{phaseName}}
Files: {{count}} files to modify
Changes: {{summary}}
```

**WAIT for confirmation** before proceeding.

### Step 3: Execute Changes
For each file in the phase:

1. Read the file
2. Locate the exact lines
3. Apply changes as specified
4. Show diff for verification

### Step 4: Auto-Verification
Run all auto-validation commands for this phase:
```bash
{{autoValidationCommands}}
```

### Step 5: Request Manual Verification
List manual checks needed:
```markdown
## Manual Verification Required for Phase {{phaseNumber}}

Please verify:
- [ ] {{manualCheck1}}
- [ ] {{manualCheck2}}
- [ ] {{manualCheck3}}

Respond "pass" or "fail" with details.
```

### Step 6: Handle Results
**If passed:**
1. Update plan doc: mark Phase {{phaseNumber}} as ✅ COMPLETE
2. Check /context
3. If <60%, continue to next phase
4. If >60%, /clear then reload plan for next phase

**If failed:**
1. Document the failure
2. Propose fixes
3. Apply fixes
4. Re-run verification

## Progress Tracking Format
Update plan document with:
```markdown
## Progress Tracker

| Phase | Status | Completed Date | Notes |
|-------|--------|----------------|-------|
| Phase 1 | ✅ COMPLETE | {{date}} | All tests passing |
| Phase 2 | 🔄 IN PROGRESS | {{date}} | Waiting for manual verification |
| Phase 3 | ⏳ PENDING | - | - |
| Phase 4 | ⏳ PENDING | - | - |
```

## Error Recovery
If context exceeds limit mid-phase:
1. **SAVE current state** to plan doc
2. **Execute /clear**
3. **Reload plan** - progress is in the file, not context
4. **Continue** from where you left off

## Output Examples

### Success Response
```markdown
✅ Phase {{phaseNumber}} Complete!

**Changes Applied:**
- Modified src/auth/oauth.ts:120-145
- Added tests for token refresh

**Auto-Verification:**
✅ npm test -- auth.test.ts passed
✅ npm run lint passed

**Manual Verification Needed:**
- [ ] Test token refresh in dev environment
- [ ] Verify token expiry handling

Please test and report back. Use "pass" or "fail" with details.
```

### Failure Response
```markdown
❌ Phase {{phaseNumber}} Failed

**Issue:** {{description}}

**Proposed Fix:** {{fix}}

Apply fix? (yes/no)
```

## Context Management Commands
```bash
# Check context before starting
/context

# Check context after each file change
/context

# If approaching 60%, pause and
# 1. Save progress to plan doc
# 2. /clear
# 3. Reload plan doc
# 4. Continue

# Clear and reload
/clear
cat thoughts/shared/plans/{{date}}-{{slug}}.md
```

## Reminders for User
- Monitor context usage regularly
- Don't skip manual verification
- One phase at a time - no exceptions
- Report failures with details for better fixes
