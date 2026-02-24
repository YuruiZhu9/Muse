# Direct Response Mode

**For simple tasks that don't need full workflow**

## When to Use This Command
Use `/direct` or simple requests when:
- Answering a question
- Explaining code
- Single-line fix
- File lookup
- Documentation
- Quick debugging

## What This Does
- Skips all workflow prompts
- Direct answer/response
- No file generation to thoughts/
- No phase planning

## Examples
```
# Direct mode (implicit)
"What does this function do?"
"How do I fix this error?"
"Find files containing 'oauth'"

# Explicit direct mode
/direct "Summarize this file"
```

---

**Note:** For complex tasks, Claude will still ask if you want workflow mode.
