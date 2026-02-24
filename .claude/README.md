# .claude Directory Configuration

**Project:** MuseRecSys
**User:** 53025
**Last Modified:** 2025-02-03

---

## 📁 Directory Structure

```
.claude/
├── user-preferences.md      # ⭐ User customization - highest priority
├── CLAUDE.md                 # Project context (optional)
├── commands/
│   ├── direct_mode.md        # Quick response mode
│   ├── workflow_mode.md      # Full 4-phase workflow
│   ├── research_codebase.md  # Phase 1: Research
│   ├── create_plan.md        # Phase 2: Plan
│   ├── implement_plan.md     # Phase 3: Implement
│   └── validate_plan.md      # Phase 4: Validate
├── agents/                   # Custom agents (optional)
└── thoughts/                  # Persistent context storage
    ├── shared/
    │   ├── research/         # Research documents
    │   ├── plans/            # Implementation plans
    │   ├── validation/       # Validation reports
    │   └── notes/            # Temporary notes
    └── local/                # User-specific notes (gitignored)
```

---

## 🎯 Quick Start

### For Simple Tasks
```
Just ask directly - no special commands needed.
Examples:
- "What does this function do?"
- "Find files with 'oauth' in the name"
- "Explain this error"
```

### For Complex Tasks
```
Use /workflow or just describe what you need:
Examples:
- /workflow "Add OAuth2 login with Google"
- "I need to implement a new recommendation algorithm"
- "Refactor the user authentication system"
```

---

## ⚙️ Configuration Priority

1. **Your current instruction** (highest)
2. `.claude/user-preferences.md`
3. Individual command files
4. Default workflow behavior

---

## 🔧 Common Commands

| Command | Purpose |
|---------|---------|
| `/direct` | Force direct response mode |
| `/workflow` | Force full 4-phase workflow |
| `/lra` | Long-Running Agent 模式（大型项目） |
| `/context` | Check current context usage |
| `/clear` | Clear context (will prompt at 60-80%) |
| `/status` | Show current configuration |

---

## 📋 User Preferences Summary

### Context Management
- **60-80%:** ⚠️ Ask user before clearing
- **>80%:** 🔄 Auto-clear immediately
- **<60%:** Continue normally

### Git Integration
- Auto-detect if in git repo
- Skip git operations if not in repo
- No git-related prompts for non-git projects

### Workflow Flexibility
- Simple tasks → Direct response
- Complex tasks → Ask: workflow or direct?
- User can override anytime

### File Editing
- All `.claude/` files are editable
- Changes take immediate effect
- Conflicts → Ask before updating

---

## 🔄 Updating Configuration

To change your preferences:

1. Edit `.claude/user-preferences.md`
2. Or edit individual command files
3. Changes apply immediately

To reset to defaults:
```
/reset - Clear all preferences
```

---

**Tip:** This file is a quick reference. See `.claude/user-preferences.md` for full details.
