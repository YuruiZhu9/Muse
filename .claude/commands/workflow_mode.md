# Full Workflow Mode

**For complex development tasks**

## When to Use This Command
Use `/workflow` when:
- Multi-file changes
- New features
- Architecture work
- Complex refactoring
- Unknown codebase areas

**对于超大型项目（100+ 功能点），考虑使用 `/lra` 模式**

## Process
1. **Research** - Understand codebase
2. **Plan** - Create implementation plan
3. **Implement** - Execute phase by phase
4. **Validate** - Verify completion

## What This Does
- Creates research docs in `thoughts/shared/research/`
- Creates plan docs in `thoughts/shared/plans/`
- Tracks progress with checkpoints
- Manages context at each step

## Example
```
/workflow "Add OAuth2 login with Google"
```

## 模式对比

| /workflow | /lra |
|-----------|------|
| 单次会话完成 | 多次会话渐进 |
| 灵活探索 | 结构化功能清单 |
| 适合复杂任务 | 适合大型项目 |
| Research → Plan → Implement | Feature by Feature |

---

**Note:** Claude will detect complexity and suggest this mode automatically.
You can also force it with /workflow command.

**相关:** [Long-Running Agent 模式](lra.md) | [CLAUDE.md](../CLAUDE.md)
