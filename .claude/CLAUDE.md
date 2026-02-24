# MuseRecSys - Claude Code 项目指南

**项目名称:** MuseRecSys
**项目类型:** 推荐系统（Recommendation System）
**最后更新:** 2025-02-24
**语言:** Python

---

## 项目概述

MuseRecSys 是一个推荐系统项目，使用 Claude Code 作为 AI 辅助开发工具。本项目集成了 **Long-Running Agent** 架构模式，支持大型项目的渐进式开发。

---

## 架构模式：Long-Running Agent

### 核心理念

将大型项目拆解为大量小型、可测试的功能（features），每次会话只实现一个功能，通过 Git commit 作为检查点，实现跨会话的连续开发。

### 关键文件

```
项目根目录/
├── feature_list.json      # 功能清单（待实现/已完成）
├── claude-progress.txt    # 会话进度记录
├── app_spec.txt          # 应用规格说明
└── init.sh               # 开发环境初始化脚本
```

### 工作流程

```
初始化阶段（一次性）
    ↓
开发循环（每会话一次）
    读取状态 → 选择功能 → 实现测试 → 提交检查点 → 记录进度
```

---

## 开发模式选择

### 模式一：传统 4-Phase Workflow（适合中小型任务）

适用场景：
- 单次会话可完成的任务
- 需要深入探索代码库
- 架构设计和重构

**命令:** `/workflow` 或直接描述需求

**流程:**
1. **Research（研究）** - 理解代码库和需求
2. **Plan（规划）** - 制定实现计划
3. **Implement（实现）** - 编写代码
4. **Validate（验证）** - 测试和验证

### 模式二：Long-Running Agent（适合大型项目）

适用场景：
- 100+ 功能点的项目
- 需要多次会话完成
- 功能清晰可测试

**命令:** `/lra` 或 `/long-running`

**流程:**
1. **Setup（设置）** - 首次运行时初始化环境
2. **Loop（循环）** - 每会话执行一个功能
   - 读取 `feature_list.json`
   - 选择下一个 `passes: false` 的功能
   - 实现并测试
   - 更新 `passes: true`
   - Git commit
   - 更新 `claude-progress.txt`

---

## 快速开始

### 首次使用：初始化项目

```
我需要为这个项目设置 Long-Running Agent 模式
```

Claude 将会：
1. 创建 `feature_list.json` 模板
2. 创建 `claude-progress.txt`
3. 创建 `app_spec.txt`（基于现有代码）
4. 创建 `init.sh`（如需要）

### 日常开发：实现下一个功能

```
继续开发下一个功能
```

Claude 将会：
1. 读取当前状态
2. 选择下一个未完成功能
3. 实现并测试
4. 提交检查点

---

## 文件说明

### feature_list.json

功能清单，包含项目中所有待实现的功能。

```json
[
  {
    "category": "functional",
    "description": "用户可以登录系统",
    "steps": [
      "打开登录页面",
      "输入用户名和密码",
      "点击登录按钮",
      "验证跳转到首页"
    ],
    "passes": false
  },
  {
    "category": "data",
    "description": "加载用户历史数据",
    "steps": [
      "用户登录后",
      "系统自动加载历史记录",
      "数据正确显示"
    ],
    "passes": false
  }
]
```

**分类（category）:**
- `functional` - 功能性需求
- `data` - 数据处理
- `ui` - 用户界面
- `security` - 安全相关
- `performance` - 性能优化
- `testing` - 测试相关

### claude-progress.txt

会话进度记录，用于跨会话保持上下文。

```
=== MuseRecSys 开发进度 ===

最后更新: 2025-02-24

已完成功能:
- [x] 项目初始化
- [x] 数据模型设计

当前进行中:
- [ ] 用户登录功能

已知问题:
- 无

下次计划:
- 实现登录表单
- 集成认证 API
```

### app_spec.txt

应用规格说明，描述项目的整体目标和需求。

```
# MuseRecSys 应用规格

## 项目目标
构建一个基于用户行为的音乐推荐系统。

## 核心功能
1. 用户认证和授权
2. 数据收集和存储
3. 推荐算法
4. 结果展示和交互
```

### init.sh

开发环境初始化脚本，确保每次环境一致。

```bash
#!/bin/bash
# MuseRecSys 开发环境初始化

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python app.py
```

---

## Git 工作流

### Commit 规范

每个功能完成后创建一个 commit：

```
feat: 实现用户登录功能

- 添加登录表单组件
- 集成认证 API
- 添加错误处理
- 更新 feature_list.json

测试通过: 所有登录相关测试用例
```

### Commit 类型

- `feat` - 新功能
- `fix` - Bug 修复
- `refactor` - 代码重构
- `test` - 测试相关
- `docs` - 文档更新
- `perf` - 性能优化

### 恢复检查点

如果需要回退：

```bash
# 查看历史
git log --oneline

# 回退到指定 commit
git reset --hard <commit-hash>
```

---

## 测试策略

### 测试优先级

1. **自动化测试** - 单元测试、集成测试
2. **手动测试** - 按照 feature_list.json 中的 steps 执行
3. **浏览器测试** - 使用 Playwright MCP 进行端到端测试

### 测试命令

```bash
# Python 项目
pytest

# 带覆盖率
pytest --cov=src

# 特定测试
pytest tests/test_auth.py
```

---

## 安全注意事项

### 文件系统访问

- **可写:** 项目目录、临时目录
- **只读:** 系统配置、其他用户目录

### Bash 命令

**允许的命令:**
- `pwd, ls, cat, cd` - 基础操作
- `git status, log, add, commit` - Git 操作
- `pytest, python` - 开发和测试

**禁止的命令:**
- `rm -rf /` - 破坏性操作
- `sudo` - 权限提升
- `curl http://internal` - 内网访问

---

## 上下文管理

### 阈值策略

| 使用率 | 操作 |
|--------|------|
| < 60% | 继续正常工作 |
| 60-80% | 询问用户是否清空上下文 |
| > 80% | 自动清空并重新加载状态 |

### 状态恢复

清空上下文后，从文件系统恢复：

1. 读取 `feature_list.json` - 获取功能清单
2. 读取 `claude-progress.txt` - 获取进度
3. 运行 `git log` - 查看最近变更
4. 继续工作

---

## 技能和工具推荐

### 常用 Skills

- `/commit` - 创建 Git commit
- `/workflow` - 启动 4-phase workflow
- `/lra` - 启动 Long-Running Agent 模式

### MCP 服务

| 服务 | 用途 |
|------|------|
| `web_reader` | 网页内容获取 |
| `4.5v_mcp__analyze_image` | 图片分析（设计稿） |
| `@modelcontextprotocol/server-sqlite` | 数据库操作 |

---

## 项目结构

```
MuseRecSys/
├── .claude/
│   ├── CLAUDE.md              # 本文件
│   ├── user-preferences.md    # 用户偏好设置
│   ├── commands/              # 自定义命令
│   └── thoughts/              # 持久化上下文
│       ├── shared/
│       │   ├── research/      # 研究文档
│       │   ├── plans/         # 实现计划
│       │   └── notes/         # 临时笔记
│       └── local/             # 用户专属笔记
├── feature_list.json          # 功能清单
├── claude-progress.txt        # 进度记录
├── app_spec.txt              # 应用规格
├── init.sh                   # 环境初始化
├── src/                      # 源代码
├── tests/                    # 测试
└── requirements.txt          # 依赖
```

---

## 常见问题

### Q: 如何在两种模式间切换？

A: 根据任务复杂度选择：
- 简单任务 → 直接描述
- 中等任务 → `/workflow`
- 大型项目 → `/lra`

### Q: 功能清单太长怎么办？

A: 按模块拆分：
```
feature_list_auth.json   # 认证模块
feature_list_data.json   # 数据模块
feature_list_ui.json     # 界面模块
```

### Q: 如何处理需要多个会话的复杂功能？

A: 将复杂功能拆解为多个子功能：
```json
{
  "category": "functional",
  "description": "用户登录 - 创建表单组件",
  "steps": ["登录表单可见", "包含用户名和密码输入框"],
  "passes": false
}
```

---

## 参考资源

- [Long-Running Agent 分析](.claude/thoughts/shared/research/long-running-agent-analysis.md)
- [Anthropic Engineering 文章](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Claude Quickstarts](https://github.com/anthropics/claude-quickstarts)

---

**记住:** 保持每次会话结束时代码处于可合并状态，这样下个会话可以从干净的状态开始。
