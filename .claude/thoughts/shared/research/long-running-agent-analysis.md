# Long-Running Agent 架构分析

**来源:** Anthropic Engineering - "Effective harnesses for long-running agents"
**分析日期:** 2025-02-24
**目的:** 拆解并理解如何使用 Claude Code 进行长期项目的渐进式开发

---

## 一、核心问题

为什么需要特殊的架构模式？

**问题:** LLM 会话有上下文限制（200K tokens），大型项目无法在单次会话中完成。

**解决方案:** 双 Agent 模式 + 文件系统状态持久化

---

## 二、双 Agent 架构

### 2.1 Initializer Agent（初始化 Agent）

**职责:** 首次运行时设置开发环境

**任务清单:**
1. 创建 `feature_list.json` - 结构化功能清单
2. 创建 `init.sh` - 开发服务器启动脚本
3. 初始化 Git 仓库
4. 创建 `claude-progress.txt` - 进度追踪文件
5. 创建 `app_spec.txt` - 应用规格说明

**一次性运行:** 项目开始时执行一次

### 2.2 Coding Agent（编码 Agent）

**职责:** 每次会话实现一个功能

**工作循环:**
1. 读取当前状态（feature_list.json, claude-progress.txt）
2. 选择下一个未完成的功能
3. 实现该功能
4. 测试验证
5. 更新 feature_list.json（标记 passes = true）
6. Git commit 作为检查点
7. 更新 claude-progress.txt
8. 结束会话

**每次会话:** 实现一个功能，保持代码可合并状态

---

## 三、核心概念详解

### 3.1 Feature List（功能清单）

**文件:** `feature_list.json`

**结构:**
```json
[
  {
    "category": "functional",
    "description": "新建聊天按钮创建新对话",
    "steps": [
      "导航到主界面",
      "点击'新建聊天'按钮"
    ],
    "passes": false
  }
]
```

**字段说明:**
- `category`: 功能分类（functional, security, performance, ui）
- `description`: 功能描述
- `steps`: 测试步骤
- `passes`: 是否通过测试

**作用:**
- 替代传统的 issue tracking
- 可被 LLM 直接理解和执行
- 作为测试清单

### 3.2 Fresh Context Pattern（全新上下文模式）

**原理:** 每次会话从空白上下文开始，从文件系统恢复状态

**优势:**
- 避免 200K token 限制
- 每次会话聚焦单一目标
- 失败时可回滚到任意 commit

**状态恢复流程:**
```
新会话开始
    ↓
读取 feature_list.json → 找到下一个 passes=false 的功能
    ↓
读取 claude-progress.txt → 了解上下文
    ↓
git log → 查看最近变更
    ↓
开始实现
```

### 3.3 Git as Checkpoint System（Git 作为检查点系统）

**Commit 风格:**
```
feat: 实现用户登录功能

- 添加登录表单组件
- 集成认证 API
- 添加错误处理
- 更新 feature_list.json

测试通过: 所有登录相关测试用例
```

**原则:**
- 每个功能一个 commit
- Commit message 必须描述变更
- 代码必须可运行、可测试
- 失败时可 git reset 回退

### 3.4 Browser Automation Testing（浏览器自动化测试）

**工具:** Puppeteer MCP 或类似工具

**用途:** 端到端验证功能

**测试流程:**
1. 启动开发服务器（init.sh）
2. 运行自动化测试
3. 验证 feature steps 中的每个步骤
4. 确认无控制台错误
5. 更新 passes 字段

### 3.5 Clean State Rule（干净状态规则）

**要求:** 每次会话结束时，代码必须满足：
- 所有测试通过
- 无编译错误
- 无 TODO 悬空
- 可立即合并到主分支

**原因:** 下次会话是全新上下文，无法理解未完成的工作

---

## 四、安全模型

### 4.1 文件系统限制

**只读访问:**
- 系统配置
- 其他用户目录

**可写访问:**
- 项目目录
- 指定的临时目录

### 4.2 Bash 命令白名单

**允许的命令:**
```bash
# 基础操作
pwd, ls, cat, cd

# Git 操作
git status, git log, git diff, git add, git commit, git push

# 开发服务器
npm run dev, python server.py, ./init.sh

# 测试
pytest, npm test
```

**禁止的命令:**
```bash
# 系统破坏
rm -rf /, sudo, dd

# 数据泄露
curl http://internal-server, scp sensitive-data
```

---

## 五、实际案例参考

### 5.1 autonomous-coding 项目统计

**来源:** anthropics/claude-quickstarts/autonomous-coding

**数据:**
- 总功能数: 200+
- 每次迭代时间: 5-15 分钟
- 总开发时间: 数周（由 AI 自主完成）

**关键文件:**
```
autonomous-coding/
├── app_spec.txt           # 应用规格
├── feature_list.json      # 功能清单
├── claude-progress.txt    # 进度记录
├── init.sh                # 环境初始化
└── prompts/
    ├── initializer_prompt.md    # 初始化 Agent 提示
    └── coding_prompt.md         # 编码 Agent 提示
```

### 5.2 Coding Agent 10 步工作流

1. **定位自己** - pwd, ls, cat 关键文件
2. **选择任务** - 从 feature_list.json 选择下一个功能
3. **验证测试** - 运行已通过的测试确认环境正常
4. **分析需求** - 理解功能规格
5. **编写代码** - 实现功能
6. **浏览器验证** - 使用 Puppeteer MCP 测试
7. **更新清单** - 修改 feature_list.json 的 passes 字段
8. **提交进度** - Git commit
9. **更新笔记** - 写入 claude-progress.txt
10. **干净结束** - 确认可合并状态

---

## 六、与传统开发的对比

| 维度 | 传统开发 | Long-Running Agent |
|------|----------|-------------------|
| 上下文 | 开发者脑中记忆 | 文件系统状态 |
| 任务管理 | JIRA/Issues | feature_list.json |
| 进度追踪 | 会议/文档 | claude-progress.txt |
| 检查点 | 手动 commit | 自动功能 commit |
| 测试 | 独立测试阶段 | 每功能必测 |
| 失败恢复 | 难以回滚 | git reset |

---

## 七、适用场景

**适合:**
- 大型项目（100+ 功能点）
- 清晰可测试的功能规格
- 需要长时间开发的项目
- 团队中有 AI 编程助手

**不适合:**
- 简单脚本或工具
- 探索性/研究性项目
- 需要频繁上下文切换的创意工作

---

## 八、关键成功因素

1. **结构化功能清单** - feature_list.json 必须详细准确
2. **可靠的测试** - 每个功能必须有可执行的测试步骤
3. **干净的提交** - 每个 commit 必须是可工作的状态
4. **渐进式开发** - 一次只做一个功能，不要贪多
5. **环境一致性** - init.sh 确保每次环境相同

---

## 九、下一步行动

基于此分析，需要：
1. 创建 `claude.md` - 整合到现有工作流
2. 创建模板文件 - feature_list.json, claude-progress.txt, init.sh
3. 更新现有命令文件 - 融入 long-running agent 模式
4. 保持与现有 4-phase workflow 的兼容性

---

**参考链接:**
- 原文: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- GitHub: https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding
