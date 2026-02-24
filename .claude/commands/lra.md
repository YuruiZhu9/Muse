# Long-Running Agent 模式

**命令:** `/lra` 或 `/long-running`

## 适用场景

当项目具有以下特征时使用：
- 大型项目（50+ 功能点）
- 功能清晰可测试
- 需要多次会话完成
- 需要跨会话保持开发进度

## 初始化阶段

**首次使用时运行:**

```
初始化 Long-Running Agent 模式
```

Claude 将会：
1. 创建 `feature_list.json` - 功能清单模板
2. 创建 `claude-progress.txt` - 进度记录文件
3. 创建 `app_spec.txt` - 应用规格说明
4. 创建 `init.sh` - 环境初始化脚本

## 开发循环

**每次会话运行:**

```
继续开发下一个功能
```

Claude 将会：
1. **读取状态** - 读取 feature_list.json 和 claude-progress.txt
2. **选择任务** - 选择下一个 `passes: false` 的功能
3. **运行测试** - 确保现有代码正常工作
4. **实现功能** - 编写代码实现功能
5. **验证测试** - 按照功能清单中的步骤测试
6. **更新清单** - 设置 `passes: true`
7. **提交检查点** - Git commit 描述变更
8. **更新进度** - 写入 claude-progress.txt
9. **干净结束** - 确保代码可合并

## 文件结构

```
项目根目录/
├── feature_list.json      # 功能清单
├── claude-progress.txt    # 进度记录
├── app_spec.txt          # 应用规格
└── init.sh               # 环境初始化
```

## 功能清单格式

```json
[
  {
    "category": "functional",
    "description": "用户可以登录系统",
    "steps": ["打开登录页面", "输入凭证", "点击登录", "验证跳转"],
    "passes": false
  }
]
```

**分类选项:**
- `functional` - 功能性需求
- `data` - 数据处理
- `ui` - 用户界面
- `security` - 安全相关
- `performance` - 性能优化
- `testing` - 测试相关

## Commit 规范

每个功能完成后创建描述性 commit：

```
feat: 实现用户登录功能

- 添加登录表单组件
- 集成认证 API
- 添加错误处理
- 更新 feature_list.json

测试通过: 所有登录相关测试用例
```

## 关键原则

1. **一次一功能** - 每次会话只实现一个功能
2. **测试优先** - 先运行现有测试确保环境正常
3. **干净提交** - 每个 commit 必须是可工作的状态
4. **详细记录** - 更新 claude-progress.txt 保持上下文

## 与传统模式的区别

| 传统模式 | LRA 模式 |
|----------|----------|
| 单次会话完成 | 多次会话渐进 |
| 脑中记忆进度 | 文件系统记录 |
| 灵活探索 | 结构化清单 |
| 适合中小任务 | 适合大型项目 |

---

**参考:** [Long-Running Agent 分析](.claude/thoughts/shared/research/long-running-agent-analysis.md)
