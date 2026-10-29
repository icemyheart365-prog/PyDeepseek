# DeepSeek 本地聊天项目 · 交接文档

> 用途：切换到 DeepSeek Harness 后快速恢复上下文。项目本身已完整提交，本文档只记录**为什么这样做**和**下一步做什么**。

---

## 1. 项目是什么

一个**本地运行、直连 DeepSeek API** 的对话程序，替代 VS Code 插件。核心特征：

- **纯标准库**（除可选的 `tokenizers`），无框架依赖
- **单入口** `chat_v10.py`，运行时 `/settings` 切换语言和模型
- **模型只负责输出 JSON 协议**，所有副作用由本地 Python 调度器决定
- 目标环境曾经包括 A-Shell（iOS），所以极度克制依赖

```
DeepSeek/
├── chat_v10.py              唯一入口（340 B）
├── agent_core.py            全部逻辑（1.3k 行）
├── agent_dispatcher.py      工具调用校验与执行
├── command_library_bridge.py JSON/Base64 信封，只做包装
├── translator.py            i18n 加载器
├── web_tools.py             DeepSeek Search/Fetch
├── bridges/                 python（默认）/ powershell / posix
├── tokenizer/               token 计数 + 6 MB 词表
├── i18n/                    zh_cn.json / en_ca.json（140 key）
├── memories/                commands.json（契约）+ commands.impl.json（实现细节）
└── library/                 文件附件区
```

---

## 2. 为什么是这个架构

### 2.1 协议隔离（最重要的设计）

模型**永远不执行任何东西**。它只能输出两种 JSON：

```json
{"type": "output", "text": "..."}
{"type": "tool_call", "command": "ls", "arguments": {"path": "."}, "reason": "..."}
```

`agent_dispatcher.py` 拿到 `tool_call` 后：
1. 用 `memories/commands.json` 的 schema **严格校验参数**（未知参数直接拒绝）
2. 强制确认门（`require_confirmation`）与输出传输门（大输出需用户批准）
3. 通过 `command_library_bridge.py` 调后端，**永远不拼接 shell 字符串**

> 这条必须保持。任何"让模型直接跑命令"的改动都会破坏整个安全模型。

### 2.2 commands.json 拆分（省 87% prompt token）

| 文件 | 内容 | 谁读 |
|---|---|---|
| `commands.json` | name / description / enabled / require_confirmation / parameters | prompt + 调度器校验 |
| `commands.impl.json` | shell_mapping / limits / notes / implementation | **无人读**，纯归档 |

`build_prompt_view()` 只发 `name(params) - description`，实测 **4606 → 570 token**。

两份文件的命令名集合**必须一致**，`_check_impl_consistency()` 在加载时硬失败。

### 2.3 教学式报错（支撑精简 schema）

因为 prompt 里不再有参数详情，校验失败时必须教会模型怎么改：

```
参数无效：bogus 不在该命令的定义中。
  - (path; type=string; default=".")  Relative path inside library.
  - (format; type=string; allowed=name,long,human,long_human; default="name")  ...
```

**只改报错文案，校验逻辑一行没放松。**

### 2.4 i18n 只翻译 UI，不翻译工具描述

命令描述刻意保持英文——它是工具协议语言。`i18n/` 只管界面文案 + 发给模型的提示词。

`check_parity()` 在启动时强制两份 JSON 的 key 集合一致。

---

## 3. 踩过的坑（避免重犯）

| 坑 | 症状 | 解法 |
|---|---|---|
| `ast` 的 `col_offset` 是 **UTF-8 字节偏移** | 中文文案替换错位 | 按字节建偏移表 |
| `ast.walk()` **遍历顺序不保证** | 两次运行配对不一致 | 按 `(lineno, col_offset)` 强排序 |
| `tokenizers` 包**没有** `TokenizerError` | import 静默失败，永远走回退 | 用 `except Exception` |
| f-string 表达式当 kwarg 名 | `t("k", path.stem=path.stem)` 语法错 | 非标识符改用 `arg0/arg1` |
| 模板含字面量 `{"limit":10}` | `.format()` 报 KeyError | 转义成 `{{"limit":10}}` |
| PowerShell here-string 破坏中文 | 文案变 `???` | 用 `\uXXXX` 转义写入 |
| **junction 逃逸** | `cat _junc/win.ini` 读到 `C:\Windows` | 逐级向上遍历路径分量 |
| CRLF 混入 | `ast.parse` 报 BOM | `.gitattributes` 锁定 |

---

## 4. 当前状态

```
远端   https://github.com/icemyheart365-prog/PyDeepseek  (PUBLIC)
分支   main（本地与 origin/main 同步）
标签   1.0.0 → draft release 已生成
```

**已验证**：中英文启动 / 全部斜杠命令 / 两个后端字节级一致 / 14 项攻击向量全拦截 / CI release 打包

### 4.1 敏感文件与 git 的关系（重要）

| 文件 | 是否跟踪 | 说明 |
|---|---|---|
| `config.json` | ❌ 忽略 | 本地真实 key 所在，`.gitignore` 第 2 行 |
| `config.example.json` | ✅ **跟踪** | 仓库内的模板，发布包靠它 |
| `HANDOFF.md` | ❌ 忽略 | 本文档，个人上下文 |
| `memories/default.json` | ❌ 忽略 | 对话历史，`memories/*.json` + 白名单例外 |
| `memories/commands.json` | ✅ 跟踪 | 命令契约 |
| `memories/commands.impl.json` | ✅ 跟踪 | 实现细节 |

**踩过的坑**：曾把模板版 `config.json` 提交又 `git rm --cached`，但 **`.gitignore` 对已跟踪文件无效**——本地改 key 后 `git status` 显示 `M config.json`，一次 `git add -A` 就会泄露。现改为 `config.example.json` + 完全忽略 `config.json`，实测改 key 后 `git status` 不显示它。

### 4.2 历史清理记录

两次 `filter-branch` 重写：
1. 移除 `memories/default.json`（真实对话）
2. 移除 `HANDOFF.md`

每次都必须**同时删掉 `refs/original/` 和 `refs/agents/` （IDE checkpoint）引用**，再 `reflog expire --expire=now --all` + `gc --prune=now`，否则旧对象仍可访问。

---

## 5. 未完成

### 3C 懒加载（暂缓，倾向不做）
prompt 里只放命令名，参数细节靠 `describe <cmd>` + 教学式报错补。问题在于模型必须盲调一次才拿到 schema，失败重试的成本可能高于省下的 token。现在已降到 570 token、缓存命中 96.8%，边际收益很小。

### 潜在改进
- `KNOWN_MODELS` 仅 `deepseek-flash` / `deepseek-v4-pro`，有 8 槽上限校验（`PRESET_MODEL_SLOTS`）
- `bridges/posix` 在 Windows 无法测试（本机无 `sh`），A-Shell 支持需实机验证
- `README.md` 只有两行，可补安装步骤与许可证说明

---

## 6. 切换到 Harness 时的注意事项

**保留的资产**
1. `memories/commands.json` — 这套命令契约可以直接复用
2. `agent_dispatcher.py` 的**校验与安全逻辑** — 比任何框架默认行为都严格
3. `tokenizer/` — 精确 token 计数，`from tokenizer import token_counter`
4. 教学式报错文案 — 模型自我修正的关键

**可能要放弃的**
- `i18n/` 双语文案层（Harness 通常自带）
- `bridges/` 三后端（Harness 有自己的工具执行层）
- `chat_v10.py` 的 REPL 循环

**绝对不能放弃的**
- 参数白名单校验
- 路径 containment 检查（尤其 reparse point）
- 确认门 / 传输门
- 输出大小上限

---

## 7. 读取指引（防止无谓消耗 token）

**先读这一节，再决定读什么。** 整个仓库约 6.4 MB，其中 99% 是不该进上下文的二进制资产。

### 7.1 硬性禁止读取

| 文件 | 体积 | token 成本 | 原因 |
|---|---|---|---|
| `tokenizer/tokenizer.json` | **6.2 MB** | **约 274 万** | 分词词表，纯数据。需要计数就 `import tokenizer` |
| `tokenizer/tokenizer_config.json` | 3 KB | 约 900 | 同上，仅供 `tokenizers` 库解析 |
| `library/logs/*.jsonl` | 变动 | 每条 3k+ | 运行时审计日志，除非排查工具调用失败 |
| `memories/*.json`（非 commands） | 变动 | 每条 4k+ | 真实对话历史，与代码逻辑无关 |

> 单是读一次 `tokenizer.json`，成本就是整个仓库其余部分总和的 **200 倍以上**。

### 7.2 推荐读取顺序（按需递增）

| 顺序 | 文件 | tokens | 何时读 |
|---|---|---|---|
| 1 | `HANDOFF.md` | 3,193 | 你正在读的 |
| 2 | `memories/commands.json` | 2,490 | 改工具行为时 |
| 3 | `agent_dispatcher.py` | 4,947 | 改校验/安全边界时 |
| 4 | `agent_core.py` | 13,268 | 只在改主流程时，且必须分段读 |

`agent_core.py` 有 1,300 行，**不要整文件读**。按任务定位：

| 任务 | 大致行号 | 函数 |
|---|---|---|
| 改斜杠命令分发 | 1120–1250 | `main()` 命令分发段 |
| 改 prompt / 协议 | 620–690 | `build_prompt_view` / `build_command_context` |
| 改参数校验 | 见 `agent_dispatcher.py` | `_validate_args` |
| 改设置菜单 | 940–1070 | `_menu_entries` / `_apply_*` / `handle_settings` |
| 改 UI 文案 | **不要改 .py** | 改 `i18n/*.json` |

### 7.3 用运行代替阅读

很多问题不需要读源码，直接跑更快也更准：

```bash
python -c "from tokenizer import token_counter; print(token_counter(open('x.py').read()))"
python -c "import agent_dispatcher as d, pathlib; print(d.dispatch_tool_call({'type':'tool_call','command':'ls','arguments':{}}, pathlib.Path('.')))"
python -c "from translator import check_parity; check_parity()"
python chat_v10.py --help
git log --oneline
git show --stat HEAD
```

### 7.4 需要 token 计数时

```python
from tokenizer import token_counter, is_available
is_available()              # tokenizers 是否可用
token_counter("some text")  # 精确计数（不可用时用 approximate_token_count）
```

**不要**为了看看 tokenizer 长什么样去打开那个 JSON。

### 7.5 改代码后的检查

```bash
python -m py_compile agent_core.py agent_dispatcher.py chat_v10.py translator.py     tokenizer/__init__.py bridges/__init__.py bridges/python_backend.py     command_library_bridge.py
```

改完 i18n 后额外跑 `check_parity()`；改完 `commands.json` 后确认与 `commands.impl.json` 命令名集合一致（加载时会硬校验）。

---

## 8. 发布流程（GitHub Actions）

`.github/workflows/release.yml`：

```yaml
on:
  push:
    tags:
      - '[0-9]+.[0-9]+.[0-9]+'   # 只认纯数字 x.y.z，带 v 前缀不触发
```

**流程**：checkout → `git archive` 打包 → 校验内容 → `gh release create --draft`

**校验兜底**（CI 会硬失败）：
- 不得包含 `config.json` / `HANDOFF.md` / `memories/default.json`
- **必须**包含 `config.example.json`

**实测产物**（1.0.0）：
```
PyDeepseek-1.0.0-source.zip   1,971,267 bytes   25 个文件
```

**发版命令**：
```powershell
git tag -a 1.0.1 -m "…"
git push origin 1.0.1
# 到 Actions 看运行，然后发布草稿
gh release edit 1.0.1 --draft=false
```

**撤销 tag**（发布前反悔时）：
```powershell
git tag -d 1.0.1
git push origin :refs/tags/1.0.1
gh release delete 1.0.1 --yes
```

> `git archive` 是关键：它只打包**已跟踪**文件，所以本地 key、对话历史天然进不了发布包。

---

---

---

## 9. 接手须知（复制仓库后）

这份仓库是**完整可用**的，不是快照。直接跑：

```powershell
cd DeepSeek
# 1. 配置 API key
copy config.example.json config.json
notepad config.json          # 填入真实 key

# 2. 可选：精确 token 计数
pip install -r requirements.txt

# 3. 运行
python chat_v10.py
```

**自检命令**（不消耗 API）：

```powershell
python chat_v10.py --help
python -c "from translator import check_parity; check_parity()"
python -m py_compile agent_core.py agent_dispatcher.py chat_v10.py translator.py tokenizer/__init__.py bridges/__init__.py bridges/python_backend.py command_library_bridge.py
```

**环境要求**：Python 3.10+（用了 `X | None` 类型标注和 `sys.stdlib_module_names`）。
`tokenizes` 可选，缺失时自动退回字节估算。

**Windows 上可切换后端**：

```powershell
$env:DEEPSEEK_BRIDGE = "powershell"    # 默认 python
```

### 别做的事

- ❌ 不要 `git add -f config.json` —— 真实 key 会进公开仓库
- ❌ 不要整文件读 `tokenizer/tokenizer.json`（274 万 token）
- ❌ 不要把 `memories/default.json` 提交进去（真实对话历史）

---

## 10. 笔记同步（notes 分支）

`HANDOFF.md` 在 `main` 上被 `.gitignore` 忽略，但**在 `notes` 分支上被跟踪**。这样笔记可以跟着 git 在家/学校之间走，又不会污染 `main`。

### 10.1 为什么需要脚本

切分支时 git 会删除"目标分支没有的已跟踪文件"。直接从 `notes` 切回 `main`，`HANDOFF.md` 会**从磁盘消失**。而且：

- `git show` 经过 PowerShell 管道时按控制台代码页解码，**中文会变乱码**
- 切到 `notes` 会用分支版本**覆盖工作区的编辑**，push 前必须先备份

`notes.ps1` 封装了这三个坑。

### 10.2 用法

```powershell
.\notes.ps1 status   # 本地与 origin/notes 是否一致（默认）
.\notes.ps1 pull     # 到学校后先拉取
.\notes.ps1 push     # 改完笔记后推送
```

脚本可以留在 `main` 上（已跟踪），它会自动切分支再切回来。

### 10.3 实现要点（改脚本时别踩回去）

| 问题 | 做法 |
|---|---|
| 切分支删文件 | `Exit-Notes` 里用 `git show` 把内容写回磁盘 |
| 中文乱码 | 用 `cmd /c "git show ... > file"` 做**字节级重定向**；PowerShell 管道和 `Start-Process -RedirectStandardOutput` 都会转码 |
| push 丢编辑 | 切分支前 `Copy-Item` 备份，到分支上再还原 |
| 误判差异 | 比较 `git hash-object` 而非文本内容，免疫 CRLF 和编码差异 |
| git 警告变异常 | `Invoke-Git` 只捕获 stdout，不合并 stderr |

### 10.4 执行策略

Windows 默认禁止运行 `.ps1`。三种解法：

```powershell
# 临时（只影响本次）
powershell -ExecutionPolicy Bypass -File notes.ps1 push

# 当前用户永久（推荐）
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 11. 未来展望：首次启动向导

**目标**：`python chat_v10.py` 第一次运行时，引导用户完成三项配置，而不是直接报"config.json 缺少有效的 api_key"。

### 11.1 期望流程

```
没有 config.json
      ↓
[1] 选择语言（先问，因为后面全用它显示）
      ↓
[2] 输入 API Key（带获取方式提示 + 格式校验）
      ↓
[3] 选择模型（deepseek-flash / deepseek-v4-pro）
      ↓
写入 config.json，进入对话
```

**顺序很重要**：语言必须第一个问，否则后面的提示词无法本地化。

### 11.2 关键设计

| 点 | 说明 |
|---|---|
| 默认语言 | **英文**（国际惯例，且用户可能来自任何地区） |
| Key 校验 | 写入前发一次极短请求（`max_tokens=1`）验证，失败则提示重试而不是写入坏配置 |
| Key 掩码 | 输入时不回显，或显示 `sk-****` |
| 已有配置 | `config.json` 存在且有效 → 跳过向导，不打扰老用户 |
| 可跳过 | 允许稍后手动填，但要给出明确提示 |
| 幂等 | 中途 Ctrl+C 不留下半写状态（用 `atomic_save_json`，已有） |

### 11.3 与 /settings 的关系

向导和 `/settings` 复用同一套函数：

- `_apply_language` / `_apply_model` 已有
- 需要抽出 `prompt_for_api_key()`，向导和将来的 `/settings apikey` 都能用
- 向导本质是"首次检测失败时自动调用一遍 `handle_settings`"

### 11.4 影响的文件

- `agent_core.py`：`main()` 启动阶段加检测分支
- `i18n/*.json`：新增向导文案（约 15 个 key）
- `README.md`：必须写清 key 获取步骤（见 11.5）

---

## 12. 未来展望：README 补全

现在 [README.md](README.md) 只有两行，几乎没信息量。目标内容：

### 12.1 必须包含

1. **一句话简介** —— 本地运行的 DeepSeek 客户端，工具调用由本地调度器把关
2. **获取 API Key 的详细步骤**
   - 打开 <https://platform.deepseek.com/api_keys>
   - 注册 / 登录
   - 点「创建 API Key」
   - 复制 `sk-` 开头的字符串（**只显示一次**，务必当场保存）
   - 前往「充值」页面确认余额（新账号可能没有赠送额度）
3. **快速开始**（三条命令）
4. **功能概览** —— `/settings`、记忆分支、工具调用、双语界面
5. **安全说明** —— 模型不能直接执行任意命令，所有操作经白名单校验 + 用户确认
6. **许可证** —— MIT

### 12.2 加分项

- 截图或终端演示（GIF 更好）
- 常见问题：`api_key` 填了但报 401 / PowerShell 执行策略 / tokenizer 未安装
- 架构简述（一张图说明 模型 → JSON → 调度器 → 后端）
- 与官方插件的差异对比

### 12.3 注意

README 是 `main` 上的跟踪文件，**推送即公开**。写作时不要贴真实 key、个人路径或对话内容。

---

## 13. 一句话总结

这是一个**把模型当成不可信 JSON 生成器**的本地 Agent：所有能力通过白名单命令暴露，所有参数严格校验，所有副作用需用户确认。省 token 是靠**把实现细节从 prompt 里拿出去**，而不是削弱安全。

---

## 14. 聊天日志

按地点记录，便于回顾「上次做到哪」。

### CHAT:

#### School:
Time: ``

#### Home:
Time: ``
