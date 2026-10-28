import json
from pathlib import Path
r=Path(__file__).resolve().parents[1];raw=r/'原始证据/prefill_pr_v1'
q=json.loads((raw/'offline_gsm8k_9b/COMPLETE.json').read_text())
(r/'交付说明.md').write_text(f'''# 9B PR 交付说明

已完成一个可供人工审阅、推送的实际代码改动：**SM89 FP8 大矩阵 prefill 流水线优化**。主要证据来自 Qwen3.5-9B，不再使用旧 0.8B 结果作为核心卖点。GitHub PR 尚未创建，按你的决定，由你运行附带脚本推送。

## 结果

- 8K 输入的四组测试：每轮平均首 token 延迟的中位数降低 **4.51%–7.14%**。
- 8K 输入 / 128 输出 / 并发 17：吞吐 **+3.85%**；同一卡前后分别 **+4.09%、+3.53%**。
- 短输入解码：吞吐 **+0.07%**，视为持平；8K 输入 / 1024 输出 / 并发 4：**+0.64%**，收益很小。
- 总计 **76 次正式测量、1,520 个成功请求**；两卡交换，四次服务启动，每个版本两次启动。六次重复并非六次独立启动，没有把观测范围当置信区间。
- 基线和修改版均 **265 passed、31 skipped**；56 组额外完整输出哈希一致，各自 CUDA Graph 与直接调用一致。
- 完整 GSM8K **1,319 题**：基线 **{q['baseline_correct']}** 题正确，修改版 **{q['candidate_correct']}** 题正确；逐 token 完全一致 **{q['exact_tokens']}/1319**；长度截断分别 **{q['baseline_truncated']}、{q['candidate_truncated']}**。质量评估固定离线调度，不能拿其运行时间当性能对比。

这是双方都使用 FP8 的公平比较，**不是 FP8 对 BF16**。验证配置为 9B 文本模式、TP=1、FA2、MRV2、关闭 prefix cache、无 MTP，每个服务用一张 4090。没有宣称 BF16、TP=2、图像输入或所有 Ada 卡都获益。

## 为什么最终选这个方向

最初“先找真实 workload，再看 trace”的思路正确；问题在于容易把小模型、独立 microbenchmark 或某个并发点的漂亮数字当成实际价值。旧的小 batch 调参对 9B 没有稳定端到端收益，因此已降级归档。

重新分析 9B 的 prefill/decode 后，大 FP8 GEMM 是值得处理的成本。最终保留原有 tile、数学指令和 epilogue，只将一部分大矩阵流水线从五级改为三级，共享内存从 80KiB 降至 48KiB。改动很小，却有原生 vLLM 整模复测支撑。这比为了做 fusion 而硬找几个小算子更容易向 reviewer 说明投入产出。

独立扩展曾显示更大的收益，但与正式编译库存在代码生成差异，不能作为 PR 数字；现在的数字全部来自真正修改并编译后的原生库。kernel fusion 仍可作为后续方向，当前不把未经验证的 fusion 拼进这个 PR。

## 代码与证据

- 源码提交：`115a32c53a6a66c8d8e407c3f8858b2290a3b071`。
- 基线：`88d3995ea18de47ba5c306789aa615474ad74592`。
- 仅两个文件：生产代码新增 10 行、删除 3 行；复用既有测试增加四组 shape。全部合计新增 20 行、删除 4 行。
- `vllm-sm89-fp8-prefill.patch`：可直接阅读的补丁。
- `vllm-sm89-fp8-prefill.bundle`：含源码分支和独立证据分支；基线作为 prerequisite，需要 fetch 后导入。
- `源码/`：最终两个文件；`PR_BODY.md`：英文 PR 正文；`REVIEW.md`：四部分审阅记录。
- `原始证据/`：全部最终测量、pytest、模型评估、trace 和构建记录；`图表/`：可直接打开的 PNG/SVG。
- `复现脚本/`：实际实验脚本及等价 CLI 配方；路径依赖和未单独执行的示例已明确说明。
- `public_evidence/`：脱敏后可随证据分支公开的内容，保留所有计时向量；脱敏前后哈希可追溯。
- `SHA256SUMS.txt`：交付文件校验；`bundle_validation.log`：bundle 校验记录。

失败探索、旧候选与负结果仍保存在相邻的 `9B主线续研_20261028`、`完整PR_20261028` 等目录。**请以本目录为最终交付，不要推旧 small-batch 分支。**

## 你如何推送

在 Windows PowerShell 中执行：

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh api user --jq .login
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\\vllm_pr\\9B_PR_20261028\\发布到个人Fork.ps1"
```

确认登录输出为 `npm-DreaMaX`。如已正确登录可跳过第一条。脚本要求 Git 和 GitHub CLI 已安装；它会验证账号、bundle 和提交号，推送到你的 `npm-DreaMaX/vllm`，创建**个人 Fork 的 Draft PR**。不会 force push，不改 main，也不会自动向官方仓库提交。证据链接在证据分支推送成功后生效。

这避免账号连接切换；脚本已做 PowerShell 语法解析，但没有替你执行 GitHub 写入操作。后续如果决定向 `vllm-project/vllm` 提交，需要重新核对 upstream 和重复 PR，按官方要求完成你自己的审阅与测试。

## reviewer 最可能追问什么

1. 为什么阈值是 M>=1024、N>=4096、K>=2048：提供 guard 边界正确性、原生大矩阵数据和未改动控制；它不是只为模型名称加特判，但也没有穷尽所有 shape。
2. L4/L40S 是否获益：目前没有这些卡的数据，不能保证；流水线减少可能使某些未测场景的延迟隐藏变差。
3. 是不是只优化 microbenchmark：有真正 9B HTTP 服务、两卡交换和原生库 SHA 审计；所有其他 native DSO 相同。
4. 是否改变质量：给出完整 GSM8K 与逐 token 对照，以及现有算子测试；不会将有限样本推广成所有输入严格等价。
5. CI 能否覆盖：现有 B200 测试会运行参数化用例，但无法覆盖 SM89 分支，已明确列出本地 4090 验证与覆盖缺口。

这是有可复核收益、范围明确的 PR 候选，不能保证 reviewer 接收。官方 `AGENTS.md` 要求提交人逐行审阅、理解改动并亲自运行相关测试；这些人工动作不能由“自动验证已通过”代替，来源和具体要求见 `REVIEW.md`。
''')
