# 复现指南（中文）

本文档逐步说明如何从公开数据复现论文 *Physics-Grounded Multimodal Large Language Models for Pretreatment pCR Prediction from Breast DCE-MRI* 的全部结果。英文说明见仓库根目录的 `README.md`。

## 最快的复现方式（不用下载影像，不用 API）

仓库的 `data/` 文件夹已经包含论文用到的全部派生数据：强化曲线、B1 的输入、预测和 SHAP 值，以及 19,640 条大模型回复。安装依赖后运行一条命令，就能重新生成论文中的全部表格、图和数字：

```bash
pip install -r requirements.txt
OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate      # 约 10 分钟
```

Windows（cmd）：

```bat
cmd /c "set OUT_DIR=%CD%\data&& set PAPER_DIR=%CD%\paper&& scripts\run_study.bat evaluate"
```

结果写在 `paper\` 里。`data/README.md` 说明了每个文件是什么、从哪一步产生。下面第 0–6 节是从原始影像开始的完整复现。

## 0. 你需要准备什么

| 项目 | 说明 |
|---|---|
| 电脑 | Windows / Linux / macOS 均可；**不需要 GPU**；内存 ≥ 16 GB 更好（第 01 步按每进程约 2 GB 自动选择并行数） |
| 硬盘 | 影像约 54 GB，结果约 2 GB |
| Python | 3.10 及以上 |
| API（第 5 步才需要） | 智谱开放平台账号、硅基流动账号，各自余额约 30 元（2026 年 10 月价格） |

## 1. 安装

```bash
git clone https://github.com/<org>/GroundedPCR-MLLM.git
cd GroundedPCR-MLLM
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

**先做离线自检**（不需要数据、不需要 API、不花钱，几分钟）：

```bash
bash tests/test_mllm_pipeline.sh
```

最后出现 `MLLM PIPELINE TEST PASSED` 说明环境没问题。Windows 上可在 Git Bash 里运行这条命令。

## 2. 下载数据

1. **BreastDCEDL 元数据**：克隆 <https://github.com/naomifridman/BreastDCEDL>，需要其中的 `BreastDCEDL_metadata.csv` 和 `ISPY2/BreastDCEDL_spy2_tcia_metadata.csv`。
2. **I-SPY2 影像包**：按 BreastDCEDL README 中的链接从 TCIA 下载 `BreastDCEDL_ISPY2`（约 54 GB，用 IBM Aspera 或 TCIA Data Retriever）。只需要这一个子集，I-SPY1 和 Duke 不需要。

下载后目录结构应为：

```
BreastDCEDL_ISPY2/
  ACRIN-6698-102212/
    dce/ACRIN-6698-102212_spy2_vis1_dce_aqc_0.nii.gz ... _aqc_7.nii.gz
    mask/ACRIN-6698-102212_spy2_vis1_mask.nii.gz
  ...
```

## 3. 设置路径

**Windows**：用记事本打开 `scripts\run_study.bat`，修改开头这几行：

```bat
if "%DATA_DIR%"=="" set DATA_DIR=D:\data\BreastDCEDL_ISPY2
if "%BREASTDCEDL_REPO%"=="" set BREASTDCEDL_REPO=D:\data\BreastDCEDL
```

结果默认保存在仓库下的 `results\`。

**Linux / macOS**：

```bash
export DATA_DIR="/data/BreastDCEDL_ISPY2"
export BREASTDCEDL_REPO=/data/BreastDCEDL
export OUT_DIR=results
export PAPER_DIR=results/paper
```

## 4. 提取特征、训练参考模型 B1（不需要 API）

| 命令（Windows） | 命令（Linux/macOS） | 内容 | 时间 |
|---|---|---|---|
| `cmd /c scripts\run_study.bat features` | `bash scripts/run_study.sh features` | 整理元数据，划分核心区/边缘区，提取强化曲线和异质性特征 | 1–2 小时 |
| `cmd /c scripts\run_study.bat b1` | `bash scripts/run_study.sh b1` | 7 种特征组合的 5 折交叉验证，选出 B1，测试集只评估一次，计算 SHAP，并跑敏感性分析 | 约 1 小时 |
| `cmd /c scripts\run_study.bat export` | `bash scripts/run_study.sh export` | 生成大模型的输入：`cases.jsonl` 和每位患者的 2×2 影像拼图 | 20–30 分钟 |

每一步都会跳过已经完成的部分，中断后重新运行同一条命令即可。

## 5. 调用大模型（收费）

**第 1 步：设置 API key**（只放在环境变量里，不要写进任何文件）。

PyCharm 终端默认是 PowerShell，必须用 `$env:` 的写法：

```powershell
$env:ZHIPU_API_KEY="你的智谱key"            # 完整 key 中间带一个“.”
$env:SILICONFLOW_API_KEY="你的硅基流动key"
```

注意：PowerShell 里 `set ZHIPU_API_KEY=...` **不会生效**；这样设置的 key 只在当前窗口有效。

**第 2 步：测试连通性**，两条都应显示 `HTTP 200`：

```bash
python scripts/check_api.py --provider zhipu --model glm-4.5v --disable_thinking
python scripts/check_api.py --provider siliconflow --model Qwen/Qwen3-VL-32B-Instruct
```

如果硅基流动返回 `Model disabled`，先用 `python scripts/check_api.py --provider siliconflow --list_models` 查看可用的视觉模型，再设置 `$env:SF_MODEL="模型名"`。

**第 3 步：小规模试跑**（约 20 个病例，几分钟，花费不到 1 元）：

```
cmd /c scripts\run_study.bat pilot        （Linux：bash scripts/run_study.sh pilot）
```

打开 `results\mllm\runs\*\*.csv`，确认 `parse_ok` 都是 1。

**第 4 步：正式运行**：

```
cmd /c scripts\run_study.bat query
```

- 共 19,640 次请求：982 例 × 5 种配置 × 2 次重复 × 2 个平台。
- 每个平台约 1,400 万 tokens，按 2026 年 10 月价格各约 20–30 元。
- 智谱约 3–4 小时，硅基流动约 8–10 小时，建议放一整夜，并关闭电脑睡眠。
- 中途断了直接重新运行，已完成的回复从缓存 `results\mllm\cache\` 读取，**不会重复扣费**。
- 出现大量 429（请求过于频繁）时，把并发数调低：`$env:WORKERS="4"`。

## 6. 生成论文的表格和图（离线，约 10 分钟）

```
cmd /c scripts\run_study.bat evaluate
```

输出在 `results\paper\`：

- `results_macros.tex`：正文中引用的全部数字；
- `tables\*.tex`：表 I–IX 的内容；
- `figures\*.pdf`：图 1、4、5、6；
- `analysis\*.csv`：各项副分析背后的原始数据。

把论文的 `main_mllm.tex` 放进这个文件夹（或把这些文件复制到论文文件夹），依次运行 `pdflatex → bibtex → pdflatex → pdflatex` 即可编译。

## 7. 各种复现起点

| 想做什么 | 需要什么 | 怎么做 |
|---|---|---|
| 重新生成论文的表格、图和数字 | 只要本仓库 | `OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate` |
| 用仓库里的曲线重新训练 B1 | 只要本仓库 | 把 `data/metadata` 和 `data/ispy2_roi` 复制到新的结果文件夹，再运行 `b1` 阶段 |
| 用同样的提示词测试别的大模型 | `data/mllm/cases.jsonl` + GitHub Release 中的影像拼图压缩包 + API key | 运行 `scripts/13_run_mllm.py` |
| 从原始影像完整复现 | 54 GB 影像 + API key | 按上面第 2–6 节 |

原始 NIfTI 影像和掩膜不放在 GitHub 上：一是体积太大（54 GB），二是 TCIA 有数据使用条款，需从 TCIA 下载。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `Set the environment variable ZHIPU_API_KEY` | 在 PowerShell 里用了 `set`；改用 `$env:ZHIPU_API_KEY="..."`，并在同一个窗口运行 |
| 智谱返回 401 | key 不完整（缺少“.”后面那段）或已过期 |
| 硅基流动返回 403 / 30003 | 该模型对你的账号不可用，换一个模型 |
| 第 01 步运行时 PyCharm 被关闭 | 内存不足；在 run_study 脚本里把 `--n_jobs 0` 改成较小的数字，例如 `--n_jobs 4` |
| 第 13 步长时间没有输出 | 正在重试超时的请求；每完成 10 条会打印一行进度，可先用 check_api.py 排查网络 |
| 第三位小数和论文略有不同 | 不同机器上 PINN 训练有微小差异（实测 AUROC 差 0.002）；大模型重复调用也会略有不同。要完全一致，请使用发布的结果包 |

## 注意事项

- 仓库的 `data/` 包含去标识化的派生数据（来源和使用条款见 `data/README.md`），但不包含原始影像和掩膜；`.gitignore` 已排除你自己运行产生的 `results/`、NIfTI 文件和 API 缓存。
- 第 13 步会把去标识化的影像拼图和特征发送给第三方 API。运行前请确认 TCIA/BreastDCEDL 的数据使用条款和你所在机构的规定允许这样做。
- 不要把 API key 提交到 GitHub。
