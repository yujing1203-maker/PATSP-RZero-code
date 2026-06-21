# PATSP / R-Zero 自博弈实验教程

本仓库在长程智能体任务上比较两种自博弈训练奖励的代码与教程。


base model - Qwen3-4B-Instruct
baseline Qwen3-4B-Instruct-GRPO
baseline Qwen3-4B-Instruct-RZero
Our

## 一、PATSP 方法

长程(long-horizon)智能体任务上,比较两种自博弈训练奖励:

- **R-Zero(基线)**:任务级自博弈。challenger 出题、solver 答题,solver 只拿稀疏的最终奖励。
- **PATSP**:轨迹级自博弈。核心是 **outcome-grounded dense credit**——从本轮 rollout 的真实成败学一张 V(progress)("走到这个进度的人最终成功的概率"),给中间进度发奖励。区别于手工子目标加分(手工 dense reward 会被 GRPO 刷分 Goodhart、训崩);从 outcome 学出的 credit 不可刷(刷无用子目标 V=0)。solver 拆成 **planner**(分解可验证子目标)+ **executor**(执行动作),各一个模型(LoRA 模式=同 base 两个 adapter;全参模式=两个独立 checkpoint)。

## 二、仓库结构

**方法 `scripts/rzero_nofsdp_lora/patsp/`:**
| 文件 | 是什么 |
|---|---|
| `patsp_selfplay.py` | 共演化 driver,一进程一 arm/一卡;每轮 = gen 题 → rollout → credit → 角色拆分 → GRPO → D2 eval;轮级续跑(`rounds.json`) |
| `outcome_credit.py` | PATSP 核心:outcome-grounded 学习的稠密 credit V(progress) |
| `weakness_extractor.py` | 从 `StageRecord.failure_type` 提取 solver 弱点 |
| `patsp_challenger.py` | 针对弱点的对抗 challenger |
| `run_herobench_selfplay.sh` / `run_selfplay_seq.sh` | 一键 launcher(HeroBench 专用双臂 / 通用全参) |

**环境 `scripts/rzero_nofsdp_lora/envs/`:** 四个 benchmark(`textcraft/scienceworld/alfworld/herobench`)都实现 FROZEN 的 `base.py::AgenticEnv`;`--env` 选项处处就是这四个。`third_party/HeroBench/` 是自带的 HeroBench FastAPI/SQLite server(按端口隔离 DB)。

**复用的 R-Zero:** `build_agentic_rollouts.py`(多轮 rollout + 评估,同一脚本)、`generate_agentic_challenger_tasks.py`(challenger)、`split_agentic_rollouts_by_role.py`(角色拆分)、`train_lora_grpo_clip_from_rollouts.py`(GRPO 训练器,LoRA/`--full_param` 双模式)。

**一键脚本 `scripts/run/`:** `check_env` / `probe` / `selfplay` / `eval` / `status`(见五,推荐入口)。
**数据/结果:** 模型在 `data/models/`;实验产物/评估集/checkpoint 在 `$RZERO_STORAGE`。

## 三、环境配置

环境配置见 `.install_blackwell.sh`:

```
torch 2.8.0+cu128 / vllm 0.11.0 / transformers 4.57.6 / peft 0.19.1
```

每个跑模型的 shell 都要带(`scripts/run/` 脚本经 `_common.sh` 已自动设好这些;只在手动跑底层脚本时需要):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate agent
export VLLM_ATTENTION_BACKEND=FLASH_ATTN        # 不设 vLLM 会反复 "init attempt 1/6 failed"
export VLLM_USE_FLASHINFER_SAMPLER=0            # 根因是 flashinfer JIT 需 gcc≥9(系统 gcc 8.5),非显存问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

路径约定:

```bash
export RZERO_PROJECT=<仓库根目录>      # 代码自动从 __file__ 推,一般不用设
export RZERO_STORAGE=<数据/结果仓>     # 默认 $HOME/RZero_storage;放大盘别放 home(全参 ckpt ~8.6G)
```

模型放 `data/models/`,实验产物进 `$RZERO_STORAGE`;**没有评估集时**:`bash scripts/data/build_eval_sets.sh` 生成等价的 held-out D2 到 `$RZERO_STORAGE/patsp_eval/`(详见五)。

## 四、Benchmark

都实现 `envs/base.py::AgenticEnv`(`reset / step / verify_final / predicate_library / sample_new_task`),在 `build_agentic_rollouts.py` 和 `generate_agentic_challenger_tasks.py` 的 `build_env` + `--env` choices **两处**注册(加新环境两处都改)。`terminal_reward` 归一到 [0,1]=子目标完成度。

**1. TextCraft**(`envs/textcraft_env.py`)— Minecraft 风格合成 DAG,**零依赖纯 Python,调试管线首选**。难度=配方树深度(`difficulty→depth 1~5`)。

**2. ScienceWorld**(`envs/scienceworld_env.py`):
```bash
pip install --no-deps scienceworld py4j     # jar 首次使用自动下载,需 /bin/java
```
任务梯子 `_LADDER`:找生物(导航)→ 煮沸/熔化/冷冻(状态变化)→ 电路/种植(长链)。`difficulty` 线性映射;**d=0.5 落在 freeze,对小模型过难**。

**3. ALFWorld**(`envs/alfworld_env.py`):
```bash
pip install --no-deps alfworld textworld
pip install "tatsu==5.8.3"
pip install mementos prompt_toolkit pybars3 hashids jericho
pip install --no-deps fast-downward-textworld   # 包名不是 fast_downward
ALFWORLD_DATA=$RZERO_STORAGE/alfworld_data alfworld-download
```
数据 `json_2.1.1/`(3553 train + 274 valid)。奖励按里程碑(take/clean/heat/cool/place)给部分分 → **raw reward 本身就 dense**,所以 PATSP 在它身上几乎不起作用。base D2≈0.667。

**4. HeroBench**(`envs/herobench_env.py` + `third_party/HeroBench/`,项目内自带)— FastAPI+SQLite client-server,adapter 懒启动,按 `HEROBENCH_PORT` 选端口:
- **每端口独立数据库**(`HEROBENCH_DB=artifact_<port>.db`):server 启动时 `rm_db()` 清库,不隔离则并行两臂互相抹掉对方世界;
- **训练/评估任务隔离**(`HEROBENCH_TRAIN_EXCLUDE=0,4,9,11`):池子仅 13 任务,这 4 个(gold/copper_ring/mushstaff/steel_boots)保留作 held-out D2。

craft 失败时 server 返回结构化失败原因 → 映射成 `missing/constraint_violation`,四环境中失败信号最结构化。base D2≈0.036(奖励极稀疏,PATSP 增益最大)。

**评估集(D2)** 在 `$RZERO_STORAGE/patsp_eval/`:`textcraft_EVAL30.jsonl`(30 个 depth-3,即难度 0.5,主力评估集)、`textcraft_D2.jsonl`、`herobench_D2.jsonl`(4 个保留任务,2~7 子目标)、`alfworld_D2.jsonl` / `scienceworld_D2_easy.jsonl`(各 10)。每行是序列化的 Task `{task_id, spec, constraints, info}`,用 `--tasks_jsonl` 喂给 rollout。**没有这些文件时**用第五节的 `build_eval_sets.sh` 生成。

## 五、训练 / 评估脚本

```bash
# ① 环境体检,逐项 [OK]/[FAIL]:conda / torch / vllm / GPU / 模型 / D2 集 / 数据 / 管线导入
bash scripts/run/check_env.sh

# ② 探针:8 episode。退出码 0=有信号,2=无信号
#    诊断:乱码或复读 = 模型不遵循格式(换 instruct);动作正常但 0 分 = 难度过高(--d0 0.0 降易端)或模型能力不足
bash scripts/run/probe.sh alfworld                          # 默认 4B-instruct, GPU0
bash scripts/run/probe.sh herobench data/models/qwen3-4b-base 1   # 指定模型和 GPU
#    退出码可链式:probe.sh <env> && selfplay.sh <env>(仅探针有信号才继续)

# ③ 共演化主实验:R-Zero=GPU0,PATSP=GPU1
bash scripts/run/selfplay.sh alfworld                # 默认:全参 4B-instruct, lr=2e-6, 6 轮
bash scripts/run/selfplay.sh herobench --rounds 6    # HeroBench 走专用 launcher(端口/DB 隔离)
bash scripts/run/selfplay.sh textcraft --lora        # LoRA 模式(用 7B base)

# ④ 可选
RZERO_MAX_SECONDS=14400 bash scripts/run/selfplay.sh alfworld --rounds 6

# ⑤ 评估:与 base 并排比较
bash scripts/run/eval.sh herobench base              # 先评 base 作参照
bash scripts/run/eval.sh herobench \
  --planner  $RZERO_STORAGE/patsp_herobench_full_lr2e6/patsp_oc/r2_planner_model \
  --executor $RZERO_STORAGE/patsp_herobench_full_lr2e6/patsp_oc/r2_exec_model
bash scripts/run/eval.sh textcraft --big --planner-lora <adapter> --executor-lora <adapter> \
  --model data/models/qwen2.5-7b-instruct          # LoRA 用 --planner-lora/--executor-lora;--big = 30任务×8

# 进度
bash scripts/run/status.sh                           # 扫 $RZERO_STORAGE/patsp_*
```

默认值均为验证过的稳定配置:lr=2e-6(全参;LoRA 默认 3e-5)、d0 按 env(ScienceWorld 0.0 易端,其余 0.5)、复读惩罚 1.2、上下文 8192。

**新 benchmark 从零到结果:**
```bash
bash scripts/run/check_env.sh                                             # ① 体检
bash scripts/run/probe.sh alfworld                                        # ② 探针
RZERO_MAX_SECONDS=14400 bash scripts/run/selfplay.sh alfworld --rounds 6   # ③ 跑 6 轮(自停+续跑)
bash scripts/run/status.sh                                                #    查看进度
bash scripts/run/eval.sh alfworld base                                    # ④ 评 base,再评 checkpoint
# 结果记入 $RZERO_STORAGE/RESULTS_MASTER.md
```

**没有评估集时**(冻结集在 `$RZERO_STORAGE`,不随代码发布):
```bash
bash scripts/data/build_eval_sets.sh        # 生成等价 held-out D2 到 $RZERO_STORAGE/patsp_eval/
```
直接调各 env 的 `sample_new_task` 在 eval 难度采样(TextCraft/ALFWorld 纯 CPU;ScienceWorld 需 java;HeroBench 用其自带数据)。



