# -*- coding: utf-8 -*-
"""PATSP co-evolution driver: R-Zero (sparse final reward) vs PATSP (trajectory-level credit).

ONE arm per process (pin to one GPU). Each round:
  gen learnable tasks (band challenger, probed by CURRENT solver)
    -> rollout group_size full trajectories
    -> [PATSP arm only] rewrite terminal_reward with outcome-grounded dense credit (outcome_credit.py)
    -> split by role -> GRPO train planner'+executor' (lr=3e-5, 1 epoch -- the STABLE config)
    -> eval on a FROZEN held-out D2 using the RAW terminal_reward (the true metric).

R-Zero arm:  --arm rzero  -> trains on raw terminal_reward (sparse).      [baseline]
PATSP  arm:  --arm patsp  -> trains on outcome-grounded dense credit (learned V over progress).

Same challenger / tasks / trainer for both arms -> the ONLY difference is the training reward,
so a D2 win isolates the trajectory-credit contribution. Single seed, GPU-only.
Writes per-round D2 to <exp>/<name>/rounds.json.  See repo-root TUTORIAL.md.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weakness_extractor import extract_profile          # noqa: E402  (component B)
from patsp_challenger import next_difficulty            # noqa: E402  (component C)

SCR = "scripts/rzero_nofsdp_lora"
PY = sys.executable
# Data/results store (trained adapters, eval sets, run outputs) -- external by convention; one env knob.
_RZ_STORAGE = os.environ.get("RZERO_STORAGE", os.path.expanduser("~/RZero_storage"))
R3 = f"{_RZ_STORAGE}/lh_halo_run3/agentic"
ENV = "textcraft"
D2_FROZEN = ""
# Fresh near-zero LoRA adapters (lora_B=0 => behaves as the base model) -- the cold-start point
# for BOTH arms (the GRPO trainer requires a --lora_in; a fresh adapter == base behaviour).
INIT_P = f"{_RZ_STORAGE}/patsp_init/planner"
INIT_E = f"{_RZ_STORAGE}/patsp_init/executor"
BASE_MODEL = os.environ["BASE_MODEL"]
GPU_UTIL_SHARED = os.environ.get("RZERO_SHARED_GPU_UTIL", "0.55")
VC = ["--temperature", "0.7", "--top_p", "0.95", "--max_tokens", "256", "--gpu_memory_utilization", GPU_UTIL_SHARED]
GEN_N = os.environ.get("RZERO_GEN_NUM_TASKS", "12")
MAXSTEPS = os.environ.get("PATSP_MAX_STEPS", "20")
MAXREPLANS = os.environ.get("PATSP_MAX_REPLANS", "3")
# Challenger learnability band. For a WEAK base on a small-pool env (HeroBench: 9 train tasks, no
# difficulty ladder) set RZERO_SCORE_MIN=0.0 so gen doesn't starve (accept all not-fully-solved tasks;
# zero-variance rows are dropped at train time anyway).
SCORE_MIN = os.environ.get("RZERO_SCORE_MIN", "0.1")
SCORE_MAX = os.environ.get("RZERO_SCORE_MAX", "0.9")

# Full-parameter mode (set in main from --full_param). In full-param both roles are independent
# full checkpoints, each served by its OWN vLLM engine on the one visible card -> halve the per-engine
# memory budget. Continuation = train with --base_model <- the role's CURRENT checkpoint (no adapter).
FULL_PARAM = False
GPU_UTIL_FULL = os.environ.get("RZERO_FULL_GPU_UTIL", "0.42")  # per-engine; two engines share one card
LR_FULL = os.environ.get("RZERO_TRAIN_LR_FULL", "1e-6")        # full-param lr (1e-4 collapses; LoRA used 3e-5)



def _same_model_path(a, b):
    """Best-effort comparison for local model paths.

    Used to distinguish true role checkpoints from the shared BASE_MODEL in
    full-param cold start. If planner/executor equal BASE_MODEL, we should use
    the shared --base_model engine rather than loading duplicate full engines.
    """
    if not a or not b:
        return False
    aa = os.path.expanduser(str(a))
    bb = os.path.expanduser(str(b))
    try:
        return os.path.samefile(aa, bb)
    except OSError:
        return os.path.abspath(aa) == os.path.abspath(bb)

def policy_args(planner, executor):
    """Role->model wiring for a gen/rollout/eval subprocess.
    LoRA mode: --planner_lora/--executor_lora on a shared-base engine.
    Full-param: --planner_model/--executor_model (two engines) + a halved gpu_util that
    OVERRIDES VC's value (argparse keeps the last --gpu_memory_utilization)."""
    a = []
    if FULL_PARAM:
        # In full-param mode, trained planner/executor checkpoints are separate full
        # models. But during cold start the driver may pass BASE_MODEL as planner/executor;
        # that should still use the shared --base_model engine from VC.
        planner_is_base = (not planner) or _same_model_path(planner, BASE_MODEL)
        executor_is_base = (not executor) or _same_model_path(executor, BASE_MODEL)
        using_role_models = bool((planner and not planner_is_base) or (executor and not executor_is_base))

        if planner and not planner_is_base:
            a += ["--planner_model", planner]
        if executor and not executor_is_base:
            a += ["--executor_model", executor]
        if using_role_models:
            a += ["--gpu_memory_utilization", GPU_UTIL_FULL]
        return a
    if planner:
        a += ["--planner_lora", planner]
    if executor:
        a += ["--executor_lora", executor]
    return a


def run(cmd, log):
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"FAILED ({rc}): {' '.join(str(c) for c in cmd[:6])}... see {log}")


def gen(d, planner, executor, out_jsonl, seed, odir, tag):
    # Stage-level reuse on restart: the summary file is written LAST by the gen script, so its
    # presence means this round's challenger output is complete -> skip the (slow, ~20min on
    # JVM envs) probing instead of redoing it after an interrupted round.
    summ = out_jsonl.replace(".jsonl", ".summary.json")
    if os.path.exists(out_jsonl) and os.path.exists(summ) and os.path.getsize(out_jsonl) > 0:
        print(f"### {tag} gen: reusing existing {out_jsonl} (complete)", flush=True)
        return json.load(open(summ)).get("stats", {})
    # LongReason is an adapted one-step static-task benchmark. Its "challenger"
    # samples from a fixed train split rather than synthesizing new interactive tasks.
    # The default learnability band rejects these tasks under random probing, so for
    # LongReason we disable band filtering and use a cheap random probe.
    gen_score_min = SCORE_MIN
    gen_score_max = SCORE_MAX
    gen_probe_k = "3"
    gen_max_attempts = "60"
    gen_policy = "vllm"
    gen_extra = ["--base_model", BASE_MODEL] + VC + policy_args(planner, executor)

    if ENV == "longreason":
        gen_score_min = os.environ.get("RZERO_LONGREASON_SCORE_MIN", "0.0")
        gen_score_max = os.environ.get("RZERO_LONGREASON_SCORE_MAX", "1.0")
        gen_probe_k = os.environ.get("RZERO_LONGREASON_PROBE_K", "1")
        gen_max_attempts = os.environ.get("RZERO_LONGREASON_MAX_ATTEMPTS", "60")
        gen_policy = os.environ.get("RZERO_LONGREASON_GEN_POLICY", "random")
        if gen_policy == "random":
            gen_extra = []
        else:
            gen_extra = ["--base_model", BASE_MODEL] + VC + policy_args(planner, executor)

    cmd = [PY, f"{SCR}/generate_agentic_challenger_tasks.py", "--env", ENV, "--out_jsonl", out_jsonl,
           "--num_tasks", GEN_N, "--difficulty", f"{d:.3f}", "--score_min", gen_score_min, "--score_max", gen_score_max,
           "--probe_k", gen_probe_k, "--max_attempts", gen_max_attempts, "--policy", gen_policy,
           "--max_steps", MAXSTEPS, "--max_replans", MAXREPLANS, "--seed", str(seed)] + gen_extra
    run(cmd, f"{odir}/{tag}_gen.log")
    st = json.load(open(out_jsonl.replace(".jsonl", ".summary.json"))).get("stats", {})
    return st


def rollout(tasks, planner, executor, out_jsonl, seed, odir, tag):
    cmd = [PY, f"{SCR}/build_agentic_rollouts.py", "--env", ENV, "--tasks_jsonl", tasks, "--num_tasks", "0",
           "--out_jsonl", out_jsonl, "--group_size", "4", "--max_steps", MAXSTEPS, "--max_replans", MAXREPLANS,
           "--policy", "vllm", "--base_model", BASE_MODEL, "--seed", str(seed)] + VC + policy_args(planner, executor)
    run(cmd, f"{odir}/{tag}_roll.log")


def train_eval(traj, planner_in, executor_in, odir, seed, tag, arm, a):
    # PATSP arm: rewrite the training reward with outcome-grounded dense credit before split/train.
    train_traj = traj
    if arm == "patsp":
        # outcome-grounded dense credit (learned V over progress buckets; not hand-crafted)
        train_traj = f"{odir}/{tag}_traj_shaped.jsonl"
        run([PY, f"{SCR}/patsp/outcome_credit.py", "--in_jsonl", traj, "--out_jsonl", train_traj,
             "--lam", str(a.outcome_lambda)], f"{odir}/{tag}_credit.log")
    pr, er = f"{odir}/{tag}_planner.jsonl", f"{odir}/{tag}_exec.jsonl"
    run([PY, f"{SCR}/split_agentic_rollouts_by_role.py", "--trajectories_jsonl", train_traj,
         "--out_planner", pr, "--out_executor", er], f"{odir}/{tag}_split.log")
    suffix = "_model" if FULL_PARAM else "_lora"
    pout, eout = f"{odir}/{tag}_planner{suffix}", f"{odir}/{tag}_exec{suffix}"
    ep = os.environ.get("RZERO_TRAIN_EPOCHS", "1")
    lr = LR_FULL if FULL_PARAM else os.environ.get("RZERO_TRAIN_LR", "3e-5")
    for lin, roll, lout in [(planner_in, pr, pout), (executor_in, er, eout)]:
        train_backend = os.environ.get("RZERO_TRAIN_BACKEND", "single").lower()
        fsdp_nproc = os.environ.get("RZERO_FSDP_NPROC", "2")
        if FULL_PARAM:
            # full-param continuation:
            # --base_model = the role's CURRENT full checkpoint (lin)
            # train ALL weights, write a fresh full checkpoint (lout).
            if train_backend == "fsdp":
                cmd = [
                    PY, "-m", "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node", fsdp_nproc,
                    f"{SCR}/train_fsdp_grpo_clip_from_rollouts.py",
                    "--base_model", lin,
                    "--full_param",
                    "--rollouts_jsonl", roll,
                    "--lora_out", lout,
                    "--objective", "grpo_clip",
                    "--epochs", ep,
                    "--credit_mode", "uniform",
                    "--max_seq_len", "16384",
                    "--lr", lr,
                ]
            else:
                cmd = [
                PY, f"{SCR}/train_lora_grpo_clip_from_rollouts.py",
                "--base_model", lin,
                "--full_param",
                "--rollouts_jsonl", roll,
                "--lora_out", lout,
                "--objective", "grpo_clip",
                "--epochs", ep,
                "--credit_mode", "uniform",
                "--max_seq_len", "16384",
                "--lr", lr,
                ]
        else:
            cmd = [
                PY, f"{SCR}/train_lora_grpo_clip_from_rollouts.py",
                "--base_model", BASE_MODEL,
                "--rollouts_jsonl", roll,
                "--lora_out", lout,
                "--objective", "grpo_clip",
                "--epochs", ep,
                "--credit_mode", "uniform",
                "--max_seq_len", "16384",
                "--lr", lr,
                ]
            if lin:
                cmd += ["--lora_in", lin]
        logp = f"{odir}/{tag}_train_{os.path.basename(lout)}.log"
        try:
            run(cmd, logp)
        except RuntimeError:
            # A "flat" round (no within-group reward variance -> all advantages 0 -> 0 usable rows)
            # is a TRANSIENT data condition on the tiny task pool, NOT a fatal error. Carry this role's
            # checkpoint forward UNCHANGED (no update this round) so multi-round self-play continues
            # instead of crashing the whole arm. Any OTHER training failure still propagates.
            txt = open(logp).read() if os.path.exists(logp) else ""
            if ("No usable rollout rows" in txt) or ("No usable examples" in txt) or ("No optimizer update" in txt):
                print(f"### {tag} {os.path.basename(lout)}: flat round (no usable rows) -> carry checkpoint "
                      f"forward unchanged (no update)", flush=True)
                if os.path.abspath(lin) != os.path.abspath(lout):
                    if os.path.isdir(lout):
                        shutil.rmtree(lout, ignore_errors=True)
                    shutil.copytree(lin, lout)
            else:
                raise
    # eval on frozen D2 with RAW terminal_reward (true metric)
    ev = f"{odir}/{tag}_eval.jsonl"
    run([PY, f"{SCR}/build_agentic_rollouts.py", "--env", ENV, "--tasks_jsonl", D2_FROZEN, "--num_tasks", "0",
         "--out_jsonl", ev, "--group_size", os.environ.get("LH_EVAL_GROUP", "2"), "--max_steps", MAXSTEPS,
         "--max_replans", MAXREPLANS, "--policy", "vllm", "--base_model", BASE_MODEL,
         "--seed", "9100"] + VC + policy_args(pout, eout), f"{odir}/{tag}_eval.log")
    d2 = json.load(open(os.path.join(os.path.dirname(ev), "summary.json")))["mean_terminal_reward"]
    return pout, eout, d2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True, choices=["rzero", "patsp"])
    ap.add_argument("--env", default="textcraft")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--d0", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=0.0, help="difficulty increment per round (0 = fixed)")
    ap.add_argument("--d2_tasks", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--credit_mode", default="outcome", choices=["outcome"],
                    help="PATSP credit form: outcome-grounded learned dense credit.")
    ap.add_argument("--outcome_lambda", type=float, default=0.5, help="weight for outcome-grounded credit")
    ap.add_argument("--use_weakness_challenger", type=int, default=0,
                    help="1 = component C: set next-round difficulty from the weakness profile")
    ap.add_argument("--full_param", type=int, default=0,
                    help="1 = full-parameter fine-tune both roles (no LoRA). Each role is its own full "
                         "checkpoint + own vLLM engine; continuation trains --base_model<-prev checkpoint.")
    a = ap.parse_args()
    global ENV, D2_FROZEN, FULL_PARAM
    ENV = a.env
    D2_FROZEN = a.d2_tasks
    FULL_PARAM = bool(a.full_param)
    odir = f"{a.exp}/{a.name}"
    os.makedirs(odir, exist_ok=True)
    # cold start = base model (both arms identical start). Full-param: the base model IS the start
    # checkpoint for both roles; LoRA: fresh near-zero adapters.
    planner, executor = (BASE_MODEL, BASE_MODEL) if FULL_PARAM else (INIT_P, INIT_E)
    d = a.d0
    history = []
    start_round = 1
    best_round, best_d2 = None, -1.0   # full-param: keep the BEST-D2 checkpoint (+latest) for later eval
    # ---- RESUME (PBS-batch robust) : skip rounds already recorded in rounds.json whose checkpoints
    # still exist on disk, and continue from the last good checkpoint. rounds.json is written only AFTER
    # a round's checkpoints are saved, so any recorded round is complete; a partial/in-flight round (no
    # record) is simply redone (its files overwritten). Disk-hygiene keeps only the LAST round's ckpts,
    # which is exactly what we resume from.
    suffix = "_model" if FULL_PARAM else "_lora"
    rounds_path = f"{odir}/rounds.json"
    if os.path.exists(rounds_path):
        try:
            prev_hist = json.load(open(rounds_path))
        except Exception:
            prev_hist = []
        done = len(prev_hist)
        usable = 0
        # Disk-hygiene keeps ONLY the latest round's checkpoints, so scan DOWN from the last recorded
        # round to the highest round whose checkpoints still exist (normally == done). Scanning up and
        # breaking at the first gap would wrongly see 0 (rounds 1..done-1 ckpts are deleted by design).
        for rr in range(done, 0, -1):
            cp, ce = f"{odir}/r{rr}_planner{suffix}", f"{odir}/r{rr}_exec{suffix}"
            if os.path.isdir(cp) and os.path.isdir(ce):
                usable = rr
                break
        if usable > 0:
            history = prev_hist[:usable]
            bi = max(range(len(history)), key=lambda i: history[i].get("D2", -1.0))
            best_round, best_d2 = history[bi]["round"], history[bi].get("D2", -1.0)
            planner = f"{odir}/r{usable}_planner{suffix}"
            executor = f"{odir}/r{usable}_exec{suffix}"
            start_round = usable + 1
            d = min(0.95, a.d0 + a.step * usable)   # fixed-step difficulty for the next round
            print(f"### RESUME {a.name}: {usable}/{a.rounds} rounds done -> continue at round {start_round} "
                  f"(planner={planner})", flush=True)
    # Clean time-budget self-stop (PnP-CM-style): never get SIGKILLed mid-round. Before STARTING a
    # round, if the wall-budget is spent, exit 0 so the PBS wrapper requeues and round-level resume
    # continues from here. 0 = disabled. The in-flight round always finishes (we only gate new rounds).
    t0 = time.time()
    max_seconds = float(os.environ.get("RZERO_MAX_SECONDS", "0"))
    for r in range(start_round, a.rounds + 1):
        if max_seconds > 0 and (time.time() - t0) > max_seconds:
            print(f"### {a.name} SELF-STOP before round {r}: time budget {max_seconds:.0f}s spent, "
                  f"{len(history)} rounds done -> clean exit for requeue", flush=True)
            break
        prev_planner, prev_executor = planner, executor
        tag = f"r{r}"
        tasks = f"{odir}/{tag}_tasks.jsonl"
        st = gen(d, planner, executor, tasks, a.seed, odir, tag)
        traj = f"{odir}/{tag}_traj.jsonl"
        rollout(tasks, planner, executor, traj, a.seed + 2000, odir, tag)
        # component B: weakness profile of THIS round's rollouts (logged for both arms)
        try:
            prof = extract_profile([json.loads(l) for l in open(traj) if l.strip()])
        except Exception:
            prof = {}
        pout, eout, d2 = train_eval(traj, planner, executor, odir, a.seed, tag, a.arm, a)
        n = sum(1 for _ in open(tasks))
        rec = {"round": r, "arm": a.arm, "difficulty": round(d, 3), "n_train_tasks": n, "D2": round(d2, 4),
               "dominant_failure": prof.get("dominant_failure"), "mean_subgoal_frac": prof.get("mean_subgoal_frac"),
               "recovery_rate": prof.get("recovery_rate"), "bottleneck_preds": prof.get("bottleneck_preds")}
        history.append(rec)
        json.dump(history, open(f"{odir}/rounds.json", "w"), indent=2)
        print(f"### {a.name} {tag} arm={a.arm} d={d:.2f} n={n} D2={d2:.4f} "
              f"dom={prof.get('dominant_failure')} comp={prof.get('mean_subgoal_frac')} "
              f"rec={prof.get('recovery_rate')}", flush=True)
        planner, executor = pout, eout
        # full-param disk hygiene: keep the BEST-D2 checkpoint (for final eval) AND the latest (for
        # resume/next round); drop everything else. Never the base model / LoRA init (outside odir).
        if FULL_PARAM:
            _od = os.path.abspath(odir)
            if d2 > best_d2:
                # new best is THIS round (current ckpt, kept anyway) -> the OLD best can now be dropped
                if best_round is not None and best_round != r:
                    for nm in (f"r{best_round}_planner{suffix}", f"r{best_round}_exec{suffix}"):
                        p = os.path.join(odir, nm)
                        if os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)
                best_round, best_d2 = r, d2
            # delete the round we advanced FROM (prev), unless it is the kept best
            for old in (prev_planner, prev_executor):
                if old and os.path.abspath(old).startswith(_od):
                    if not (best_round is not None and os.path.basename(old).startswith(f"r{best_round}_")):
                        shutil.rmtree(old, ignore_errors=True)
        # component C: weakness-targeted difficulty for next round (else fixed step)
        if a.use_weakness_challenger and prof:
            d, reason = next_difficulty(d, prof, step=(a.step or 0.15))
            print(f"### {a.name} {tag} challenger: d->{d:.2f} ({reason})", flush=True)
        else:
            d = min(0.95, d + a.step)
    print(f"### {a.name}_ALL_DONE {json.dumps(history)}", flush=True)


if __name__ == "__main__":
    main()
