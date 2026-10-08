"""
Command line of the reproduction pipeline:

    python -m shapfusion download                  data (+ Natural Earth for the maps), md5-checked
    python -m shapfusion run [filters]             train + infer every job (resumes: done stages are skipped)
    python -m shapfusion status [filters]          which stages are done
    python -m shapfusion report [--only A B ...]   every figure / table / number of the paper

Filters: --models CoM EmbraceNet ... --experiments main noise cloud_gap --levels 0.5 3 ...
--folds 0 1 ... ; --stages train vdict ps local ; --force re-runs done stages.
--smoke runs the whole pipeline on a 3,000-sample subset in minutes (runs_smoke/,
paper_outputs_smoke/): it checks that everything runs, its numbers are meaningless.
"""
import argparse
import sys
import time

from shapfusion import config as C


def _parser():
    p = argparse.ArgumentParser(prog="python -m shapfusion", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="default: configs/paper.yaml")
    common.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, 2 folds (minutes)")
    common.add_argument("--runs-dir", default=None)
    common.add_argument("--out-dir", default=None)
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--models", nargs="+")
    filters.add_argument("--experiments", nargs="+", choices=C.EXPERIMENTS)
    filters.add_argument("--levels", nargs="+", type=float)
    filters.add_argument("--folds", nargs="+", type=int)
    sub.add_parser("download", parents=[common], help="download the data")
    r = sub.add_parser("run", parents=[common, filters], help="train and infer the jobs")
    r.add_argument("--stages", nargs="+", default=["train", "vdict", "ps", "local"],
                   choices=["train", "vdict", "ps", "local"])
    r.add_argument("--force", action="store_true")
    r.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("status", parents=[common, filters], help="done / missing stages")
    rep = sub.add_parser("report", parents=[common], help="figures, tables, numbers")
    rep.add_argument("--only", nargs="+", choices=list("ABCDEFGH"))
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    cfg = C.load(args.config, smoke=args.smoke, runs_dir=args.runs_dir, out_dir=args.out_dir)
    if args.cmd == "download":
        from shapfusion.download import download
        download(cfg)
        return
    if args.cmd == "report":
        from shapfusion.report.run import run
        status = run(cfg, only=args.only)
        sys.exit(0 if all(s.startswith("done") for s in status.values()) else 1)
    from shapfusion import job as J
    jobs = J.all_jobs(cfg, args.models, args.experiments, args.levels, args.folds)
    if args.cmd == "status":
        n_done = 0
        for j in jobs:
            st = J.stages_of(cfg, j)
            ok = [s for s in st if J.done(cfg, j, s)]
            n_done += len(ok) == len(st)
            print(f"{str(j):<45} " + "  ".join(f"{s}:{'ok' if s in ok else '--'}" for s in st))
        print(f"{n_done}/{len(jobs)} jobs complete")
        return
    if args.dry_run:
        for j in jobs:
            todo = [s for s in J.stages_of(cfg, j) if s in args.stages and (args.force or not J.done(cfg, j, s))]
            print(f"{str(j):<45} {', '.join(todo) or 'done'}")
        return
    runner = J.Runner(cfg, device=args.device)
    t0, n = time.time(), 0
    for i, j in enumerate(jobs, 1):
        if runner.run(j, stages=args.stages, force=args.force):
            n += 1
            print(f"    ({i}/{len(jobs)} jobs, {(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
    print(f"{n} job(s) run, {len(jobs) - n} already done, {(time.time() - t0) / 60:.1f} min")
