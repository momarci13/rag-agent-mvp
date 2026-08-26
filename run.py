#!/usr/bin/env python
"""Quant Research + IBKR RAG Agents CLI entrypoint.

Examples:
    python run.py "Compute the volatility of SPY over 2020-2024 with bootstrap CIs"
    python run.py "Design and backtest a 50/200 SMA crossover on SPY, QQQ, IWM"
    python run.py "Write a 3-page report on momentum factor anomalies"
    python run.py --ingest data/papers/   # build the KB
    python run.py --healthcheck
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from agents.llm import HostedLLM, LLMConfig, ModelSpec, ModelSelectionStrategy
from tools.backtest import (
    BacktestConfig,
    compile_signal,
    run_portfolio_backtest,
)
from tools.data import MultiSourceFetcher
from tools.multifidelity_kan import (
    ResidualKAN,
    evaluate_regression,
    generate_multifidelity_dataset,
)
from tools.sandbox import run_py, run_cpp, run_c
from tools.tex import build_latex_artifact


ROOT = Path(__file__).resolve().parent


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    embedding_override = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
    if embedding_override:
        cfg["rag"]["embedding_model"] = embedding_override
    return cfg


def make_llm_config(cfg: dict) -> LLMConfig:
    """Create the direct OpenAI client config; secrets come from the environment.

    NOTE: every call made through this config incurs real OpenAI API cost --
    there is no "free tier" route to gate on the way the old OpenRouter setup
    had. Model IDs below are placeholders; confirm the exact IDs you want
    billed against before deploying.
    """
    llm_cfg = cfg["llm"]
    models: list[ModelSpec] | None = None
    if "models" in llm_cfg:
        models = [ModelSpec(**m) for m in llm_cfg["models"]]

    model_override = os.getenv("OPENAI_MODEL", "").strip()
    selected_model = model_override or llm_cfg["model"]
    if model_override:
        models = [ModelSpec(name=model_override, priority=0)] + [
            item for item in (models or []) if item.name != model_override
        ]

    strategy = ModelSelectionStrategy.PRIORITY
    if "selection_strategy" in llm_cfg:
        strategy = ModelSelectionStrategy(llm_cfg["selection_strategy"])

    role_models = dict(llm_cfg.get("role_models", {}))
    if model_override:
        role_models = {role: model_override for role in role_models}

    return LLMConfig(
        model=selected_model,
        base_url=os.getenv("OPENAI_BASE_URL") or llm_cfg.get("base_url") or None,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider_name=llm_cfg.get("provider_name", "OpenAI"),
        require_api_key=bool(llm_cfg.get("require_api_key", True)),
        temperature=llm_cfg.get("temperature", 0.2),
        timeout_s=llm_cfg.get("timeout_s", 180),
        max_output_tokens=llm_cfg.get("max_output_tokens", 8192),
        models=models,
        selection_strategy=strategy,
        fallback_timeout_s=llm_cfg.get("fallback_timeout_s", 60),
        role_models=role_models,
    )


def make_tools(cfg: dict):
    """Bind config into tool callables the agents can use."""
    s_timeout = cfg["agent"]["sandbox_timeout_s"]
    s_mem = cfg["agent"]["sandbox_mem_mb"]

    def _run_py(code: str) -> dict:
        return run_py(code, timeout_s=s_timeout, mem_mb=s_mem, workdir=str(ROOT / "output" / "ds"))

    def _run_cpp(code: str) -> dict:
        return run_cpp(code, timeout_s=s_timeout, mem_mb=s_mem, workdir=str(ROOT / "output" / "ds"))

    def _run_c(code: str) -> dict:
        return run_c(code, timeout_s=s_timeout, mem_mb=s_mem, workdir=str(ROOT / "output" / "ds"))

    def _backtest(spec) -> dict:
        try:
            fetcher = MultiSourceFetcher()
            prices = fetcher.fetch(
                spec.universe,
                start=_lookback_start(spec.lookback_days),
                end="2030-01-01",  # Far future
            )
            if not prices:
                return {"error": "no data fetched"}
            bt_cfg = BacktestConfig(
                initial_equity=cfg["trading"]["initial_equity"],
                rebalance_days=spec.rebalance_days,
                max_weight=cfg["trading"]["max_position_pct"],
                long_only=True,
            )
            sig = compile_signal(spec.signal_code)
            result = run_portfolio_backtest(prices, sig, bt_cfg, n_trials=1)
            # Strip unpicklable pandas objects for JSON-friendliness
            return {
                "sharpe": result["sharpe"],
                "sortino": result["sortino"],
                "deflated_sharpe": result["deflated_sharpe"],
                "max_drawdown": result["max_drawdown"],
                "cagr": result["cagr"],
                "win_rate": result["win_rate"],
                "n_trades": result["n_trades"],
                "final_equity": result["final_equity"],
                "var_99": result["var_99"],
                "symbols": result["symbols"],
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _latex(tex: str, rag_bib: str = "") -> dict:
        bib_path = ROOT / "data" / "papers" / "refs.bib"
        static_bib = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
        combined_bib = static_bib + ("\n\n" + rag_bib if rag_bib else "")
        tex_p, pdf_p, dropped = build_latex_artifact(
            tex, combined_bib, out_dir=str(ROOT / "output" / "paper"),
        )
        return {"tex": str(tex_p), "pdf": str(pdf_p) if pdf_p else None, "dropped_keys": dropped}

    return {"run_py": _run_py, "run_cpp": _run_cpp, "run_c": _run_c, "backtest": _backtest, "latex_build": _latex}


def _lookback_start(days: int) -> str:
    import datetime as dt
    return (dt.date.today() - dt.timedelta(days=max(days, 365) + 60)).isoformat()


def run_kan_demo(samples: int = 400, random_state: int = 123) -> dict[str, Any]:
    data = generate_multifidelity_dataset(
        n_samples=samples,
        n_low_features=3,
        n_high_features=4,
        noise_low=0.45,
        noise_high=0.18,
        random_state=random_state,
    )
    model = ResidualKAN()
    model.fit(data["X_low"], data["y_low"], data["X_high"], data["y_high"])

    y_pred = model.predict(data["X_low"], data["X_high"])
    baseline_pred = model.predict_low(data["X_low"])

    return {
        "kan_metrics": evaluate_regression(data["y_high"], y_pred),
        "baseline_metrics": evaluate_regression(data["y_high"], baseline_pred),
        "n_samples": samples,
        "low_features": data["X_low"].shape[1],
        "high_features": data["X_high"].shape[1],
    }


def healthcheck(cfg: dict) -> int:
    from rag.hybrid import LiteHybridRAG

    print("== Quant Research Agent healthcheck ==")
    llm_cfg = make_llm_config(cfg)
    llm = HostedLLM(llm_cfg)
    ok = llm.health()
    print(f"[{'OK' if ok else 'FAIL'}] OpenAI key validation at {llm_cfg.base_url}")
    if not ok:
        print("    -> Set OPENAI_API_KEY to a key from the OpenAI dashboard.")
        return 1

    try:
        reply = llm.chat(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            temperature=0.0,
        )
        ok_llm = "pong" in reply.lower()
        print(f"[{'OK' if ok_llm else 'WARN'}] Hosted model {llm_cfg.model} responded: {reply[:60]!r}")
    except Exception as e:
        print(f"[FAIL] Model call failed: {e}")
        print("    -> Confirm the OpenAI key is valid and the selected model is available.")
        return 1

    # RAG
    try:
        rag = LiteHybridRAG(
            db_path=cfg["rag"]["db_path"],
            collection=cfg["rag"].get("collection", "openai-embed-v1"),
            embedding_model=cfg["rag"]["embedding_model"],
            alpha_dense=cfg["rag"]["alpha_dense"],
            query_expansion_enabled=cfg["rag"]["query_expansion"]["enabled"],
            query_expansion_method=cfg["rag"]["query_expansion"]["method"],
            max_expansions=cfg["rag"]["query_expansion"]["max_expansions"],
            reranking_enabled=cfg["rag"]["reranking"]["enabled"],
            reranking_model=cfg["rag"]["reranking"]["model"],
            top_k_before_rerank=cfg["rag"]["reranking"]["top_k_before_rerank"],
            top_k_after_rerank=cfg["rag"]["reranking"]["top_k_after_rerank"],
        )
        print(f"[OK] RAG ready. {len(rag)} chunks in store.")
        probe = rag.emb.encode(["hosted embedding health probe"])
        if probe.shape[0] != 1 or probe.shape[1] < 1:
            raise ValueError(f"Unexpected embedding shape: {probe.shape}")
        print(
            f"[OK] Hosted embedding {cfg['rag']['embedding_model']} "
            f"returned {probe.shape[1]} dimensions."
        )
    except Exception as e:
        print(f"[FAIL] RAG/embedding check: {e}")
        return 1

    # Market data
    try:
        fetcher = MultiSourceFetcher()
        data = fetcher.fetch(["SPY"], start="2024-01-01", end="2025-01-01")
        print(f"[{'OK' if data else 'WARN'}] yfinance SPY: {len(data.get('SPY', []))} bars")
    except Exception as e:
        print(f"[WARN] yfinance: {e}")

    print("\nAll green - run a task with: python run.py \"your task here\"")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Hosted Free-Model Quant Research + IBKR RAG Agents")
    ap.add_argument("task", nargs="?", help="natural-language task")
    ap.add_argument("--ingest", metavar="PATH", help="ingest files/dir into the KB")
    ap.add_argument("--collection", default=None, help="override the RAG collection name")
    ap.add_argument("--chunk-tokens", type=int, default=None, help="override ingestion chunk size")
    ap.add_argument("--overlap-tokens", type=int, default=None, help="override ingestion chunk overlap")
    ap.add_argument("--source-tag", default=None, help="tag ingested documents with a source label")
    ap.add_argument("--skip-existing", action="store_true", help="skip docs already present in the KB")
    ap.add_argument("--dry-run", action="store_true", help="scan and report ingestion without adding to the KB")
    ap.add_argument("--healthcheck", action="store_true")
    ap.add_argument("--kan-demo", action="store_true", help="Run a built-in Multifidelity KAN demo")
    ap.add_argument("--research", action="store_true",
                    help="Full staged research pipeline (literature + hypothesis + experiment + KG)")
    ap.add_argument("--auto-research", action="store_true",
                    help="Autonomous iterative research loop (SEARCH->GAP-DETECT->EXPAND->REPORT)")
    ap.add_argument("--quant-team", action="store_true",
                    help="Run the RAG/research/model/risk/IBKR multi-agent pipeline")
    ap.add_argument("--submit-paper", action="store_true",
                    help="Interactively approve submission to configured IBKR paper TWS")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Number of iterations for --auto-research (default: from config)")
    ap.add_argument("--n-papers", type=int, default=None,
                    help="Number of arXiv papers to acquire in research mode")
    ap.add_argument("--no-kg", action="store_true",
                    help="Disable knowledge-graph update in research mode")
    ap.add_argument("--kg-summary", action="store_true",
                    help="Print a summary of the knowledge graph and exit")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    cfg = load_config(args.config)

    import logging
    Path(args.out).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(Path(args.out) / "agent.log")),
        ],
    )

    if args.healthcheck:
        sys.exit(healthcheck(cfg))

    if args.ingest:
        from rag.hybrid import LiteHybridRAG
        from rag.ingest import ingest_path

        rag = LiteHybridRAG(
            db_path=cfg["rag"]["db_path"],
            collection=args.collection or cfg["rag"].get("collection", "main"),
            embedding_model=cfg["rag"]["embedding_model"],
            alpha_dense=cfg["rag"]["alpha_dense"],
            query_expansion_enabled=cfg["rag"]["query_expansion"]["enabled"],
            query_expansion_method=cfg["rag"]["query_expansion"]["method"],
            max_expansions=cfg["rag"]["query_expansion"]["max_expansions"],
            reranking_enabled=cfg["rag"]["reranking"]["enabled"],
            reranking_model=cfg["rag"]["reranking"]["model"],
            top_k_before_rerank=cfg["rag"]["reranking"]["top_k_before_rerank"],
            top_k_after_rerank=cfg["rag"]["reranking"]["top_k_after_rerank"],
        )
        ingest_cfg = {
            "chunk_tokens": args.chunk_tokens if args.chunk_tokens is not None else cfg["rag"]["chunk_tokens"],
            "overlap": args.overlap_tokens if args.overlap_tokens is not None else cfg["rag"]["chunk_overlap"],
            "source_tag": args.source_tag,
            "skip_existing": args.skip_existing or cfg["rag"].get("skip_existing", False),
            "dry_run": args.dry_run,
        }
        counts = ingest_path(args.ingest, rag, **ingest_cfg)
        added = counts["added"]
        skipped = counts["skipped"]
        total = counts["total"]
        if args.dry_run:
            print(f"Dry run complete. {added} chunks would be added, {skipped} skipped, {total} seen.")
        else:
            print(f"Done. Collection now has {len(rag)} chunks (+{added} new; {skipped} duplicates skipped).")
        return

    if args.kan_demo:
        demo = run_kan_demo()
        print("\n>> Multifidelity KAN demo results")
        print(f"Samples: {demo['n_samples']}")
        print(f"Low features: {demo['low_features']}, high features: {demo['high_features']}")
        print("Baseline metrics:")
        for k, v in demo["baseline_metrics"].items():
            print(f"  {k}: {v:.6f}")
        print("KAN metrics:")
        for k, v in demo["kan_metrics"].items():
            print(f"  {k}: {v:.6f}")
        return

    if args.kg_summary:
        try:
            from kg.graph import ResearchKnowledgeGraph
            kg = ResearchKnowledgeGraph()
            print(kg.summarize())
            papers = kg.find_by_type("paper")
            if papers:
                print(f"\nPapers ({len(papers)}):")
                for p in papers[:10]:
                    print(f"  [{p.get('year','')}] {p.get('title','')[:70]}")
            findings = kg.find_by_type("finding")
            if findings:
                print(f"\nFindings ({len(findings)}):")
                for f in findings[:5]:
                    print(f"  [{f.get('source_task_id','')[:8]}] {f.get('text','')[:80]}...")
        except Exception as e:
            print(f"Knowledge graph unavailable: {e}", file=sys.stderr)
        return

    if not args.task:
        ap.print_help()
        return

    from agents.graph import run, research_run
    from rag.hybrid import LiteHybridRAG

    # Build services
    llm_cfg = make_llm_config(cfg)
    llm = HostedLLM(llm_cfg)
    if not llm.health():
        print("ERROR: OpenAI is not configured or reachable. Run `python run.py --healthcheck`.", file=sys.stderr)
        sys.exit(1)

    rag = LiteHybridRAG(
        db_path=cfg["rag"]["db_path"],
        collection=cfg["rag"].get("collection", "openai-embed-v1"),
        embedding_model=cfg["rag"]["embedding_model"],
        alpha_dense=cfg["rag"]["alpha_dense"],
        query_expansion_enabled=cfg["rag"]["query_expansion"]["enabled"],
        query_expansion_method=cfg["rag"]["query_expansion"]["method"],
        max_expansions=cfg["rag"]["query_expansion"]["max_expansions"],
        reranking_enabled=cfg["rag"]["reranking"]["enabled"],
        reranking_model=cfg["rag"]["reranking"]["model"],
        top_k_before_rerank=cfg["rag"]["reranking"]["top_k_before_rerank"],
        top_k_after_rerank=cfg["rag"]["reranking"]["top_k_after_rerank"],
    )

    tools = make_tools(cfg)
    max_iter = args.max_iter or cfg["agent"]["max_iterations"]

    if args.quant_team:
        from agents.quant_factory import create_quant_team

        team = create_quant_team(cfg, llm, rag, tools["backtest"])
        team_run = team.run(args.task)
        safe_payload = team_run.model_dump(mode="json")
        if safe_payload.get("execution_risk"):
            safe_payload["execution_risk"]["approval_token"] = None
        quant_dir = Path(args.out) / "quant_team"
        quant_dir.mkdir(parents=True, exist_ok=True)
        quant_path = quant_dir / f"{team_run.run_id}.json"
        quant_path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
        print(json.dumps(safe_payload, indent=2))
        print(f"\n[saved] {quant_path}")

        if args.submit_paper:
            risk = team_run.execution_risk
            if not risk or not risk.approved or not team_run.trade_intent:
                print("Order was not eligible for IBKR submission.", file=sys.stderr)
                sys.exit(2)
            approval_phrase = (
                f"APPROVE {team_run.trade_intent.symbol} "
                f"{team_run.trade_intent.action} {team_run.trade_intent.quantity}"
            )
            confirmation = input(
                f"Type {approval_phrase} to submit this exact paper order: "
            ).strip()
            if confirmation != approval_phrase:
                print("Order cancelled; confirmation did not match.")
                return
            receipt = team.execute(team_run, risk.approval_token or "")
            print(json.dumps(receipt.model_dump(mode="json"), indent=2))
        return

    if args.auto_research:
        from tools.research_loop import autonomous_research_loop
        from tools.citation_dag import CitationDAG
        loop_cfg = cfg.get("autonomous_loop", {})
        n_iter = args.iterations or loop_cfg.get("n_iterations", 3)
        n_papers_per_iter = args.n_papers or loop_cfg.get("n_papers_per_iter", 6)
        print(f"\n>> Autonomous research: {args.task}")
        print(f"   iterations={n_iter}  papers_per_iter={n_papers_per_iter}\n")
        dag = CitationDAG()
        result = autonomous_research_loop(
            topic=args.task,
            llm=llm,
            rag=rag,
            citation_dag=dag,
            n_iterations=n_iter,
            n_papers_per_iter=n_papers_per_iter,
            quality_threshold=loop_cfg.get("quality_threshold", 0.4),
        )
        print("=" * 60)
        print(f"Topic     : {result.topic}")
        print(f"Task ID   : {result.task_id}")
        print(f"Iterations: {len(result.iterations)}")
        print(f"Papers    : {result.total_papers_ingested}")
        print(f"DAG nodes : {result.citation_dag_nodes}")
        for it in result.iterations:
            print(f"  iter={it.iteration}  quality={it.quality_score:.3f}  "
                  f"new_papers={it.papers_found}  gaps={it.gaps_detected}")
        if result.final_report:
            report_path = (Path(args.out) / "tasks" / result.task_id / "research_report.md")
            print(f"\nReport    : {report_path}")
        return

    if args.research:
        research_cfg = cfg.get("research", {})
        n_papers = args.n_papers or research_cfg.get("n_papers", 8)
        kg_enabled = not args.no_kg and research_cfg.get("kg_enabled", True)
        print(f"\n>> Research task: {args.task}\n")
        state = research_run(
            args.task, llm, rag,
            max_iter=max_iter, tools=tools,
            n_papers=n_papers, kg_enabled=kg_enabled,
        )
    else:
        print(f"\n>> Task: {args.task}\n")
        state = run(args.task, llm, rag, max_iter=max_iter, tools=tools)

    # Persist + print summary
    out_dir = Path(args.out) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(out_dir.glob("run_*.json")))
    run_file = out_dir / f"run_{idx:04d}.json"
    run_file.write_text(
        json.dumps(asdict(state), indent=2, default=str), encoding="utf-8",
    )

    print("=" * 60)
    print(f"Task type : {state.task_type}")
    print(f"Subtasks  : {state.subtasks}")
    print(f"Iterations: {state.iterations}   Accepted: {state.accepted}")
    if state.artifacts:
        last = state.artifacts[-1]
        print(f"\nArtifact type: {last['type']}")
        if last["type"] == "ds":
            pl = last["payload"]
            print("stdout tail:\n" + (pl.get("stdout", "") or "")[-600:])
            if pl.get("stderr"):
                print("stderr tail:\n" + pl["stderr"][-400:])
        elif last["type"] == "quant":
            print(json.dumps(last["payload"], indent=2, default=str)[:1500])
        elif last["type"] == "writing":
            print(f"TeX : {len(last['payload']['tex'])} chars")
            print(f"PDF : {last['payload'].get('pdf')}")
    print(f"\n[saved] {run_file}")


if __name__ == "__main__":
    main()
