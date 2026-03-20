#!/usr/bin/env python3
"""
One-click full pipeline CLI: crawl -> preprocess -> generate data -> train SFT -> train DPO -> evaluate.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import click
import logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


@click.group()
def cli():
    """E-commerce RAG Assistant - Full Pipeline CLI"""
    pass


@cli.command()
@click.option("--products-per-category", default=12, type=int, help="每个分类生成的商品数")
@click.option("--order-count", default=240, type=int, help="生成的订单数")
@click.option("--seed", default=42, type=int, help="随机种子")
def bootstrap_demo(products_per_category, order_count, seed):
    """Step 0: 生成稳定可复现的合成演示数据。"""
    console.print(Panel("Step 0: Bootstrapping Synthetic Demo Data", style="bold cyan"))

    from src.config import Settings
    from src.data import ensure_runtime_data

    settings = Settings()
    result = ensure_runtime_data(
        settings=settings,
        force=True,
        products_per_category=products_per_category,
        order_count=order_count,
        seed=seed,
    )
    summary = result.get("summary", {})
    console.print(
        f"[green]Synthetic demo data ready:[/green] "
        f"{summary.get('product_count', 0)} products / {summary.get('order_count', 0)} orders"
    )


@cli.command()
@click.option("--source", default="jd", type=click.Choice(["jd", "taobao", "all"]))
@click.option("--max-per-category", default=50, type=int)
def crawl(source, max_per_category):
    """Step 1: Crawl product data."""
    console.print(Panel("Step 1: Crawling Product Data", style="bold blue"))

    from scripts.crawl_data import main as crawl_main
    sys.argv = ["crawl_data.py", "--source", source, "--max-per-category", str(max_per_category)]
    crawl_main()


@cli.command()
@click.option("--model", default="BAAI/bge-base-zh-v1.5", help="Embedding model")
def preprocess(model):
    """Step 2: Preprocess data and create embeddings."""
    console.print(Panel("Step 2: Preprocessing Data", style="bold blue"))

    from scripts.preprocess_data import main as preprocess_main
    preprocess_main()


@cli.command()
@click.option("--sft-count", default=3000, type=int, help="Target SFT sample count")
@click.option("--dpo-count", default=1000, type=int, help="Target DPO pair count")
@click.option("--seed", default=42, type=int, help="随机种子")
def generate_data(sft_count, dpo_count, seed):
    """Step 3: Generate SFT and DPO training data."""
    console.print(Panel("Step 3: Generating Training Data", style="bold blue"))

    from src.config import Settings
    from src.data import ensure_runtime_data
    from training.data_gen.dpo_generator import DPODataGenerator
    from training.data_gen.quality_filter import QualityFilter
    from training.data_gen.sft_generator import SFTDataGenerator

    settings = Settings()
    ensure_runtime_data(settings=settings)

    sft_scale = sft_count / SFTDataGenerator.base_target_total()
    dpo_scale = dpo_count / DPODataGenerator.base_target_total()

    console.print("[bold]Generating SFT data...[/bold]")
    sft_gen = SFTDataGenerator(
        product_path=str(settings.PRODUCT_DATA_PATH),
        order_path=str(settings.ORDER_DATA_PATH),
        seed=seed,
    )
    raw_sft = sft_gen.generate_all(scale=sft_scale)

    console.print("[bold]Generating DPO data...[/bold]")
    dpo_gen = DPODataGenerator(
        product_path=str(settings.PRODUCT_DATA_PATH),
        order_path=str(settings.ORDER_DATA_PATH),
        seed=seed,
    )
    raw_dpo = dpo_gen.generate_all(scale=dpo_scale)

    console.print("[bold]Running quality filter...[/bold]")
    qf = QualityFilter()
    clean_sft = qf.filter_sft(raw_sft)
    clean_dpo = qf.filter_dpo(raw_dpo)

    sft_gen.samples = clean_sft
    dpo_gen.samples = clean_dpo
    sft_train_path, sft_val_path = sft_gen.save()
    dpo_train_path, dpo_val_path = dpo_gen.save()

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "sft": {
            "requested_count": sft_count,
            "scale": round(sft_scale, 4),
            "generated_before_filter": len(raw_sft),
            "generated_after_filter": len(clean_sft),
            "train_path": str(sft_train_path),
            "val_path": str(sft_val_path),
            "filter_stats": qf.sft_stats.rejection_summary(),
        },
        "dpo": {
            "requested_count": dpo_count,
            "scale": round(dpo_scale, 4),
            "generated_before_filter": len(raw_dpo),
            "generated_after_filter": len(clean_dpo),
            "train_path": str(dpo_train_path),
            "val_path": str(dpo_val_path),
            "filter_stats": qf.dpo_stats.rejection_summary(),
        },
    }
    manifest_path = settings.TRAINING_DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(
        "[green]Training data generation complete![/green]\n"
        f"SFT: {len(clean_sft)} samples -> {sft_train_path}\n"
        f"DPO: {len(clean_dpo)} pairs -> {dpo_train_path}\n"
        f"Manifest: {manifest_path}"
    )


@cli.command()
def train_sft():
    """Step 4: Run SFT training with QLoRA."""
    console.print(Panel("Step 4: SFT Training", style="bold blue"))

    from training.sft_train import train
    from training.config import SFTConfig

    cfg = SFTConfig()
    output_dir = train(cfg)
    console.print(f"[green]SFT model saved to: {output_dir}[/green]")


@cli.command()
def train_dpo():
    """Step 5: Run DPO training."""
    console.print(Panel("Step 5: DPO Training", style="bold blue"))

    from training.dpo_train import train
    from training.config import DPOConfig

    cfg = DPOConfig()
    output_dir = train(cfg)
    console.print(f"[green]DPO model saved to: {output_dir}[/green]")


@cli.command()
@click.option("--adapter", default="outputs/dpo", help="LoRA adapter path")
def merge(adapter):
    """Step 6: Merge LoRA adapter into base model."""
    console.print(Panel("Step 6: Merging LoRA Weights", style="bold blue"))

    from training.merge_lora import merge as merge_lora
    from training.config import MergeConfig

    cfg = MergeConfig(adapter_path=adapter)
    output_dir = merge_lora(cfg)
    console.print(f"[green]Merged model saved to: {output_dir}[/green]")


@cli.command()
@click.option("--stages", default="base,sft,dpo", help="Comma-separated model stages to evaluate")
@click.option("--output", default="outputs/model_showcase/benchmark_results.json", help="评测结果 JSON 路径")
@click.option("--profile", default="base_mining", type=click.Choice(["quick", "base_mining"]), help="benchmark 场景规模")
@click.option("--case-ids", default="", help="只跑指定 case_id，逗号分隔")
@click.option("--scenario-types", default="", help="只跑指定 scenario_type，逗号分隔")
@click.option("--judge-model", default="", help="用于 LLM 评审的本地模型路径")
@click.option("--disable-llm-judge", is_flag=True, help="关闭 LLM-as-a-Judge，仅使用规则分")
@click.option("--judge-weight", default=0.55, type=float, help="综合分中 LLM judge 的权重")
@click.option("--answer-max-new-tokens", default=320, type=int, help="模型回答的最大生成 token")
@click.option("--judge-max-new-tokens", default=320, type=int, help="judge 的最大生成 token")
def evaluate(
    stages,
    output,
    profile,
    case_ids,
    scenario_types,
    judge_model,
    disable_llm_judge,
    judge_weight,
    answer_max_new_tokens,
    judge_max_new_tokens,
):
    """Step 7: Run benchmark evaluation."""
    console.print(Panel("Step 7: Evaluation Benchmark", style="bold blue"))

    from evaluation.benchmark import Benchmark

    stage_list = [s.strip() for s in stages.split(",")]
    benchmark_kwargs = {
        "use_llm_judge": not disable_llm_judge,
        "judge_weight": judge_weight,
        "answer_max_new_tokens": answer_max_new_tokens,
        "judge_max_new_tokens": judge_max_new_tokens,
        "benchmark_profile": profile,
    }
    if case_ids.strip():
        benchmark_kwargs["selected_case_ids"] = [item.strip() for item in case_ids.split(",") if item.strip()]
    if scenario_types.strip():
        benchmark_kwargs["selected_scenario_types"] = [
            item.strip() for item in scenario_types.split(",") if item.strip()
        ]
    if judge_model.strip():
        benchmark_kwargs["judge_model"] = judge_model.strip()

    benchmark = Benchmark(**benchmark_kwargs)
    results = benchmark.run(stages=stage_list, output_path=output)

    # Print summary
    for stage, data in results.get("stages", {}).items():
        console.print(
            f"\n[bold]{stage}[/bold]: avg_score={data.get('avg_score', 0):.4f}, "
            f"latency={data.get('avg_latency', 0):.2f}s"
        )


@cli.command()
@click.option("--benchmark", default="outputs/model_showcase/benchmark_results.json", help="benchmark 结果 JSON")
@click.option("--output", default="outputs/model_showcase/data_plan.json", help="数据计划输出 JSON")
def build_data_plan(benchmark, output):
    """Step 8: Build a closed-loop data plan from benchmark failures."""
    console.print(Panel("Step 8: Building Data Plan", style="bold blue"))

    from scripts.build_data_plan import build_plan, save_plan

    plan = build_plan(benchmark)
    paths = save_plan(plan, output)
    summary = plan.get("summary", {})
    console.print(
        "[green]Data plan generated![/green]\n"
        f"SFT cases: {summary.get('sft_cases', 0)} / samples: {summary.get('total_sft_samples', 0)}\n"
        f"DPO cases: {summary.get('dpo_cases', 0)} / pairs: {summary.get('total_dpo_pairs', 0)}\n"
        f"JSON: {paths['json']}\n"
        f"Markdown: {paths['markdown']}"
    )


@cli.command()
@click.option("--sft-train", default="data/training/sft_train.jsonl", help="SFT train 数据路径")
@click.option("--sft-val", default="data/training/sft_val.jsonl", help="SFT val 数据路径")
@click.option("--dpo-train", default="data/training/dpo_train.jsonl", help="DPO train 数据路径")
@click.option("--dpo-val", default="data/training/dpo_val.jsonl", help="DPO val 数据路径")
@click.option("--output-dir", default="data/training/focused", help="focused 数据输出目录")
@click.option(
    "--profile",
    default="repair_v1",
    type=click.Choice(["repair_v1", "qwen25_3b_struct"]),
    help="focused 数据导出 profile",
)
@click.option(
    "--sft-slices",
    default="",
    help="SFT focused 切片，逗号分隔；留空则使用 profile 默认值",
)
@click.option(
    "--dpo-slices",
    default="",
    help="DPO focused 切片，逗号分隔；留空则使用 profile 默认值",
)
def export_focus_data(sft_train, sft_val, dpo_train, dpo_val, output_dir, profile, sft_slices, dpo_slices):
    """Step 9: Export focused training slices for targeted experiments."""
    console.print(Panel("Step 9: Exporting Focused Data", style="bold blue"))

    from scripts.export_focus_data import build_focus_export, save_focus_export
    from training.focus_slices import get_focus_profile

    profile_spec = get_focus_profile(profile)
    if output_dir == "data/training/focused" and profile != "repair_v1":
        output_dir = profile_spec["recommended_output_dir"]

    manifest = build_focus_export(
        sft_train_path=sft_train,
        sft_val_path=sft_val,
        dpo_train_path=dpo_train,
        dpo_val_path=dpo_val,
        output_dir=output_dir,
        focus_profile=profile,
        sft_slice_ids=[item.strip() for item in sft_slices.split(",") if item.strip()],
        dpo_slice_ids=[item.strip() for item in dpo_slices.split(",") if item.strip()],
    )
    paths = save_focus_export(manifest, output_dir)
    console.print(
        "[green]Focused data exported![/green]\n"
        f"Profile: {profile}\n"
        f"SFT train/val: {manifest['sft']['train_count']} / {manifest['sft']['val_count']}\n"
        f"DPO train/val: {manifest['dpo']['train_count']} / {manifest['dpo']['val_count']}\n"
        f"JSON: {paths['json']}\n"
        f"Markdown: {paths['markdown']}"
    )


@cli.command()
@click.option("--sft-train", default="data/training/sft_train.jsonl", help="SFT train 数据路径")
@click.option("--sft-val", default="data/training/sft_val.jsonl", help="SFT val 数据路径")
@click.option("--dpo-train", default="data/training/dpo_train.jsonl", help="DPO train 数据路径")
@click.option("--dpo-val", default="data/training/dpo_val.jsonl", help="DPO val 数据路径")
@click.option(
    "--plan",
    default="outputs/model_showcase/data_plan_base_qwen7b_judge_mined_v1.json",
    help="失败数据计划 JSON 路径",
)
@click.option(
    "--profile",
    default="repair_v1",
    type=click.Choice(["repair_v1", "qwen25_3b_struct"]),
    help="focused 摘要 profile",
)
@click.option("--output", default="outputs/model_showcase/p0_data_summary.json", help="摘要输出 JSON 路径")
@click.option("--preview-count", default=2, type=int, help="每个切片保留多少个样本预览")
def summarize_focus_data(sft_train, sft_val, dpo_train, dpo_val, plan, profile, output, preview_count):
    """Step 10: Summarize focus training slices for review."""
    console.print(Panel("Step 10: Summarizing Focus Data", style="bold blue"))

    from scripts.summarize_focus_data import build_summary, save_summary

    summary = build_summary(
        sft_train_path=sft_train,
        sft_val_path=sft_val,
        dpo_train_path=dpo_train,
        dpo_val_path=dpo_val,
        plan_path=plan,
        focus_profile=profile,
        preview_count=preview_count,
    )
    paths = save_summary(summary, output)
    totals = summary.get("totals", {})
    console.print(
        "[green]Focus data summary generated![/green]\n"
        f"Profile: {profile}\n"
        f"Focus SFT total: {totals.get('focus_sft_total', 0)}\n"
        f"Focus DPO total: {totals.get('focus_dpo_total', 0)}\n"
        f"JSON: {paths['json']}\n"
        f"Markdown: {paths['markdown']}"
    )


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def serve(host, port):
    """Start the API server."""
    console.print(Panel(f"Starting API Server on {host}:{port}", style="bold green"))

    import uvicorn
    from src.config import Settings
    from src.data import ensure_runtime_data

    ensure_runtime_data(settings=Settings())
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


@cli.command()
@click.pass_context
def all(ctx):
    """Run the complete pipeline (crawl -> preprocess -> generate -> train -> evaluate)."""
    console.print(Panel("Running Complete Pipeline", style="bold magenta"))

    steps = [
        ("crawl", {"source": "jd", "max_per_category": 50}),
        ("preprocess", {"model": "BAAI/bge-base-zh-v1.5"}),
        ("generate_data", {"sft_count": 3000, "dpo_count": 1000}),
        ("train_sft", {}),
        ("train_dpo", {}),
        ("merge", {"adapter": "outputs/dpo"}),
        ("evaluate", {"stages": "base,sft,dpo"}),
    ]

    for step_name, kwargs in steps:
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Running: {step_name}[/bold]")
        console.print(f"{'=' * 60}")
        try:
            ctx.invoke(globals()[step_name.replace("-", "_")], **kwargs)
        except Exception as e:
            console.print(f"[red]Error in {step_name}: {e}[/red]")
            if not click.confirm("Continue with next step?"):
                break

    console.print("\n[bold green]Pipeline complete![/bold green]")


if __name__ == "__main__":
    cli()
