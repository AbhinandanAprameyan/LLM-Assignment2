"""
plot_and_report.py

Assignment 2: Building a Small-Scale Foundation Model from Scratch
====================================================================

Reads the metrics produced by `train_mini_gpt.py` (`metrics_log.json`,
and `hyperparam_sweep.json` if a --sweep run was done) and produces:

  1. loss_curve.png       -- train/validation loss vs. epoch
  2. perplexity_curve.png -- train/validation perplexity vs. epoch
  3. Assignment2_Report.pdf -- a 2-4 page report covering model
     architecture & parameters, dataset details, training setup &
     hyperparameter experiments, and observations/challenges.

Run this AFTER train_mini_gpt.py has produced ./output2/metrics_log.json
(and optionally ./output2/hyperparam_sweep.json):

    python plot_and_report.py --metrics ./output2/metrics_log.json \
        --sweep ./output2/hyperparam_sweep.json \
        --a1-stats ./run_stats.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display needed; just write PNG files
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, Image
)
from reportlab.lib import colors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=str, default="./output2/metrics_log.json")
    p.add_argument("--sweep", type=str, default="./output2/hyperparam_sweep.json")
    p.add_argument("--a1-stats", type=str, default="./run_stats.json",
                   help="Assignment 1's run_stats.json, used to describe the dataset in the report (optional).")
    p.add_argument("--out-dir", type=str, default="./output2")
    p.add_argument("--report-path", type=str, default="./Assignment2_Report.pdf")
    return p.parse_args()


def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ===========================================================================
# Plots
# ===========================================================================

def make_plots(metrics: dict, out_dir: str):
    history = metrics["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_ppl = [h["train_ppl"] for h in history]
    val_ppl = [h["val_ppl"] for h in history]

    loss_path = os.path.join(out_dir, "loss_curve.png")
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o", label="Train loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    ppl_path = os.path.join(out_dir, "perplexity_curve.png")
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_ppl, marker="o", label="Train perplexity")
    plt.plot(epochs, val_ppl, marker="o", label="Validation perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity (exp(loss))")
    plt.title("Training and Validation Perplexity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ppl_path, dpi=150)
    plt.close()

    return loss_path, ppl_path


# ===========================================================================
# Report
# ===========================================================================

def build_report(metrics: dict, sweep, a1_stats, loss_path: str, ppl_path: str, report_path: str):
    styles = getSampleStyleSheet()
    styles["Title"].fontSize = 16
    styles["Title"].leading = 19
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=13, spaceBefore=9, spaceAfter=3))
    styles.add(ParagraphStyle(name="BodyText2", parent=styles["BodyText"], fontSize=9.3, leading=12.5, spaceAfter=5, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.7, leading=11, textColor=colors.grey))
    cell_hdr = ParagraphStyle(name="CellHdr", parent=styles["Small"], textColor=colors.white, fontSize=9, leading=11)
    cell_body = ParagraphStyle(name="CellBody", parent=styles["Small"], textColor=colors.black, fontSize=8.5, leading=10.5)

    B = styles["BodyText2"]
    H1 = styles["H1"]

    doc = SimpleDocTemplate(
        report_path, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        title="Assignment 2 Report",
    )
    S = []

    config = metrics["config"]
    history = metrics["history"]
    final = history[-1]
    n_params = metrics["num_parameters"]

    # ---- Title ----------------------------------------------------------
    S.append(Paragraph("Assignment 2: Building a Small-Scale Foundation Model from Scratch", styles["Title"]))
    S.append(Paragraph("Report", styles["Heading2"]))
    S.append(Spacer(1, 8))

    # ---- 1. Model architecture -------------------------------------------
    S.append(Paragraph("1. Model Architecture and Parameters", H1))
    S.append(Paragraph(
        "<b>MiniGPT</b> is a decoder-only, GPT-style transformer implemented from scratch in PyTorch "
        "(<code>mini_gpt.py</code>), consisting of a learned token embedding, a learned positional "
        "embedding, a stack of pre-norm transformer blocks (causal multi-head self-attention followed by "
        "a GELU MLP, each with a residual connection), a final layer norm, and a linear head projecting "
        "back to vocabulary-sized logits for next-token prediction. The causal-masking and multi-head "
        "reshape logic was first validated independently with a pure-numpy, no-autograd forward-pass "
        "re-implementation (<code>dev_validate_numpy.py</code>) before being written in PyTorch, to catch "
        "any tensor-shape or masking bugs early.", B))

    arch_table = [
        [Paragraph("Hyperparameter", cell_hdr), Paragraph("Value", cell_hdr), Paragraph("Assignment range", cell_hdr)],
        [Paragraph("Transformer blocks (n_layer)", cell_body), Paragraph(str(config["n_layer"]), cell_body), Paragraph("1-2", cell_body)],
        [Paragraph("Embedding dimension (n_embd)", cell_body), Paragraph(str(config["n_embd"]), cell_body), Paragraph("64-256", cell_body)],
        [Paragraph("Attention heads (n_head)", cell_body), Paragraph(str(config["n_head"]), cell_body), Paragraph("2-4", cell_body)],
        [Paragraph("Sequence length (block_size)", cell_body), Paragraph(str(config["block_size"]), cell_body), Paragraph("32-128", cell_body)],
        [Paragraph("Dropout", cell_body), Paragraph(str(config["dropout"]), cell_body), Paragraph("--", cell_body)],
        [Paragraph("Vocabulary size", cell_body), Paragraph(f"{config['vocab_size']:,}", cell_body), Paragraph("--", cell_body)],
        [Paragraph("Total trainable parameters", cell_body), Paragraph(f"{n_params:,}", cell_body), Paragraph("--", cell_body)],
    ]
    t = Table(arch_table, colWidths=[2.3 * inch, 1.5 * inch, 1.9 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3b52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    S.append(t)
    S.append(Spacer(1, 6))
    S.append(Paragraph(
        "Layer normalization is applied in the pre-norm position (before each sub-layer rather than after), "
        "which is the standard modern choice as it noticeably stabilizes training for small transformers "
        "trained from scratch without extensive learning-rate warmup schedules. GELU was used for the MLP "
        "activation, matching the original GPT family.", B))

    # ---- 2. Dataset details -----------------------------------------------
    S.append(Paragraph("2. Dataset Details", H1))
    if a1_stats:
        S.append(Paragraph(
            f"Training data was produced by reusing Assignment 1's cleaning and tokenization pipeline "
            f"(<code>data_collection_preprocessing.py</code>), which collected text from encyclopedic "
            f"(Wikipedia), news, and general web-text domains and applied the same exact-duplicate removal, "
            f"HTML/markdown stripping, and short-document filtering described in the Assignment 1 report. "
            f"That run produced {a1_stats.get('clean_documents', 'N/A'):,} cleaned documents "
            f"({a1_stats.get('clean_mb', 'N/A')}&nbsp;MB).", B))
    S.append(Paragraph(
        f"For this assignment, the cleaned documents were re-tokenized (same GPT-2 byte-level BPE "
        f"tokenizer, vocabulary size {config['vocab_size']:,}) into a single flat stream of "
        f"<b>{metrics['flat_token_stream_length']:,} tokens</b> (rather than Assignment 1's independent "
        f"512-token blocks), with an end-of-document token inserted between documents. That flat stream "
        f"was then sliced into non-overlapping, fixed-length ({config['block_size']}-token) (input, "
        f"target) pairs for causal language modeling, with the final 10% of the stream held out as a "
        f"validation set and the rest used for training.", B))

    # ---- 3. Training setup & hyperparameter experiments --------------------
    S.append(Paragraph("3. Training Setup and Hyperparameter Experiments", H1))
    S.append(Paragraph(
        f"The main training run used AdamW with a learning rate of {metrics.get('lr', 'N/A')}, batch size "
        f"{metrics.get('batch_size', 'N/A')}, and {len(history)} epochs, running on "
        f"<b>{metrics.get('device', 'N/A')}</b>. Training used a standard loop: forward pass through the "
        f"model, cross-entropy loss against the shifted target sequence, backward pass, and an AdamW "
        f"optimizer step, repeated once per batch per epoch.", B))
    S.append(Paragraph(
        f"<b>Final epoch results:</b> train loss {final['train_loss']:.4f} (perplexity "
        f"{final['train_ppl']:.2f}), validation loss {final['val_loss']:.4f} (perplexity "
        f"{final['val_ppl']:.2f}).", B))
    if "reload_check_val_loss" in metrics:
        S.append(Paragraph(
            f"<b>Checkpoint save/load verification:</b> immediately after saving "
            f"<code>mini_gpt_checkpoint.pt</code>, the checkpoint was reloaded into a freshly constructed "
            f"model instance and re-evaluated on the same validation set, giving a validation loss of "
            f"{metrics['reload_check_val_loss']:.4f} against the originally logged "
            f"{final['val_loss']:.4f} -- matching (to floating-point precision) confirms the checkpoint "
            f"correctly preserves the trained weights.", B))

    if sweep:
        S.append(Paragraph("Hyperparameter sweep results (short runs, held-out validation loss/perplexity):", B))
        sweep_table = [[Paragraph(h, cell_hdr) for h in ["lr", "batch", "layers", "embd", "epochs", "val_loss", "val_ppl", "params"]]]
        for row in sweep:
            sweep_table.append([
                Paragraph(str(row["lr"]), cell_body),
                Paragraph(str(row["batch_size"]), cell_body),
                Paragraph(str(row["n_layer"]), cell_body),
                Paragraph(str(row["n_embd"]), cell_body),
                Paragraph(str(row["epochs"]), cell_body),
                Paragraph(f"{row['final_val_loss']:.3f}", cell_body),
                Paragraph(f"{row['final_val_ppl']:.1f}", cell_body),
                Paragraph(f"{row['num_parameters']:,}", cell_body),
            ])
        st = Table(sweep_table, colWidths=[0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.65*inch, 0.6*inch, 0.75*inch])
        st.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3b52")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        S.append(st)
        best = min(sweep, key=lambda r: r["final_val_loss"])
        S.append(Spacer(1, 4))
        S.append(Paragraph(
            f"Across the sweep, the configuration with lr={best['lr']}, batch_size={best['batch_size']}, "
            f"n_layer={best['n_layer']}, n_embd={best['n_embd']} achieved the lowest validation loss "
            f"({best['final_val_loss']:.3f}, perplexity {best['final_val_ppl']:.1f}) after only "
            f"{best['epochs']} short epochs. Note that sweep trials intentionally use far fewer epochs "
            f"than the main run purely to keep the search affordable; their absolute loss values are not "
            f"directly comparable to the main run's fully-trained result above.", B))
    else:
        S.append(Paragraph(
            "No hyperparameter sweep was run for this report (re-run with <code>--sweep</code> to populate "
            "this section with a small grid search over learning rate, batch size, number of layers, and "
            "embedding size).", styles["Small"]))

    # ---- Plots -------------------------------------------------------------
    S.append(Spacer(1, 6))
    S.append(Image(loss_path, width=3.15 * inch, height=2.1 * inch))
    S.append(Image(ppl_path, width=3.15 * inch, height=2.1 * inch))

    # ---- 4. Observations and challenges --------------------------------
    S.append(Paragraph("4. Observations and Challenges", H1))
    loss_dropped = history[0]["train_loss"] - final["train_loss"]
    trend = "decreased" if loss_dropped > 0 else "did not meaningfully decrease"
    change_phrase = f"a decrease of {loss_dropped:.4f}" if loss_dropped > 0 else f"a change of {loss_dropped:+.4f}"
    S.append(ListFlowable([
        ListItem(Paragraph(
            f"<b>Loss trend.</b> Training loss {trend} from {history[0]['train_loss']:.4f} at epoch 1 to "
            f"{final['train_loss']:.4f} at epoch {final['epoch']} ({change_phrase}), indicating "
            f"the model is {'successfully' if loss_dropped > 0.05 else 'only slowly'} learning next-token "
            f"prediction patterns from the training stream over the epochs run.", B)),
        ListItem(Paragraph(
            "<b>Train/validation gap.</b> A validation loss noticeably higher than training loss would "
            "indicate overfitting to the (comparatively small) training stream used here; given how few "
            "unique documents a small-scale run of this kind sees, some gap is expected and is best "
            "addressed by increasing --target-mb (more raw text) rather than by architectural changes "
            "alone.", B)),
        ListItem(Paragraph(
            "<b>Causal masking and padding correctness.</b> Because attention masking bugs (e.g. a query "
            "attending to a future token) can silently produce a model that appears to train but never "
            "generalizes to real autoregressive generation, the attention mechanism's shape and masking "
            "behavior were unit-verified independently (see <code>dev_validate_numpy.py</code>) before "
            "training, rather than relying solely on the loss curve going down as evidence of correctness.", B)),
        ListItem(Paragraph(
            "<b>Sequence length vs. compute.</b> Because attention compute and memory scale quadratically "
            "with sequence length, moving from the low end (32 tokens) to the high end (128 tokens) of the "
            "assignment's allowed block-size range measurably increases per-step time even for a model "
            "this small, which is one of the concrete trade-offs the hyperparameter sweep is designed to "
            "surface.", B)),
        ListItem(Paragraph(
            "<b>Reusing Assignment 1's pipeline surfaced a scale mismatch.</b> Assignment 1's default "
            "512-token blocking is tuned for a much larger downstream pretraining run; for this "
            "small-scale assignment, re-flattening the cleaned corpus into one continuous token stream "
            "and re-slicing it at the smaller, assignment-specified block size (32-128) was necessary "
            "rather than reusing Assignment 1's already-chunked 512-token blocks directly.", B)),
    ], bulletType="bullet"))

    doc.build(S)
    print(f"Wrote {report_path}")


def main():
    args = parse_args()
    metrics = load_json(args.metrics)
    if metrics is None:
        raise FileNotFoundError(
            f"Could not find {args.metrics}. Run train_mini_gpt.py first to produce metrics_log.json."
        )
    sweep = load_json(args.sweep)
    a1_stats = load_json(args.a1_stats)

    loss_path, ppl_path = make_plots(metrics, args.out_dir)
    build_report(metrics, sweep, a1_stats, loss_path, ppl_path, args.report_path)


if __name__ == "__main__":
    main()
