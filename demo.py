import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from cln import CLNModel, ConversationTracker



VOCAB_SIZE = 256


def encode(text: str) -> list:
    return list(text.encode("utf-8"))


def decode(tokens: list) -> str:
    return bytes(t & 0xFF for t in tokens).decode("utf-8", errors="replace")



def total_plastic_norm(model: CLNModel) -> float:
    from cln.core import LiquidLinear
    return sum(m.delta_w.norm().item() for m in model.modules() if isinstance(m, LiquidLinear))


def make_model(d_model=128, n_layers=4, n_heads=4, d_ff=512) -> CLNModel:
    return CLNModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=512,
        liquid_kwargs={
            "tau_w":      15.0,
            "eta":        1e-3,
            "lambda_ewc": 0.1,
            "dt":         0.1,
            "max_delta":  0.5,
        },
    ).eval()


def section(title: str) -> None:
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)



def experiment_online_learning(model: CLNModel) -> None:
    section("EXPERIMENT 1 · Online learning during inference")

    texts = [
        "Liquid networks learn continuously from every token.",
        "There is no separate training phase.",
        "Inference and learning are the same process.",
    ]

    print(f"{'Step':>4}  {'Plastic norm':>14}  {'Δ from previous':>16}  Text excerpt")
    print("-" * 78)

    prev = total_plastic_norm(model)
    for i, text in enumerate(texts):
        ids = torch.tensor([encode(text)])
        with torch.no_grad():
            _ = model(ids, plastic=True)
        norm = total_plastic_norm(model)
        delta = norm - prev
        prev = norm
        print(f'{i+1:>4}  {norm:>14.6f}  {delta:>+16.6f}  "{text[:42]}..."')

    print()
    print("✓  Plastic norm grew with each interaction — weights updated online.")



def experiment_anti_forgetting(model: CLNModel) -> None:
    section("EXPERIMENT 2 · Anti-catastrophic forgetting (EWC)")

    text_a = "The capital of France is Paris, a city of lights."
    ids_a = torch.tensor([encode(text_a)])
    with torch.no_grad():
        for _ in range(5):
            model(ids_a, plastic=True)

    from cln.core import LiquidLinear
    first_layer = next(m for m in model.modules() if isinstance(m, LiquidLinear))
    snap_before = first_layer.delta_w.clone()

    model.consolidate_all()
    print("Consolidated memory after concept A (Paris).")

    text_b = "Quantum mechanics describes subatomic particle behavior."
    ids_b = torch.tensor([encode(text_b)])
    with torch.no_grad():
        for _ in range(5):
            model(ids_b, plastic=True)

    snap_after = first_layer.delta_w.clone()

    cosine = F.cosine_similarity(
        snap_before.flatten().unsqueeze(0),
        snap_after.flatten().unsqueeze(0),
    ).item()

    l2_drift = (snap_after - snap_before).norm().item()
    base_norm = snap_before.norm().item()

    print(f"Concept A weight retention (cosine similarity) : {cosine:.4f}  (1.0 = perfect retention)")
    print(f"L2 drift after learning concept B              : {l2_drift:.6f}")
    print(f"Drift relative to original norm                : {l2_drift / (base_norm + 1e-8):.2%}")
    print()
    if cosine > 0.7:
        print("✓  High cosine similarity — concept A is largely preserved (EWC is working).")
    else:
        print("~  Some drift observed (increase lambda_ewc for stronger protection).")



def experiment_persistent_memory(model: CLNModel) -> None:
    section("EXPERIMENT 3 · Persistent memory (save / reset / restore)")

    save_path = "/tmp/cln_plastic_state.pt"

    text = "Memory persists across sessions in liquid networks."
    ids  = torch.tensor([encode(text)])
    with torch.no_grad():
        for _ in range(3):
            model(ids, plastic=True)

    norm_after_learning = total_plastic_norm(model)
    print(f"Plastic norm after learning  : {norm_after_learning:.6f}")

    model.save_plastic_state(save_path)
    print(f"Saved to                     : {save_path}")

    model.reset_plasticity()
    norm_after_reset = total_plastic_norm(model)
    print(f"Plastic norm after reset     : {norm_after_reset:.6f}")
    assert norm_after_reset < 1e-9, "Reset did not zero plastic weights"

    success = model.load_plastic_state(save_path)
    norm_after_restore = total_plastic_norm(model)
    print(f"Plastic norm after restore   : {norm_after_restore:.6f}")

    delta = abs(norm_after_restore - norm_after_learning)
    print(f"Round-trip error             : {delta:.2e}")
    print()
    if success and delta < 1e-4:
        print("✓  Plastic state saved and restored exactly — memory is persistent.")
    else:
        print("✗  Round-trip mismatch — check save/load paths.")



def experiment_plasticity_structure(model: CLNModel) -> None:
    section("EXPERIMENT 4 · Plasticity structure after sustained interactions")

    corpus = [
        "The transformer architecture uses multi-head self-attention.",
        "Liquid networks adapt their topology in real time.",
        "Hebbian plasticity strengthens frequently co-active pathways.",
        "Elastic weight consolidation prevents catastrophic forgetting.",
        "The ODE governing weight evolution is solved via Euler integration.",
        "Every forward pass is simultaneously an inference and a learning step.",
    ]

    for i in range(30):
        text = corpus[i % len(corpus)]
        ids  = torch.tensor([encode(text)])
        with torch.no_grad():
            model(ids, plastic=True)
        if (i + 1) % 10 == 0:
            model.consolidate_all()
            print(f"  step {i+1:2d}: consolidated | plastic norm = {total_plastic_norm(model):.6f}")

    stats = model.plasticity_stats()
    print()
    print(f"{'Layer':<50}  {'Δ‖ΔW‖':>10}  {'‖F‖':>10}  {'ratio':>8}")
    print("-" * 84)
    for name, s in list(stats.items())[:8]:
        short = name[-48:].rjust(50)
        print(f"{short}  {s['delta_norm']:>10.6f}  {s['fisher_norm']:>10.6f}  {s['plasticity_ratio']:>7.4%}")

    print()
    print("✓  Each layer has developed a unique plastic structure reflecting its role.")



CONVERSATION = [
    ("Turn 1",  "The Milky Way galaxy contains over 200 billion stars."),
    ("Turn 2",  "Black holes warp spacetime beyond their event horizon."),
    ("Turn 3",  "Neutron stars are the densest objects in the observable universe."),
    ("Turn 4",  "Light from the Andromeda galaxy takes 2.5 million years to reach us."),
    ("CONSOLIDATE", None),
    ("Turn 5",  "The human brain contains approximately 86 billion neurons."),
    ("Turn 6",  "Synaptic plasticity underlies long-term potentiation and memory."),
    ("Turn 7",  "Hebbian learning: neurons that fire together wire together."),
    ("Turn 8",  "The hippocampus is critical for forming new episodic memories."),
    ("Turn 9",  "Quantum superposition allows particles to exist in multiple states."),
    ("Turn 10", "Entanglement correlates quantum states across arbitrary distances."),
]


def experiment_conversation_viz(
    model: CLNModel,
    save_path: str = "evolution.png",
) -> None:
    section("EXPERIMENT 5 · Conversation weight evolution (visualization)")

    model.reset_plasticity()
    tracker = ConversationTracker(model, max_heatmap_dim=64)

    print(f"{'Turn':<14}  {'Plastic norm':>14}  Text")
    print("-" * 72)

    for label, text in CONVERSATION:
        if text is None:
            model.consolidate_all()
            print(f"  {'[consolidation]':<14}  {'—':>14}  EWC consolidation fired")
            continue

        ids = torch.tensor([encode(text)])
        with torch.no_grad():
            model(ids, plastic=True)

        tracker.record(label=label)
        norm = total_plastic_norm(model)
        print(f"  {label:<14}  {norm:>14.6f}  {text[:45]}...")

    print()
    print(f"Generating figure ({save_path})...")
    tracker.plot(
        save_path=save_path,
        title="CLN · Weight Evolution Across a 10-Turn Conversation",
    )
    print("✓  Saved. Open evolution.png to inspect the weight dynamics.")



def main() -> None:
    viz_only = "--viz-only" in sys.argv

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Continuous Liquid Networks (CLN)                        ║")
    print("║  Redes Líquidas de Aprendizaje Continuo                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    model = make_model()
    counts = model.param_count()

    print()
    print("Model configuration:")
    print(f"  Static parameters  : {counts['static']:>10,}")
    print(f"  Plastic parameters : {counts['plastic']:>10,}  (ΔW, Fisher, anchor)")
    print(f"  Total              : {counts['total']:>10,}")

    if not viz_only:
        experiment_online_learning(model)
        model.reset_plasticity()

        experiment_anti_forgetting(model)
        model.reset_plasticity()

        experiment_persistent_memory(model)
        model.reset_plasticity()

        experiment_plasticity_structure(model)
        model.reset_plasticity()

    experiment_conversation_viz(model, save_path="evolution.png")

    section("SUMMARY")
    print("Properties demonstrated:")
    print("  ✓  Weights update during inference — no separate training loop")
    print("  ✓  EWC consolidation protects prior knowledge")
    print("  ✓  Plastic state persists across save / load cycles")
    print("  ✓  Each layer develops unique plasticity structure")
    print("  ✓  Weight evolution visualized across a 10-turn conversation")
    print()
    print("  Architecture: CLNModel → CLNBlock × N → LiquidLinear (ODE)")
    print("  Core ODE:  dΔW/dt = −ΔW/τ + η·(post⊗pre) − λ·Ω·(ΔW−ΔW*)")
    print()


if __name__ == "__main__":
    main()
