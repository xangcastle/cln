"""
CLNChat — interactive chat interface for HuggingFace models with CLN plasticity.

HuggingFace backend
    Uses any causal HuggingFace model with plasticity injected by
    ``inject_plasticity()``. Loaded via ``load_hf()``.

    Generation/learning strategy:

    1. Fast inference with KV-cache and ``plastic=False`` — O(N + M).
    2. Deferred learning: one full forward pass over the complete context +
       response block with ``plastic=True`` after generation completes.
       Much more efficient than updating ΔW token-by-token with KV-cache.
"""

import sys
from pathlib import Path
from typing import List, Optional, Set

import torch
import torch.nn.functional as F

from .core import LiquidLinear, set_plastic_mode

HF_SYSTEM = (
    "You are a precise and helpful assistant. "
    "Read the conversation history carefully and use it to answer accurately. "
    "If the user explained something in a previous message, remember it and use it in your reply. "
    "Do not make up information. If you are not sure about something, say so clearly. "
    "Reply in the same language the user writes in."
)


def _sample(
    logits: torch.Tensor,
    generated: List[int],
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
) -> int:
    """Sample the next token id from a logit distribution.

    Applies repetition penalty, temperature scaling, top-k filtering, and
    nucleus (top-p) sampling in sequence.

    Args:
        logits: Raw logit vector of shape [vocab_size].
        generated: Token ids generated so far in the current response.
            Used to compute the repetition penalty over the last 64 tokens.
        temperature: Softmax temperature. Higher values increase randomness.
        top_k: Number of top-logit tokens to keep before nucleus sampling.
            Pass 0 to disable.
        top_p: Cumulative probability threshold for nucleus sampling.
        rep_penalty: Penalty factor applied to previously generated tokens.
            Values > 1.0 discourage repetition; 1.0 disables the penalty.

    Returns:
        The sampled token id as a plain Python ``int``.
    """
    logits = logits.clone().float()

    if rep_penalty != 1.0 and generated:
        for tid in set(generated[-64:]):
            logits[tid] = logits[tid] / rep_penalty if logits[tid] > 0 else logits[tid] * rep_penalty

    logits /= max(temperature, 1e-6)

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    s = sorted_probs.sum()
    sorted_probs = sorted_probs / s if s > 0 else sorted_probs * 0 + 1 / sorted_probs.size(0)

    return sorted_idx[torch.multinomial(sorted_probs, 1).item()].item()


class CLNChat:
    """Interactive chat interface for CLN models with HuggingFace backend.

    Example — HuggingFace backend::

        from cln.loader import load_hf
        model, tokenizer = load_hf("microsoft/Phi-3-mini-4k-instruct")
        CLNChat(model, tokenizer=tokenizer).run()

    Attributes:
        model: The underlying HuggingFace causal language model with injected
            CLN plasticity.
        memory_path: Path used to persist and restore plastic state between
            sessions.
        history: Ordered list of conversation turns, each a dict with
            ``role`` and ``content`` keys (OpenAI message format).
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        memory_path: str = "cln_memory.pt",
        max_context_tokens: int = 3000,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.92,
        rep_penalty: float = 1.15,
        plastic: bool = True,
        deferred_learning: bool = True,
    ):
        """Initialize CLNChat.

        Args:
            model: A HuggingFace causal language model with plasticity injected
                via ``inject_plasticity()``.
            tokenizer: HuggingFace tokenizer associated with ``model``.
            memory_path: File path for saving and loading plastic state. The
                file is written automatically after each chat turn when
                ``plastic=True``.
            max_context_tokens: Maximum number of tokens in the prompt context.
                Older history turns are dropped to stay within this limit.
            max_new_tokens: Maximum number of new tokens to generate per turn.
            temperature: Sampling temperature. Lower values produce more
                deterministic output.
            top_k: Top-k filtering threshold. Pass 0 to disable.
            top_p: Nucleus sampling cumulative probability threshold.
            rep_penalty: Repetition penalty factor. Values above 1.0 discourage
                the model from repeating recently generated tokens.
            plastic: When True, plastic weights are updated after each turn and
                the session state is saved to ``memory_path``.
            deferred_learning: When True, Hebbian updates run in a single batch
                forward pass after generation rather than token-by-token.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.memory_path = memory_path
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rep_penalty = rep_penalty
        self.plastic = plastic
        self.deferred_learning = deferred_learning

        self.history: List[dict] = []
        self._last_ctx: Optional[torch.Tensor] = None
        self._last_response_ids: List[int] = []

        self._stop_ids = self._get_hf_stop_ids()
        self._device = next(model.parameters()).device

        self._load_memory()

    def _get_hf_stop_ids(self) -> Set[int]:
        """Collect token ids that signal end-of-response for the HF tokenizer.

        Tries a set of common special token strings (``<|end|>``, ``</s>``,
        ``<|im_end|>``, etc.) and always includes ``eos_token_id``.

        Returns:
            Set of integer token ids that should stop generation when sampled.
        """
        candidates = [
            "<|end|>", "<|endoftext|>", "<|eot_id|>",
            "</s>", "<|im_end|>", "<|end_of_text|>",
        ]
        ids: Set[int] = set()
        for tok in candidates:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if tid not in (None, 0, -1, self.tokenizer.unk_token_id):
                    ids.add(tid)
            except Exception:
                pass
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    def _build_context_hf(self, user_input: str) -> torch.Tensor:
        """Build a tokenized prompt tensor for the HuggingFace backend.

        Applies the tokenizer's chat template to the system message, truncated
        conversation history, and the new user turn. History turns are dropped
        oldest-first (in pairs) until the token count fits within
        ``max_context_tokens``.

        Args:
            user_input: The user's current message.

        Returns:
            Integer token tensor of shape [1, T] on ``self._device``.
        """
        system_msg  = {"role": "system", "content": HF_SYSTEM}
        current_msg = {"role": "user",   "content": user_input}

        def _apply(msgs):
            kwargs = dict(tokenize=True, add_generation_prompt=True, return_tensors="pt")
            try:
                ret = self.tokenizer.apply_chat_template(msgs, **kwargs)
            except Exception:
                ret = self.tokenizer.apply_chat_template(
                    [m for m in msgs if m["role"] != "system"], **kwargs
                )
            if hasattr(ret, "input_ids"):
                return ret.input_ids
            elif isinstance(ret, dict) and "input_ids" in ret:
                return ret["input_ids"]
            return ret

        history = list(self.history)
        while True:
            msgs = [system_msg] + history + [current_msg]
            ids  = _apply(msgs)
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor([ids])
            if ids.shape[1] <= self.max_context_tokens or not history:
                break
            history = history[2:] if len(history) >= 2 else []

        self._last_context_tokens = ids.shape[1]
        return ids.to(self._device)

    def _load_memory(self) -> None:
        """Restore plastic state from ``memory_path`` if the file exists."""
        if not Path(self.memory_path).exists():
            return
        from .loader import load_plastic_state_hf
        load_plastic_state_hf(self.model, self.memory_path)

    def _save_memory(self) -> None:
        """Persist the current plastic state to ``memory_path``."""
        from .loader import save_plastic_state_hf
        save_plastic_state_hf(self.model, self.memory_path)

    def _total_plastic_norm(self) -> float:
        """Return the sum of L2 norms of ``delta_w`` across all LiquidLinear layers."""
        return sum(
            m.plastic_delta.norm().item()
            for m in self.model.modules()
            if isinstance(m, LiquidLinear)
        )

    @torch.no_grad()
    def _generate(self, user_input: str) -> str:
        """Generate a response token by token, streaming each piece to stdout.

        Attempts KV-cache inference (O(N+M)). Falls back to full-sequence
        recomputation (O(N×M)) if the model or installed version of
        transformers does not support ``DynamicCache``.

        Args:
            user_input: The user's current message.

        Returns:
            The complete generated response as a stripped string.
        """
        self.model.eval()

        ctx = self._build_context_hf(user_input)
        self._last_ctx = ctx
        self._last_response_ids = []

        generated: List[int] = []
        response  = ""
        prev_decoded = ""

        set_plastic_mode(False)

        _kvcache = False
        try:
            from transformers.cache_utils import DynamicCache
            out = self.model(ctx, past_key_values=DynamicCache(), use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[0, -1, :]
            _kvcache = True
        except Exception:
            out = self.model(ctx, use_cache=False)
            logits = out.logits[0, -1, :]
            past_key_values = None

        for step in range(self.max_new_tokens):
            next_id = _sample(logits, generated, self.temperature,
                              self.top_k, self.top_p, self.rep_penalty)
            if next_id in self._stop_ids:
                break

            generated.append(next_id)

            current = self.tokenizer.decode(generated, skip_special_tokens=True)
            piece = current[len(prev_decoded):]
            prev_decoded = current
            print(piece, end="", flush=True)
            response += piece

            if step < self.max_new_tokens - 1:
                if _kvcache:
                    next_input = torch.tensor([[next_id]], device=self._device)
                    out = self.model(next_input,
                                    past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                    logits = out.logits[0, -1, :]
                else:
                    full_ids = torch.cat(
                        [ctx, torch.tensor([generated], device=self._device)], dim=1
                    )
                    out = self.model(full_ids, use_cache=False)
                    logits = out.logits[0, -1, :]

        self._last_response_ids = generated

        set_plastic_mode(True)
        print()
        return response.strip()

    @torch.no_grad()
    def _learn_deferred(self) -> None:
        """Run a batch Hebbian update over the full context + response block.

        Concatenates the stored prompt context with the generated response ids
        and performs one forward pass with ``plastic=True``. This updates all
        ΔW tensors in a single step, which is significantly more efficient
        than updating ΔW token-by-token inside the KV-cache generation loop.
        """
        if not self._last_response_ids or self._last_ctx is None:
            return
        resp = torch.tensor([self._last_response_ids], device=self._device)
        full_ids = torch.cat([self._last_ctx, resp], dim=1)

        max_len = getattr(
            self.model.config, "max_position_embeddings",
            self.max_context_tokens + self.max_new_tokens,
        )
        if full_ids.shape[1] > max_len:
            full_ids = full_ids[:, -max_len:]

        set_plastic_mode(True)
        self.model(full_ids, use_cache=False)
        set_plastic_mode(False)

    def chat(self, user_input: str) -> str:
        """Process one user turn: generate a response and optionally learn from it.

        After generation, runs deferred learning, appends both sides of the
        turn to ``history``, and saves plastic state to disk when
        ``plastic=True``.

        Args:
            user_input: The user's message.

        Returns:
            The assistant's response string.
        """
        response = self._generate(user_input)
        if self.plastic and self.deferred_learning:
            self._learn_deferred()
        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": response})
        if self.plastic:
            self._save_memory()
        return response

    def _handle_command(self, raw: str) -> None:
        """Dispatch a slash command entered at the REPL prompt.

        Supported commands: ``/help``, ``/save``, ``/load``, ``/stats``,
        ``/teach``, ``/consolidate``, ``/reset``, ``/clear``, ``/plastic``,
        ``/temp``, ``/tokens``, ``/quit``, ``/exit``.

        Args:
            raw: The raw input string including the leading ``/``.
        """
        parts = raw.strip().split()
        cmd = parts[0].lower()

        if cmd == "/help":
            print("""
  Commands
  ─────────────────────────────────────────────────────────
  /help              Show this help message
  /save [path]       Save plastic state  (default: cln_memory.pt)
  /load [path]       Load saved plastic state
  /stats             Per-layer plasticity statistics
  /teach <file>      Learn from a document file (modifies ΔW)
  /consolidate       EWC consolidation — protect current knowledge
  /reset             Reset ΔW to zero (wipe all online learning)
  /clear             Clear conversation history (ΔW is preserved)
  /plastic on|off    Enable / disable online learning
  /temp <n>          Sampling temperature  (e.g. /temp 0.7)
  /tokens <n>        Max tokens per response  (e.g. /tokens 200)
  /quit  /exit       Save and exit
  ─────────────────────────────────────────────────────────
""")
        elif cmd == "/save":
            path = parts[1] if len(parts) > 1 else self.memory_path
            if path == self.memory_path:
                self._save_memory()
            else:
                __import__(
                    "cln.loader", fromlist=["save_plastic_state_hf"]
                ).save_plastic_state_hf(self.model, path)
            print(f"  [CLN] Saved → {path}")

        elif cmd == "/load":
            path = parts[1] if len(parts) > 1 else self.memory_path
            self._load_memory()
            print(f"  [CLN] Loaded ← {path}")

        elif cmd == "/stats":
            total = self._total_plastic_norm()
            layers = [
                (n, m) for n, m in self.model.named_modules()
                if isinstance(m, LiquidLinear)
            ]
            print(f"\n  Total ‖ΔW‖ = {total:.5f}  |  liquid layers = {len(layers)}")
            print(f"  {'Layer':<48}  {'‖ΔW‖':>9}  {'‖Ω‖':>9}  {'ratio':>7}")
            print("  " + "─" * 76)
            for name, m in layers[:12]:
                label = name[-46:].ljust(48)
                dn = m.plastic_delta.norm().item()
                fn = m.fisher.float().norm().item()
                bn = m.weight.data.float().norm().item()
                print(f"  {label}  {dn:>9.5f}  {fn:>9.5f}  {dn/(bn+1e-8):>6.3%}")
            if len(layers) > 12:
                print(f"  ... and {len(layers)-12} more layers")
            print()

        elif cmd == "/consolidate":
            from .loader import consolidate_hf
            consolidate_hf(self.model)
            print("  [CLN] Memory consolidated — current knowledge is now protected.")

        elif cmd == "/reset":
            for m in self.model.modules():
                if isinstance(m, LiquidLinear):
                    m.reset_plasticity()
            print("  [CLN] ΔW reset — all online memory has been wiped.")

        elif cmd == "/clear":
            self.history.clear()
            print("  [CLN] Conversation history cleared (ΔW preserved).")

        elif cmd == "/plastic":
            if len(parts) > 1:
                self.plastic = parts[1].lower() in ("on", "true", "1", "yes")
            print(f"  [CLN] Online learning: {'ON' if self.plastic else 'OFF'}")

        elif cmd == "/temp":
            try:
                self.temperature = float(parts[1])
                print(f"  [CLN] Temperature → {self.temperature}")
            except (IndexError, ValueError):
                print("  Usage: /temp <number>")

        elif cmd == "/tokens":
            try:
                self.max_new_tokens = int(parts[1])
                print(f"  [CLN] Max tokens → {self.max_new_tokens}")
            except (IndexError, ValueError):
                print("  Usage: /tokens <integer>")

        elif cmd == "/teach":
            if len(parts) < 2:
                print("  Usage: /teach <file.txt> [epochs=3] [eta=10]")
                return
            path      = parts[1]
            epochs    = int(parts[2])   if len(parts) > 2 else 3
            eta_mult  = float(parts[3]) if len(parts) > 3 else 10.0
            try:
                from .learn import learn_file
                tok = self.tokenizer
                learn_file(
                    self.model, path, tokenizer=tok,
                    epochs=epochs, eta_multiplier=eta_mult,
                    save_path=self.memory_path,
                )
            except FileNotFoundError as e:
                print(f"  Error: {e}")
            except Exception as e:
                print(f"  Error while learning from file: {e}")

        elif cmd in ("/quit", "/exit", "/q"):
            self._exit()

        else:
            print(f"  Unknown command: {cmd}  (type /help)")

    def _exit(self) -> None:
        """Save plastic state (if learning is enabled) and exit the process."""
        if self.plastic:
            self._save_memory()
            print(f"  [CLN] Memory saved → {self.memory_path}")
        print("  Goodbye.")
        sys.exit(0)

    def run(self) -> None:
        """Start the interactive REPL loop.

        Prints a welcome banner with model and session information, then
        enters a ``input()`` loop. Lines starting with ``/`` are dispatched
        to ``_handle_command()``; all other lines are sent to ``chat()``.
        The loop exits cleanly on ``EOFError`` or ``KeyboardInterrupt``.
        """
        model_name = getattr(self.model.config, "_name_or_path", "HuggingFace")

        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  CLN Chat  —  Continuous Liquid Network                  ║")
        print("╚══════════════════════════════════════════════════════════╝")
        learn_mode = (
            ("deferred" if self.deferred_learning else "online")
            if self.plastic else "OFF"
        )
        print(f"  Model              : {model_name}")
        print(f"  Online learning    : {learn_mode}")
        print(f"  Memory             : {self.memory_path}")
        print(f"  Temp / top-k / top-p : {self.temperature} / {self.top_k} / {self.top_p}")

        if Path(self.memory_path).exists():
            norm = self._total_plastic_norm()
            print(f"  [Prior session restored — ‖ΔW‖ = {norm:.5f}]")

        print()
        print("  Type /help to see available commands.\n")

        while True:
            try:
                user_input = input("You:  ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self._exit()

            if not user_input:
                continue
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            print("CLN: ", end="", flush=True)
            self.chat(user_input)
            print()
