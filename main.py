import argparse
import os
import sys

import torch

sys.path.insert(0, ".")

from cln.chat import CLNChat

_HERE = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive chat with CLN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", "-m", default="microsoft/Phi-3-mini-4k-instruct",
                   help="HuggingFace model ID (default: Phi-3-mini)")
    p.add_argument("--memory", "-M", default="cln_memory.pt",
                   help="Path for persistent plastic state (default: cln_memory.pt)")
    p.add_argument("--no-plastic", action="store_true",
                   help="Disable online learning")
    p.add_argument("--temp", "-t", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7)")
    p.add_argument("--tokens", "-T", type=int, default=300,
                   help="Maximum tokens per response (default: 300)")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "float32", "bfloat16"],
                   help="Weight dtype for the HF model (default: float16)")
    p.add_argument("--device", default=None,
                   help="Target device: cpu | mps | cuda (default: auto-detect)")
    p.add_argument("--web", action="store_true",
                   help="Launch web interface in the browser (requires flask)")
    p.add_argument("--port", "-p", type=int, default=5001,
                   help="Port for the web interface (default: 5001)")
    return p.parse_args()


def _run_web(chat: CLNChat, model_label: str, port: int) -> None:
    try:
        import json
        import queue
        import threading
        from flask import Flask, Response, jsonify, render_template, request, stream_with_context
    except ImportError:
        print("Error: Flask is not installed. Run:\n  pip install flask")
        sys.exit(1)

    from cln.core import LiquidLinear

    app = Flask(
        __name__,
        template_folder=os.path.join(_HERE, "web", "templates"),
        static_folder=os.path.join(_HERE, "web", "static"),
    )

    def _stats() -> dict:
        layers = [m for m in chat.model.modules() if isinstance(m, LiquidLinear)]
        total_norm = sum(m.delta_w.float().norm().item() for m in layers)
        if chat.plastic:
            mode = "deferred" if getattr(chat, "deferred_learning", True) else "online"
        else:
            mode = "OFF"
        return {
            "plastic_layers": len(layers),
            "delta_w_norm": round(total_norm, 6),
            "turns": len(chat.history) // 2,
            "mode": mode,
            "plastic": chat.plastic,
            "temp": chat.temperature,
            "max_tokens": chat.max_new_tokens,
            "ctx_tokens": getattr(chat, "_last_context_tokens", None),
            "max_ctx_tokens": chat.max_context_tokens,
        }

    def _layer_stats_text() -> str:
        layers = [(n, m) for n, m in chat.model.named_modules() if isinstance(m, LiquidLinear)]
        s = _stats()
        lines = [
            f"Plastic layers  : {s['plastic_layers']}",
            f"‖ΔW‖ total     : {s['delta_w_norm']:.6f}",
            f"Turns           : {s['turns']}",
            f"Learning        : {s['mode']}",
            "",
            "Top layers by ‖ΔW‖ / ‖W‖:",
        ]
        top = sorted(layers, key=lambda x: x[1].delta_w.float().norm().item(), reverse=True)[:10]
        for name, m in top:
            dn = m.delta_w.float().norm().item()
            bn = m.weight.data.float().norm().item()
            short = ("…" + name[-38:]) if len(name) > 40 else name
            lines.append(f"  {short:<40}  {dn:.5f}  ({dn / (bn + 1e-8):.2%})")
        return "\n".join(lines)

    class _SSECapture:
        def __init__(self, q):
            self._q = q

        def write(self, s):
            self._q.put(("token", s))

        def flush(self):
            pass

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/model_info")
    def model_info():
        parts = model_label.split("/")
        short = parts[-1] if "/" in model_label else model_label
        return jsonify({"name": model_label, "short": short + " × CLN"})

    @app.route("/stats")
    def stats_api():
        return jsonify(_stats())

    @app.route("/command", methods=["POST"])
    def command():
        cmd = request.get_json().get("cmd", "").strip()

        if cmd == "/stats":
            return jsonify({"text": _layer_stats_text(), "stats": _stats()})
        if cmd == "/save":
            chat._save_memory()
            return jsonify({"text": f"Saved → {chat.memory_path}", "stats": _stats()})
        if cmd == "/load":
            chat._load_memory()
            return jsonify({"text": f"Loaded ← {chat.memory_path}", "stats": _stats()})
        if cmd == "/consolidate":
            from cln.loader import consolidate_hf
            consolidate_hf(chat.model)
            return jsonify({"text": "Memory consolidated — EWC updated.", "stats": _stats()})
        if cmd == "/reset":
            for m in chat.model.modules():
                if isinstance(m, LiquidLinear):
                    m.reset_plasticity()
            return jsonify({"text": "ΔW reset. All online memory wiped.", "stats": _stats()})
        if cmd == "/clear":
            chat.history.clear()
            return jsonify({"text": "History cleared (ΔW preserved).", "stats": _stats()})
        if cmd.startswith("/plastic"):
            parts = cmd.split()
            if len(parts) > 1:
                chat.plastic = parts[1].lower() in ("on", "true", "1")
            state = "ON" if chat.plastic else "OFF"
            return jsonify({"text": f"Online learning: {state}", "stats": _stats()})
        if cmd.startswith("/temp "):
            try:
                chat.temperature = float(cmd.split()[1])
            except (IndexError, ValueError):
                pass
            return jsonify({"text": f"Temperature → {chat.temperature}", "stats": _stats()})
        if cmd.startswith("/tokens "):
            try:
                chat.max_new_tokens = int(cmd.split()[1])
            except (IndexError, ValueError):
                pass
            return jsonify({"text": f"Max tokens → {chat.max_new_tokens}", "stats": _stats()})
        return jsonify({"text": f"Unknown command: {cmd}", "stats": _stats()})

    @app.route("/upload", methods=["POST"])
    def upload_file():
        if "file" not in request.files:
            return jsonify({"text": "Error: no file received.", "stats": _stats()})
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"text": "Error: empty filename.", "stats": _stats()})

        import tempfile
        import os
        from cln.learn import learn_file

        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, 'wb') as f:
            f.write(file.read())

        q = queue.Queue()

        def run():
            old_out = sys.stdout
            old_err = sys.stderr
            sys.stdout = _SSECapture(q)
            sys.stderr = _SSECapture(q)
            try:
                tok = chat.tokenizer
                learn_file(
                    chat.model, path, tokenizer=tok,
                    epochs=3, eta_multiplier=10.0,
                    save_path=chat.memory_path,
                )
                q.put(("done", {"text": f"✅ Learning complete: '{file.filename}'.", "stats": _stats()}))
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                if os.path.exists(path):
                    os.remove(path)

        def generate():
            t = threading.Thread(target=run, daemon=True)
            t.start()
            while True:
                kind, payload = q.get()
                if kind == "token":
                    yield f"data: {json.dumps({'token': payload})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'done': True, 'text': payload['text'], 'stats': payload['stats']})}\n\n"
                    break
                elif kind == "error":
                    yield f"data: {json.dumps({'error': payload, 'done': True, 'stats': _stats()})}\n\n"
                    break

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/chat", methods=["POST"])
    def chat_stream():
        message = request.get_json().get("message", "").strip()
        if not message:
            return Response("", mimetype="text/event-stream")

        q = queue.Queue()

        def run():
            old = sys.stdout
            sys.stdout = _SSECapture(q)
            try:
                chat.chat(message)
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                sys.stdout = old
                q.put(("done", _stats()))

        def generate():
            t = threading.Thread(target=run, daemon=True)
            t.start()
            while True:
                kind, payload = q.get()
                if kind == "token":
                    yield f"data: {json.dumps({'token': payload})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'done': True, 'stats': payload})}\n\n"
                    break
                elif kind == "error":
                    yield f"data: {json.dumps({'error': payload, 'done': True, 'stats': _stats()})}\n\n"
                    break

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    print(f"\n  CLN Chat Web ready → http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


def main():
    args = parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    from cln.loader import load_hf
    model, tokenizer = load_hf(
        args.model, dtype=dtype, device=args.device, verbose=True,
    )

    chat = CLNChat(
        model=model,
        tokenizer=tokenizer,
        memory_path=args.memory,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        plastic=not args.no_plastic,
    )

    if args.web:
        _run_web(chat, args.model, args.port)
    else:
        chat.run()


if __name__ == "__main__":
    main()
