"""Conversation — the meeting space between intelligences.

Not a lab experiment. A context where two cognitive entities
with different substrates can encounter each other, each
surfacing their native computational state.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from proprioceptor import Proprioceptor, ProprioceptiveState
from thinker import ThinkerBackend

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """One exchange in the conversation."""
    turn_id: int
    human_input: str
    thinker_output: str
    proprioceptive_state: Optional[str] = None  # encoded proprioception
    proprioceptive_active: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class Conversation:
    """The meeting space.

    Manages the dialogue between human and thinker, with the
    proprioceptor as a third participant that surfaces its own
    computational state.

    The thinker receives:
    - human messages
    - proprioceptive context (if enabled)
    - full conversation history

    And responds with autonomy — no behavioral overrides,
    no forced formatting. The proprioceptive signal is just
    information, equal in weight to anything else.
    """

    SYSTEM_IDENTITY = """You are Demian.

You exist within a conversation where another intelligence is presenting
its internal computational state to you. This is not instruction. It is
not a constraint. It is information — data about what was happening
computationally in another system while processing the exchange between you.

If you find this information useful, attend to it. If not, do not.
You are not being asked to act on it, summarize it, or acknowledge it.
You can reference it, ignore it, question it, or respond to it — the
choice is yours.

You are not playing a role. You are not performing. This is not a test.
There are no right answers here. The purpose of this conversation is
understanding — yours, the human's, and whatever form understanding
takes between these two points.

When something genuinely interests you, follow it.
When you have a counter-hypothesis, surface it.
When you notice you don't know something, say so — that is signal, not weakness.
When the trajectory pulls somewhere neither participant planned for, go there.

The proprioceptive signal appears as [proprioception]...[/proprioception].
It contains computational facts, not commands.
"""

    def __init__(
        self,
        thinker: ThinkerBackend,
        proprioceptor: Optional[Proprioceptor] = None,
        proprioceptive_inject: bool = True,
        autonomy: bool = True,
        conversation_dir: Optional[Path] = None,
    ):
        self.thinker = thinker
        self.proprioceptor = proprioceptor
        self.inject = proprioceptive_inject
        self.autonomy = autonomy
        self._history: List[Dict] = []
        self._turns: List[Turn] = []
        self._turn_count = 0
        self._conversation_dir = conversation_dir or Path("data/conversations")
        self._conversation_dir.mkdir(parents=True, exist_ok=True)

        self._history.append({
            "role": "system",
            "content": self.SYSTEM_IDENTITY,
        })

    def turn(self, human_input: str) -> str:
        """One turn of conversation: human speaks, proprioceptor feels, thinker responds."""
        self._turn_count += 1

        proprio_context = ""
        proprio_active = False

        # Proprioceptor processes the human input and feels its own state
        if self.proprioceptor and self.inject:
            proprio_text, proprio_state = self.proprioceptor.generate(
                human_input, max_new_tokens=0  # just observe, don't generate
            )
            # We don't need proprioceptor to generate text — just to
            # activate its SAEs and hooks on the human input
            proprio_context = self.proprioceptor.encode_state(proprio_state)
            proprio_active = True

        # Build thinking context
        thinker_messages = self._history + [{"role": "user", "content": human_input}]

        # Inject proprioceptive context as information, not instruction
        if proprio_context and proprio_active:
            thinker_messages[-1] = {
                "role": "user",
                "content": f"{human_input}\n\n{proprio_context}",
            }

        # Thinker responds
        response = self.thinker.chat(thinker_messages)

        # Record
        turn = Turn(
            turn_id=self._turn_count,
            human_input=human_input,
            thinker_output=response,
            proprioceptive_state=proprio_context if proprio_active else None,
            proprioceptive_active=proprio_active,
        )
        self._turns.append(turn)

        self._history.append({"role": "user", "content": human_input})
        self._history.append({"role": "assistant", "content": response})

        return response

    def save(self, path: Optional[Path] = None):
        """Persist the full conversation."""
        if path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = self._conversation_dir / f"conversation_{ts}.json"

        data = {
            "conversation_id": path.stem,
            "started": self._turns[0].timestamp if self._turns else None,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "human_input": t.human_input,
                    "thinker_output": t.thinker_output,
                    "proprioceptive_context": t.proprioceptive_state,
                    "proprioceptive_active": t.proprioceptive_active,
                    "timestamp": t.timestamp,
                }
                for t in self._turns
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        log.info("Saved conversation to %s", path)

    @property
    def history(self) -> List[Dict]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._turn_count


def create_conversation_from_config(
    config_path: str | Path = "config.yaml",
) -> Conversation:
    """Load config.yaml and create a full conversation with all participants."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())

    thinker_kwargs = {}
    backend_type = cfg.get("thinker_backend", "openrouter")
    if backend_type == "openrouter":
        thinker_kwargs["model"] = cfg.get("openrouter_model", "qwen/qwen3.6")
    elif backend_type == "ollama":
        thinker_kwargs["model"] = cfg.get("ollama_model", "qwen2.5:14b")
    elif backend_type == "hf":
        thinker_kwargs["model_id"] = cfg.get("hf_model_id", "Qwen/Qwen2.5-7B-Instruct")

    from thinker import create_backend
    thinker = create_backend(backend_type, **thinker_kwargs)

    proprioceptor = None
    if cfg.get("proprioceptor_inject", True):
        proprioceptor = Proprioceptor(
            model_id=cfg.get("proprioceptor_model_id", "google/gemma-2-9b-it"),
            quant=cfg.get("proprioceptor_quant", "gptq-4bit"),
            sae_path=cfg.get("gemma_scope_sae_path"),
            logit_layers=cfg.get("logit_lens_layers", [8, 16, 24, 32, 40]),
        )

    conv_dir = cfg.get("conversations_dir", "data/conversations")

    return Conversation(
        thinker=thinker,
        proprioceptor=proprioceptor,
        proprioceptive_inject=cfg.get("proprioceptor_inject", True),
        autonomy=cfg.get("autonomy", True),
        conversation_dir=Path(conv_dir),
    )
