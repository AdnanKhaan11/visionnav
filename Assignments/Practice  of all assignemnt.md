# MINI-PROJECT: VisionNav Agent Simulation Harness (VASH)

---

## 1. Project Title

**VASH — VisionNav Agent Simulation Harness**

Subtitle: *Test your agent's intelligence before touching a real screen.*

---

## 2. Real-World Purpose

When you write a test for a calculator, you don't need a real calculator. You simulate input and verify output.

VASH applies the same principle to AI agents.

It creates **simulated GUI environments** — pure Python state machines that behave like real applications — and runs agent policies through them. The agent sends actions, the environment transitions, you record and evaluate everything.

No real screen. No pyautogui. No OS dependencies. Runs perfectly in Google Colab.

---

## 3. Why This Matters in AI Systems

Every serious AI lab has an offline evaluation system. Google DeepMind uses simulated Atari games. OpenAI used simulated robotic arms. Anthropic uses eval harnesses for Claude.

The reason is simple:

```
Real environment testing:
  Slow (real actions take real time)
  Expensive (GPU + real computer)
  Non-deterministic (OS state changes)
  Hard to reproduce (screen changes between runs)

Simulated environment testing:
  Fast (no real I/O)
  Free (pure Python)
  Deterministic (same input = same output)
  Reproducible (same seed = same result)
  Scalable (run 1000 episodes overnight)
```

VASH teaches you how to build this class of system — which is one of the most important skills in production AI engineering.

---

## 4. Full Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        VASH                             │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ Environment │    │   Policy     │    │  Storage  │  │
│  │ (what world │    │ (what agent  │    │  (JSONL   │  │
│  │  looks like)│    │  decides)    │    │   files)  │  │
│  └──────┬──────┘    └──────┬───────┘    └─────┬─────┘  │
│         │                  │                  │        │
│         └──────────┬───────┘                  │        │
│                    ▼                          │        │
│          ┌─────────────────┐                  │        │
│          │ SimulationRunner│ ◄────────────────┘        │
│          │  (async engine) │                           │
│          └────────┬────────┘                           │
│                   │                                    │
│                   ▼                                    │
│          ┌─────────────────┐    ┌───────────────────┐  │
│          │   Episode       │───►│   EpisodeScorer   │  │
│          │ (full record of │    │ (quality metrics) │  │
│          │  what happened) │    └────────┬──────────┘  │
│          └─────────────────┘             │             │
│                                          ▼             │
│                                 ┌─────────────────┐    │
│                                 │ ReportFormatter │    │
│                                 │ (human output)  │    │
│                                 └─────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Data flow in one episode:**

```
1. Policy receives: current environment state description
2. Policy produces: one action string
3. Runner calls:    environment.step(action)
4. Environment:     validates action → transitions → returns new state
5. Runner records:  EpisodeStep(state, action, new_state, reward)
6. Repeat until:    goal reached OR max_steps exceeded
7. Scorer rates:    the complete episode
8. Storage saves:   episode as JSONL
```

---

## 5. Folder Structure

```
vash/                                ← project root (one folder in Colab)
│
├── environments/
│   ├── __init__.py
│   ├── base.py          ← abstract SimulatedEnvironment
│   ├── states.py        ← EnvironmentState, Transition, StepResult
│   └── scenarios.py     ← NotepadEnv, BrowserEnv, DialogEnv (3 concrete envs)
│
├── policies/
│   ├── __init__.py
│   ├── base.py          ← abstract AgentPolicy
│   └── scripted.py      ← ScriptedPolicy, RandomPolicy, VLMPolicy stub
│
├── harness/
│   ├── __init__.py
│   ├── episode.py       ← Episode, EpisodeStep dataclasses
│   ├── runner.py        ← async SimulationRunner
│   └── session.py       ← SimulationSession context manager
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py       ← EpisodeMetrics, BatchMetrics dataclasses
│   ├── scorer.py        ← EpisodeScorer
│   └── reports.py       ← ReportFormatter
│
├── storage/
│   ├── __init__.py
│   └── jsonl_store.py   ← EpisodeStore (save/load JSONL)
│
├── pipeline.py          ← EvaluationPipeline (ties everything together)
├── cli.py               ← command line entry point
└── tests/
    ├── test_environments.py
    ├── test_policies.py
    ├── test_runner.py
    ├── test_scorer.py
    └── test_storage.py
```

---

## 6. Responsibilities of Every File

```
environments/base.py
  → Defines the contract every environment must follow
  → Abstract methods: reset(), step(), describe(), is_goal_reached()
  → No implementation — only the interface

environments/states.py
  → Pure data: EnvironmentState, Transition, StepResult
  → All dataclasses, all frozen where appropriate
  → No logic — only structure

environments/scenarios.py
  → THREE concrete environments: NotepadEnv, BrowserEnv, DialogEnv
  → Each is a real state machine with defined transitions
  → You implement these — this is the hardest file

policies/base.py
  → Abstract AgentPolicy: one method — choose_action(state) → str
  → Async because future VLM policies will call external APIs

policies/scripted.py
  → ScriptedPolicy: follows a hardcoded list of actions
  → RandomPolicy: picks random valid actions (baseline)
  → VLMPolicy: stub for future LLM integration

harness/episode.py
  → EpisodeStep: one step record (state, action, reward, timestamp)
  → Episode: full episode record (all steps + outcome + metadata)
  → All dataclasses — data only, no logic

harness/runner.py
  → SimulationRunner: async engine that runs episodes
  → run_episode(policy, env) → Episode
  → run_batch(policy, env, n) → list[Episode]

harness/session.py
  → SimulationSession: context manager for managing runner lifecycle
  → Handles setup, teardown, error recovery

evaluation/metrics.py
  → EpisodeMetrics: success rate, avg steps, efficiency ratio
  → BatchMetrics: aggregates across multiple episodes

evaluation/scorer.py
  → EpisodeScorer: converts Episode → EpisodeMetrics
  → Scores: did it succeed? how efficient? how safe?

evaluation/reports.py
  → ReportFormatter: EpisodeMetrics/BatchMetrics → printable string
  → Same bar chart style as TrajectoryAnalyzer (you already know this pattern)

storage/jsonl_store.py
  → EpisodeStore: save Episode → JSONL, load JSONL → Episode
  → Uses generators for streaming (same as ActionRecorder)

pipeline.py
  → EvaluationPipeline: ties everything together
  → run(policy, environments, n_episodes) → BatchReport
  → The only file users need to import

cli.py
  → argparse entry point
  → python vash/cli.py --env notepad --policy scripted --episodes 10
```

---

## 7. Implementation Roadmap

```
Day 1 (2-3 hours):
  1. Build environments/states.py (pure dataclasses — easy warmup)
  2. Build environments/base.py (abstract class)
  3. Build ONE environment in scenarios.py: NotepadEnv
  4. Test: can you manually step through it?

Day 2 (2-3 hours):
  5. Build harness/episode.py (more dataclasses)
  6. Build policies/base.py and policies/scripted.py
  7. Build harness/runner.py (async — uses asyncio skills)
  8. Test: can ScriptedPolicy solve NotepadEnv?

Day 3 (2-3 hours):
  9. Build evaluation/metrics.py and evaluation/scorer.py
  10. Build evaluation/reports.py
  11. Build storage/jsonl_store.py
  12. Write all tests

Day 4 (1-2 hours):
  13. Build BrowserEnv and DialogEnv
  14. Build harness/session.py
  15. Build pipeline.py
  16. Build cli.py
  17. Run full evaluation, read the report
```

---

## 8. Starter Code for Every File

### `environments/states.py`

```python
"""
Pure data structures for simulated GUI environments.

No logic here — only data shapes.
Logic lives in the environment classes (scenarios.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TransitionOutcome(Enum):
    """Result of attempting a state transition."""
    SUCCESS         = "success"     # action valid, state changed
    INVALID_ACTION  = "invalid"     # action not valid in current state
    ALREADY_AT_GOAL = "at_goal"     # goal already reached


@dataclass(frozen=True)
class EnvironmentState:
    """
    One screen/state in a simulated GUI environment.

    Think of it like a page in a choose-your-own-adventure book.
    Each page has a name, a description, and available actions.

    frozen=True: states are immutable facts about the world.
    """
    name:            str
    description:     str                      # what the agent "sees"
    valid_actions:   frozenset[str]           # which actions are valid here
    is_goal:         bool = False             # True = task complete
    metadata:        dict[str, Any] = field(
        default_factory=dict,
        compare=False,                        # don't include in equality check
    )


@dataclass(frozen=True)
class Transition:
    """
    Defines what happens when an action is taken in a state.

    from_state + action → to_state + reward
    """
    from_state:  str    # name of the state we start in
    action:      str    # the action taken (regex pattern allowed)
    to_state:    str    # name of the state we end up in
    reward:      float  # reward signal (+1 = good, -0.1 = penalty, 0 = neutral)
    description: str = ""  # what happened (for logging)


@dataclass
class StepResult:
    """
    Result of calling environment.step(action).

    Returned after every agent action.
    Contains everything needed to decide next action.
    """
    new_state:   EnvironmentState
    reward:      float
    done:        bool           # True if goal reached or episode over
    outcome:     TransitionOutcome
    info:        str = ""       # human readable explanation
```

---

### `environments/base.py`

```python
"""
Abstract base class that all simulated environments must implement.

Every environment is a state machine:
  - has states (screens)
  - has transitions (what actions cause)
  - has one or more goal states
  - has a reset() method to start fresh
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from vash.environments.states import EnvironmentState, StepResult


class SimulatedEnvironment(ABC):
    """
    Base class for all simulated GUI environments.

    Subclass this and implement all abstract methods
    to create a new simulated environment.

    Example:
        class MyApp(SimulatedEnvironment):
            def reset(self): ...
            def step(self, action): ...
            def describe(self): ...
            def is_goal_reached(self): ...
            def valid_actions(self): ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this environment."""
        ...

    @abstractmethod
    def reset(self) -> EnvironmentState:
        """
        Reset environment to initial state.
        Must be called before the first step().
        Returns the initial state.
        """
        ...

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """
        Apply one action and return the result.

        Args:
            action: string describing the action
                    (e.g. "key:win+r", "type:notepad", "click:submit")

        Returns:
            StepResult with new state, reward, done flag
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """
        Return a text description of the current state.
        This is what the agent "sees" — like an OCR summary.
        """
        ...

    @abstractmethod
    def is_goal_reached(self) -> bool:
        """Return True if the task has been completed successfully."""
        ...

    @abstractmethod
    def valid_actions(self) -> frozenset[str]:
        """Return the set of valid actions in the current state."""
        ...

    @property
    def current_state(self) -> EnvironmentState:
        """Current environment state. Implemented by subclasses."""
        raise NotImplementedError

    def render(self) -> str:
        """
        Optional: render a human-readable view of current state.
        Default implementation uses describe().
        Override for prettier output.
        """
        return f"[{self.name}] {self.describe()}"
```

---

### `environments/scenarios.py`

```python
"""
Concrete simulated GUI environments.

Three environments of increasing complexity:
  1. NotepadEnv    — open Notepad, type text, save file
  2. BrowserEnv    — open browser, navigate to URL, find content
  3. FormEnv       — fill a multi-field form, submit it

Each is a proper state machine with:
  - defined states
  - defined transitions
  - defined goal condition
  - defined rewards
"""
from __future__ import annotations

import re
from typing import ClassVar

from vash.environments.base import SimulatedEnvironment
from vash.environments.states import (
    EnvironmentState,
    StepResult,
    Transition,
    TransitionOutcome,
)


# ─── NotepadEnv ───────────────────────────────────────────────────────────────

class NotepadEnv(SimulatedEnvironment):
    """
    Simulate: Open Notepad, type a message, save the file.

    State graph:
      DESKTOP
        → key:win+r → RUN_DIALOG
      RUN_DIALOG
        → type:notepad → NOTEPAD_TYPED
      NOTEPAD_TYPED
        → key:enter → NOTEPAD_OPEN
      NOTEPAD_OPEN
        → type:{any text} → TEXT_ENTERED
      TEXT_ENTERED
        → key:ctrl+s → SAVE_DIALOG
      SAVE_DIALOG
        → type:{filename} → FILENAME_TYPED
      FILENAME_TYPED
        → key:enter → FILE_SAVED  ← GOAL
    """

    @property
    def name(self) -> str:
        return "NotepadEnv"

    def reset(self) -> EnvironmentState:
        """Reset to desktop — Notepad not open."""
        self._state_name = "DESKTOP"
        self._typed_text = ""
        self._filename   = ""
        return self.current_state

    def step(self, action: str) -> StepResult:
        """
        Apply action and transition to new state.

        YOUR IMPLEMENTATION:
          Define transitions for all state+action combinations.
          Use the Transition objects for clarity.
          Return StepResult with appropriate reward.

          Rewards:
            +1.0 = goal reached
            +0.2 = correct action, moved forward
             0.0 = no-op (action had no effect)
            -0.1 = wrong action (penalize bad choices)
        """
        # TODO: Implement state machine transitions
        # Hint: use a dict mapping (state_name, action_pattern) → (new_state, reward)
        raise NotImplementedError("Implement NotepadEnv.step()")

    def describe(self) -> str:
        """
        Return text description of current state.
        This is what the agent reads — like OCR output.

        YOUR IMPLEMENTATION:
          Return different descriptions for each state.
          Include relevant context (what's visible, what's typed, etc.)
        """
        # TODO: Implement per-state descriptions
        raise NotImplementedError("Implement NotepadEnv.describe()")

    def is_goal_reached(self) -> bool:
        return self._state_name == "FILE_SAVED"

    def valid_actions(self) -> frozenset[str]:
        """
        Return valid actions for current state.
        Agent should only attempt these actions.

        YOUR IMPLEMENTATION:
          Map state names to their valid action sets.
        """
        # TODO: Implement per-state valid action sets
        raise NotImplementedError("Implement NotepadEnv.valid_actions()")

    @property
    def current_state(self) -> EnvironmentState:
        """Build current EnvironmentState from internal state name."""
        return EnvironmentState(
            name          = self._state_name,
            description   = self.describe() if hasattr(self, '_state_name') else "",
            valid_actions = self.valid_actions() if hasattr(self, '_state_name') else frozenset(),
            is_goal       = self.is_goal_reached() if hasattr(self, '_state_name') else False,
        )


# ─── BrowserEnv ───────────────────────────────────────────────────────────────

class BrowserEnv(SimulatedEnvironment):
    """
    Simulate: Open browser, navigate to a URL, verify content loaded.

    State graph:
      DESKTOP
        → key:win+r → RUN_DIALOG
      RUN_DIALOG
        → type:chrome → APP_TYPED
      APP_TYPED
        → key:enter → BROWSER_OPEN
      BROWSER_OPEN
        → click:address_bar → ADDRESS_BAR_FOCUSED
      ADDRESS_BAR_FOCUSED
        → type:{url} → URL_TYPED
      URL_TYPED
        → key:enter → PAGE_LOADING
      PAGE_LOADING
        → wait → PAGE_LOADED  ← GOAL

    YOUR IMPLEMENTATION:
      Implement this environment following the same pattern as NotepadEnv.
      Be creative with the describe() output — make it feel like a real browser.
    """

    @property
    def name(self) -> str:
        return "BrowserEnv"

    def reset(self) -> EnvironmentState:
        self._state_name = "DESKTOP"
        self._url        = ""
        return self.current_state

    def step(self, action: str) -> StepResult:
        raise NotImplementedError("Implement BrowserEnv.step()")

    def describe(self) -> str:
        raise NotImplementedError("Implement BrowserEnv.describe()")

    def is_goal_reached(self) -> bool:
        return self._state_name == "PAGE_LOADED"

    def valid_actions(self) -> frozenset[str]:
        raise NotImplementedError("Implement BrowserEnv.valid_actions()")

    @property
    def current_state(self) -> EnvironmentState:
        return EnvironmentState(
            name          = self._state_name,
            description   = self.describe() if hasattr(self, '_state_name') else "",
            valid_actions = self.valid_actions() if hasattr(self, '_state_name') else frozenset(),
            is_goal       = self.is_goal_reached() if hasattr(self, '_state_name') else False,
        )


# ─── FormEnv ──────────────────────────────────────────────────────────────────

class FormEnv(SimulatedEnvironment):
    """
    Simulate: Fill a multi-field form and submit it.

    Fields: name, email, message → Submit button

    This is the hardest environment because:
      - Multiple fields must all be filled (order matters)
      - Each field has validation (email must contain @)
      - Submit only works when all fields are valid

    State graph:
      FORM_EMPTY
        → click:name_field → NAME_FOCUSED
      NAME_FOCUSED
        → type:{name} → NAME_FILLED
      NAME_FILLED
        → click:email_field → EMAIL_FOCUSED
      EMAIL_FOCUSED
        → type:{valid_email} → EMAIL_FILLED       (must contain @)
        → type:{invalid}     → EMAIL_ERROR        (no @ → error)
      EMAIL_ERROR
        → click:email_field → EMAIL_FOCUSED       (try again)
      EMAIL_FILLED
        → click:message_field → MESSAGE_FOCUSED
      MESSAGE_FOCUSED
        → type:{message} → MESSAGE_FILLED
      MESSAGE_FILLED
        → click:submit → FORM_SUBMITTED  ← GOAL

    YOUR IMPLEMENTATION:
      This is the most complex environment.
      Pay attention to the email validation logic.
      An invalid email should NOT progress toward the goal.
    """

    @property
    def name(self) -> str:
        return "FormEnv"

    def reset(self) -> EnvironmentState:
        self._state_name   = "FORM_EMPTY"
        self._name_value   = ""
        self._email_value  = ""
        self._message_value = ""
        return self.current_state

    def step(self, action: str) -> StepResult:
        raise NotImplementedError("Implement FormEnv.step()")

    def describe(self) -> str:
        raise NotImplementedError("Implement FormEnv.describe()")

    def is_goal_reached(self) -> bool:
        return self._state_name == "FORM_SUBMITTED"

    def valid_actions(self) -> frozenset[str]:
        raise NotImplementedError("Implement FormEnv.valid_actions()")

    @property
    def current_state(self) -> EnvironmentState:
        return EnvironmentState(
            name          = self._state_name,
            description   = self.describe() if hasattr(self, '_state_name') else "",
            valid_actions = self.valid_actions() if hasattr(self, '_state_name') else frozenset(),
            is_goal       = self.is_goal_reached() if hasattr(self, '_state_name') else False,
        )
```

---

### `harness/episode.py`

```python
"""
Data structures representing one simulation episode.

An episode is a complete record of one agent attempt at one task.
Like a flight data recorder — everything that happened, in order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from vash.environments.states import EnvironmentState, StepResult


class EpisodeOutcome(Enum):
    """How the episode ended."""
    SUCCESS       = "success"       # goal reached
    FAILURE       = "failure"       # max steps reached without goal
    INVALID_POLICY = "invalid"      # policy produced unparseable actions
    ERROR         = "error"         # exception during episode


@dataclass
class EpisodeStep:
    """
    One step within an episode.

    Records exactly what happened at each moment:
      - What state the agent saw
      - What action it chose
      - What happened as a result
    """
    step_index:    int
    state_name:    str            # state BEFORE action
    state_desc:    str            # description the agent received
    action:        str            # action the agent chose
    reward:        float          # reward received
    new_state_name: str           # state AFTER action
    valid_action:  bool           # was action in valid_actions?
    timestamp:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    info:          str = ""       # extra context from environment


@dataclass
class Episode:
    """
    Complete record of one agent attempt at one task.

    Contains:
      - All steps in order
      - Final outcome (success/failure)
      - Metadata for later analysis
    """
    episode_id:    str
    env_name:      str
    policy_name:   str
    task:          str
    steps:         list[EpisodeStep] = field(default_factory=list)
    outcome:       EpisodeOutcome    = EpisodeOutcome.FAILURE
    total_reward:  float             = 0.0
    started_at:    str               = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at:   str = ""
    max_steps:     int = 20

    # ── Computed properties ────────────────────────────────────────────────

    @property
    def success(self) -> bool:
        return self.outcome == EpisodeOutcome.SUCCESS

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def efficiency(self) -> float:
        """
        How efficiently did the agent solve the task?
        1.0 = solved in minimum possible steps
        0.5 = took twice as many steps as needed
        0.0 = failed entirely

        Computed as: (goal_reached) / (steps_taken / max_steps)
        """
        if not self.success or self.n_steps == 0:
            return 0.0
        return 1.0 - (self.n_steps / self.max_steps)

    @property
    def invalid_action_ratio(self) -> float:
        """
        Fraction of steps where agent chose an invalid action.
        Lower is better. 0.0 = perfect. 1.0 = all invalid.
        """
        if not self.steps:
            return 0.0
        invalid = sum(1 for s in self.steps if not s.valid_action)
        return invalid / len(self.steps)
```

---

### `policies/base.py`

```python
"""Abstract base class for all agent policies."""
from __future__ import annotations

from abc import ABC, abstractmethod

from vash.environments.states import EnvironmentState


class AgentPolicy(ABC):
    """
    Abstract agent policy.

    A policy takes a state description and returns an action.
    This is the interface between the environment and the agent.

    Why async?
    Future policies will call VLM APIs over HTTP.
    Making the base class async now means we never need to rewrite.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this policy."""
        ...

    @abstractmethod
    async def choose_action(
        self,
        state:       EnvironmentState,
        history:     list[str],        # previous actions taken
        task:        str,              # the goal instruction
    ) -> str:
        """
        Choose the next action given current state.

        Args:
            state:   current environment state
            history: list of actions already taken (strings)
            task:    the goal instruction ("Open Notepad and type hello")

        Returns:
            action string (e.g. "key:win+r", "type:notepad", "click:submit")
        """
        ...

    async def reset(self) -> None:
        """
        Called before each new episode.
        Override if your policy has internal state to reset.
        """
        pass
```

---

### `policies/scripted.py`

```python
"""
Concrete policy implementations.

ScriptedPolicy: follows a pre-written action script
RandomPolicy:   picks random valid actions (establishes baseline)
"""
from __future__ import annotations

import random
from typing import Iterator

from vash.environments.states import EnvironmentState
from vash.policies.base import AgentPolicy


class ScriptedPolicy(AgentPolicy):
    """
    Follows a hardcoded sequence of actions.

    Use this to test that a specific action sequence solves an environment.
    Also useful as a baseline: if ScriptedPolicy fails, the environment is wrong.

    Usage:
        policy = ScriptedPolicy(
            name="notepad_solver",
            actions=[
                "key:win+r",
                "type:notepad",
                "key:enter",
                "type:Hello World",
                "key:ctrl+s",
                "type:my_file",
                "key:enter",
            ]
        )
    """

    def __init__(self, name: str, actions: list[str]) -> None:
        self._name    = name
        self._actions = actions
        self._index   = 0

    @property
    def name(self) -> str:
        return self._name

    async def reset(self) -> None:
        """Reset action index for new episode."""
        self._index = 0

    async def choose_action(
        self,
        state:   EnvironmentState,
        history: list[str],
        task:    str,
    ) -> str:
        """Return next action in script. Return 'done' when script exhausted."""
        if self._index >= len(self._actions):
            return "done"
        action       = self._actions[self._index]
        self._index += 1
        return action


class RandomPolicy(AgentPolicy):
    """
    Picks a random valid action at each step.

    Used to establish a random baseline:
    "How well does random chance solve this environment?"
    Any trained policy should score significantly higher than this.

    Also useful for stress-testing environments:
    random actions will hit every transition eventually.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "RandomPolicy"

    async def choose_action(
        self,
        state:   EnvironmentState,
        history: list[str],
        task:    str,
    ) -> str:
        """Pick a random action from the valid set."""
        if not state.valid_actions:
            return "done"
        return self._rng.choice(sorted(state.valid_actions))


class VLMPolicy(AgentPolicy):
    """
    STUB: Future integration with a real VLM.

    In Phase 10+ this will call your fine-tuned Qwen2.5-VL model.
    For now it just returns a placeholder.

    YOUR TASK:
      Wire this to LocalModelBackend.predict_action()
      The environment's describe() output becomes the observation.
      The task string becomes the instruction.
    """

    @property
    def name(self) -> str:
        return "VLMPolicy"

    async def choose_action(
        self,
        state:   EnvironmentState,
        history: list[str],
        task:    str,
    ) -> str:
        # TODO: Call real VLM when available
        # For now: always return done (stub)
        return "done"
```

---

### `harness/runner.py`

```python
"""
Async simulation runner — runs episodes and records everything.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

import structlog

from vash.environments.base import SimulatedEnvironment
from vash.environments.states import TransitionOutcome
from vash.harness.episode import Episode, EpisodeOutcome, EpisodeStep
from vash.policies.base import AgentPolicy

log = structlog.get_logger(__name__)


class SimulationRunner:
    """
    Runs one agent policy through one simulated environment.

    Async because:
      - Policies may call async APIs (VLMs, databases)
      - Running many episodes concurrently becomes possible
      - Consistent with VisionNav's async architecture

    Usage:
        runner = SimulationRunner(max_steps=20)
        episode = await runner.run_episode(policy, env, task="Open Notepad")
    """

    def __init__(self, max_steps: int = 20) -> None:
        self._max_steps = max_steps

    async def run_episode(
        self,
        policy:  AgentPolicy,
        env:     SimulatedEnvironment,
        task:    str,
    ) -> Episode:
        """
        Run one complete episode.

        Steps:
          1. Reset environment and policy
          2. Loop: policy chooses action → environment steps
          3. Record every step
          4. Stop when: goal reached OR max_steps exceeded OR error
          5. Return complete Episode record

        Args:
            policy: the agent deciding actions
            env:    the simulated environment
            task:   natural language task description

        Returns:
            Complete Episode with all steps and outcome
        """
        episode_id = str(uuid.uuid4())[:8]

        # ── Setup ─────────────────────────────────────────────────────────
        state = env.reset()
        await policy.reset()

        episode = Episode(
            episode_id  = episode_id,
            env_name    = env.name,
            policy_name = policy.name,
            task        = task,
            max_steps   = self._max_steps,
        )

        log.info(
            "episode_started",
            episode_id = episode_id,
            env        = env.name,
            policy     = policy.name,
            task       = task,
        )

        action_history: list[str] = []

        # ── Episode loop ──────────────────────────────────────────────────
        for step_index in range(self._max_steps):
            try:
                # Policy decides what to do
                action = await policy.choose_action(
                    state   = state,
                    history = action_history,
                    task    = task,
                )

                # Was this action valid?
                valid = action in state.valid_actions

                # Environment processes the action
                result = env.step(action)

                # Record this step
                step = EpisodeStep(
                    step_index     = step_index,
                    state_name     = state.name,
                    state_desc     = state.description,
                    action         = action,
                    reward         = result.reward,
                    new_state_name = result.new_state.name,
                    valid_action   = valid,
                    info           = result.info,
                )
                episode.steps.append(step)
                episode.total_reward += result.reward
                action_history.append(action)

                log.debug(
                    "episode_step",
                    episode_id  = episode_id,
                    step        = step_index,
                    action      = action,
                    state_from  = state.name,
                    state_to    = result.new_state.name,
                    reward      = result.reward,
                    valid       = valid,
                )

                # Move to next state
                state = result.new_state

                # Check if goal reached
                if env.is_goal_reached():
                    episode.outcome = EpisodeOutcome.SUCCESS
                    break

            except Exception as exc:
                log.error(
                    "episode_error",
                    episode_id = episode_id,
                    step       = step_index,
                    error      = str(exc),
                    exc_info   = True,
                )
                episode.outcome = EpisodeOutcome.ERROR
                break
        else:
            # Loop completed without break — max steps reached
            episode.outcome = EpisodeOutcome.FAILURE

        # ── Finalize ──────────────────────────────────────────────────────
        episode.finished_at = datetime.now(timezone.utc).isoformat()

        log.info(
            "episode_finished",
            episode_id = episode_id,
            outcome    = episode.outcome.value,
            steps      = episode.n_steps,
            reward     = round(episode.total_reward, 3),
            success    = episode.success,
        )

        return episode

    async def run_batch(
        self,
        policy:     AgentPolicy,
        env:        SimulatedEnvironment,
        task:       str,
        n_episodes: int,
    ) -> list[Episode]:
        """
        Run multiple episodes sequentially.

        Why not concurrent?
        Most environments have internal state — running them
        concurrently would require separate instances per episode.
        Sequential is simpler and correct.

        For concurrent runs: create N env instances and use asyncio.gather.
        """
        episodes: list[Episode] = []
        for i in range(n_episodes):
            log.info("batch_progress", episode=i + 1, total=n_episodes)
            episode = await self.run_episode(policy, env, task)
            episodes.append(episode)
        return episodes
```

---

### `harness/session.py`

```python
"""
SimulationSession — context manager for managing evaluation sessions.

Handles setup, tracking, and cleanup automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import structlog

from vash.environments.base import SimulatedEnvironment
from vash.harness.episode import Episode
from vash.harness.runner import SimulationRunner
from vash.policies.base import AgentPolicy
from vash.storage.jsonl_store import EpisodeStore

log = structlog.get_logger(__name__)


class SimulationSession:
    """
    Context manager that runs a complete evaluation session.

    Manages:
      - Runner lifecycle
      - Automatic episode storage
      - Session-level statistics

    Usage:
        async with SimulationSession(
            policy   = scripted_policy,
            env      = notepad_env,
            task     = "Open Notepad and save a file",
            save_dir = Path("data/episodes"),
        ) as session:
            episodes = await session.run(n_episodes=10)
            print(f"Success rate: {session.success_rate:.0%}")
    """

    def __init__(
        self,
        policy:    AgentPolicy,
        env:       SimulatedEnvironment,
        task:      str,
        save_dir:  Path | None = None,
        max_steps: int = 20,
    ) -> None:
        self._policy    = policy
        self._env       = env
        self._task      = task
        self._save_dir  = Path(save_dir) if save_dir else None
        self._runner    = SimulationRunner(max_steps=max_steps)
        self._episodes: list[Episode] = []
        self._store:    EpisodeStore | None = None

    async def __aenter__(self) -> "SimulationSession":
        """Set up storage when entering context."""
        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            store_path    = self._save_dir / f"{self._env.name}_episodes.jsonl"
            self._store   = EpisodeStore(store_path)

        log.info(
            "session_started",
            env    = self._env.name,
            policy = self._policy.name,
            task   = self._task,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Log summary when leaving context."""
        log.info(
            "session_finished",
            total_episodes = len(self._episodes),
            success_rate   = f"{self.success_rate:.0%}",
        )

    async def run(self, n_episodes: int = 1) -> list[Episode]:
        """
        Run N episodes and save each one.

        Returns:
            List of all completed episodes
        """
        episodes = await self._runner.run_batch(
            policy     = self._policy,
            env        = self._env,
            task       = self._task,
            n_episodes = n_episodes,
        )
        self._episodes.extend(episodes)

        # Save each episode if storage is configured
        if self._store:
            for ep in episodes:
                self._store.save(ep)

        return episodes

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that succeeded."""
        if not self._episodes:
            return 0.0
        return sum(1 for e in self._episodes if e.success) / len(self._episodes)

    @property
    def episodes(self) -> list[Episode]:
        return list(self._episodes)
```

---

### `evaluation/metrics.py`

```python
"""
Metrics computed from episode data.
Pure data — no logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    """Metrics for one episode."""
    episode_id:           str
    env_name:             str
    policy_name:          str
    success:              bool
    n_steps:              int
    total_reward:         float
    efficiency:           float    # 0.0 to 1.0
    invalid_action_ratio: float    # lower is better
    outcome:              str


@dataclass
class BatchMetrics:
    """Aggregate metrics across multiple episodes."""
    env_name:             str
    policy_name:          str
    n_episodes:           int
    n_successes:          int
    success_rate:         float    # 0.0 to 1.0
    avg_steps:            float
    avg_reward:           float
    avg_efficiency:       float
    avg_invalid_ratio:    float
    episode_metrics:      list[EpisodeMetrics] = field(default_factory=list)

    @property
    def n_failures(self) -> int:
        return self.n_episodes - self.n_successes
```

---

### `evaluation/scorer.py`

```python
"""
Converts raw Episode data into scored EpisodeMetrics and BatchMetrics.
"""
from __future__ import annotations

from vash.evaluation.metrics import BatchMetrics, EpisodeMetrics
from vash.harness.episode import Episode


class EpisodeScorer:
    """
    Scores one or many episodes.

    Usage:
        scorer  = EpisodeScorer()
        metrics = scorer.score_episode(episode)
        batch   = scorer.score_batch(episodes)
    """

    def score_episode(self, episode: Episode) -> EpisodeMetrics:
        """Convert one Episode into EpisodeMetrics."""
        return EpisodeMetrics(
            episode_id           = episode.episode_id,
            env_name             = episode.env_name,
            policy_name          = episode.policy_name,
            success              = episode.success,
            n_steps              = episode.n_steps,
            total_reward         = round(episode.total_reward, 3),
            efficiency           = round(episode.efficiency, 3),
            invalid_action_ratio = round(episode.invalid_action_ratio, 3),
            outcome              = episode.outcome.value,
        )

    def score_batch(self, episodes: list[Episode]) -> BatchMetrics:
        """
        Compute aggregate metrics across all episodes.

        YOUR IMPLEMENTATION:
          1. Score each episode individually
          2. Aggregate: success rate, avg steps, avg reward, etc.
          3. Return BatchMetrics

        Hint: use sum() and len() on the list of episode metrics.
        """
        # TODO: Implement batch scoring
        raise NotImplementedError("Implement EpisodeScorer.score_batch()")
```

---

### `evaluation/reports.py`

```python
"""
Formats metrics as human-readable terminal reports.
"""
from __future__ import annotations

from vash.evaluation.metrics import BatchMetrics, EpisodeMetrics


class ReportFormatter:
    """
    Converts metrics into human-readable reports.

    Same bar-chart style as TrajectoryAnalyzer.format_report().
    Your Assignment 5 experience directly applies here.
    """

    def format_episode(self, metrics: EpisodeMetrics) -> str:
        """
        Format one episode's metrics.

        Example output:
          Episode abc12345  [NotepadEnv / ScriptedPolicy]
          Outcome:   SUCCESS ✓
          Steps:     7 / 20
          Reward:    +1.8
          Efficiency: 0.65
          Invalid actions: 0.0%
        """
        # TODO: Implement episode report formatting
        raise NotImplementedError("Implement ReportFormatter.format_episode()")

    def format_batch(self, metrics: BatchMetrics) -> str:
        """
        Format batch metrics with bar charts.

        Example output:
          ══════════════════════════════════════════════
          Batch Evaluation Report
          Environment: NotepadEnv | Policy: ScriptedPolicy
          ══════════════════════════════════════════════
          Episodes:      10
          Success Rate:  80%  ████████░░
          Avg Steps:     8.2 / 20
          Avg Reward:   +1.4
          Avg Efficiency: 0.59
          ══════════════════════════════════════════════
        """
        # TODO: Implement batch report formatting
        raise NotImplementedError("Implement ReportFormatter.format_batch()")
```

---

### `storage/jsonl_store.py`

```python
"""
Save and load Episode objects as JSONL files.
Uses generators for memory-efficient streaming.
Same pattern as ActionRecorder — you already know this.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import structlog

from vash.harness.episode import Episode, EpisodeOutcome, EpisodeStep

log = structlog.get_logger(__name__)


class EpisodeStore:
    """
    Persist Episode objects to disk as JSONL.

    One line per episode. Streaming read with generators.
    Append mode — never overwrites existing data.

    YOUR IMPLEMENTATION:
      save(episode) → serialize to JSON and append one line
      load()        → yield Episode objects from file (generator)
      load_successful() → yield only successful episodes

    Hint: Episode contains EpisodeStep objects inside it.
    You need to serialize the full nested structure.
    Use asdict() from dataclasses for serialization.
    For deserialization, reconstruct EpisodeStep objects from dicts.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def save(self, episode: Episode) -> None:
        """
        Serialize episode to one JSON line and append to file.

        YOUR IMPLEMENTATION:
          1. Convert episode to dict (use asdict from dataclasses)
          2. Convert EpisodeOutcome enum to its .value string
          3. Write as single JSON line (mode="a")
          4. Log the save
        """
        raise NotImplementedError("Implement EpisodeStore.save()")

    def load(self) -> Iterator[Episode]:
        """
        Stream all episodes from file.

        YOUR IMPLEMENTATION:
          Generator — yield one Episode per line.
          Skip empty lines and corrupt JSON (log warning).
          Reconstruct EpisodeStep objects from dicts.
          Reconstruct EpisodeOutcome enum from string.
        """
        raise NotImplementedError("Implement EpisodeStore.load()")

    def load_successful(self) -> Iterator[Episode]:
        """
        Stream only successful episodes.
        Uses load() generator — filters in-place.
        One line — shows power of generators.
        """
        return (ep for ep in self.load() if ep.success)

    def count(self) -> int:
        """Count total episodes without loading all into memory."""
        if not self._path.exists():
            return 0
        with open(self._path) as f:
            return sum(1 for line in f if line.strip())
```

---

### `pipeline.py`

```python
"""
Master evaluation pipeline.
Ties everything together in one clean interface.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from vash.environments.base import SimulatedEnvironment
from vash.evaluation.metrics import BatchMetrics
from vash.evaluation.reports import ReportFormatter
from vash.evaluation.scorer import EpisodeScorer
from vash.harness.episode import Episode
from vash.harness.session import SimulationSession
from vash.policies.base import AgentPolicy
from vash.storage.jsonl_store import EpisodeStore

log = structlog.get_logger(__name__)


class EvaluationPipeline:
    """
    Run → Score → Report → Save — in one call.

    Usage:
        pipeline = EvaluationPipeline(save_dir=Path("data/results"))
        report   = await pipeline.run(
            policy      = ScriptedPolicy("solver", NOTEPAD_ACTIONS),
            env         = NotepadEnv(),
            task        = "Open Notepad and save a file",
            n_episodes  = 20,
        )
        print(report)
    """

    def __init__(self, save_dir: Path | None = None) -> None:
        self._save_dir = save_dir
        self._scorer   = EpisodeScorer()
        self._reporter = ReportFormatter()

    async def run(
        self,
        policy:     AgentPolicy,
        env:        SimulatedEnvironment,
        task:       str,
        n_episodes: int = 10,
    ) -> str:
        """
        Run full evaluation and return formatted report.

        YOUR IMPLEMENTATION:
          1. Use SimulationSession as context manager
          2. Run n_episodes
          3. Score with EpisodeScorer.score_batch()
          4. Format with ReportFormatter.format_batch()
          5. Return formatted string
        """
        raise NotImplementedError("Implement EvaluationPipeline.run()")
```

---

## 9. Test Cases

```python
# tests/test_environments.py

def test_notepad_initial_state():
    """Environment starts at DESKTOP."""
    env = NotepadEnv()
    state = env.reset()
    assert state.name == "DESKTOP"
    assert not env.is_goal_reached()

def test_notepad_goal_not_reached_initially():
    env = NotepadEnv()
    env.reset()
    assert env.is_goal_reached() is False

def test_notepad_valid_actions_at_desktop():
    env = NotepadEnv()
    env.reset()
    assert "key:win+r" in env.valid_actions()

def test_notepad_invalid_action_penalized():
    env = NotepadEnv()
    env.reset()
    result = env.step("type:garbage")   # invalid at DESKTOP
    assert result.reward <= 0
    assert result.outcome == TransitionOutcome.INVALID_ACTION

def test_notepad_full_sequence_succeeds():
    env = NotepadEnv()
    env.reset()
    actions = ["key:win+r", "type:notepad", "key:enter",
               "type:Hello World", "key:ctrl+s", "type:myfile", "key:enter"]
    for action in actions:
        result = env.step(action)
    assert env.is_goal_reached()

def test_form_env_email_validation():
    """Invalid email should not progress toward goal."""
    env = FormEnv()
    env.reset()
    env.step("click:name_field")
    env.step("type:John")
    env.step("click:email_field")
    result = env.step("type:notanemail")   # no @ → error
    assert result.outcome != TransitionOutcome.SUCCESS or \
           env.current_state.name == "EMAIL_ERROR"


# tests/test_runner.py

@pytest.mark.asyncio
async def test_scripted_policy_solves_notepad():
    policy = ScriptedPolicy("solver", [
        "key:win+r", "type:notepad", "key:enter",
        "type:Hello World", "key:ctrl+s", "type:myfile", "key:enter"
    ])
    env     = NotepadEnv()
    runner  = SimulationRunner(max_steps=20)
    episode = await runner.run_episode(policy, env, "Open Notepad and save")
    assert episode.success is True

@pytest.mark.asyncio
async def test_max_steps_produces_failure():
    policy  = ScriptedPolicy("forever", ["key:win+r"] * 100)
    env     = NotepadEnv()
    runner  = SimulationRunner(max_steps=5)
    episode = await runner.run_episode(policy, env, "Open Notepad")
    assert episode.outcome == EpisodeOutcome.FAILURE
    assert episode.n_steps == 5

@pytest.mark.asyncio
async def test_random_policy_baseline():
    policy  = RandomPolicy(seed=42)
    env     = NotepadEnv()
    runner  = SimulationRunner(max_steps=50)
    episodes = await runner.run_batch(policy, env, "Open Notepad", n_episodes=20)
    # Random policy should occasionally succeed
    success_rate = sum(1 for e in episodes if e.success) / len(episodes)
    assert success_rate >= 0.0   # at minimum doesn't crash
    assert len(episodes) == 20


# tests/test_storage.py

def test_save_and_load_episode(tmp_path):
    store   = EpisodeStore(tmp_path / "episodes.jsonl")
    episode = Episode(episode_id="test01", env_name="NotepadEnv",
                      policy_name="test", task="test task")
    store.save(episode)
    loaded = list(store.load())
    assert len(loaded) == 1
    assert loaded[0].episode_id == "test01"

def test_load_successful_filters(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.jsonl")
    e1 = Episode(episode_id="a", env_name="X", policy_name="X",
                 task="t", outcome=EpisodeOutcome.SUCCESS)
    e2 = Episode(episode_id="b", env_name="X", policy_name="X",
                 task="t", outcome=EpisodeOutcome.FAILURE)
    store.save(e1)
    store.save(e2)
    successful = list(store.load_successful())
    assert len(successful) == 1
    assert successful[0].episode_id == "a"
```

---

## 10. Common Engineering Mistakes to Avoid

```
Mistake 1: Mutable default arguments
  WRONG:  def __init__(self, steps=[]):
  RIGHT:  def __init__(self, steps=None): self.steps = steps or []

  Python creates ONE list object shared across ALL instances.
  Use None and create inside __init__.

Mistake 2: Not resetting environment between episodes
  WRONG:  run two episodes without calling env.reset()
  RIGHT:  runner always calls env.reset() at start of each episode

  Without reset, episode 2 starts where episode 1 ended.
  All your data is corrupted.

Mistake 3: Using time.sleep in async code
  WRONG:  time.sleep(1.0)    inside async function
  RIGHT:  await asyncio.sleep(1.0)

  time.sleep blocks the event loop. Everything freezes.

Mistake 4: Catching Exception too broadly
  WRONG:  except Exception: pass
  RIGHT:  except (JSONDecodeError, KeyError) as exc: log.warning(...)

  Silent failures produce mysterious bugs hours later.

Mistake 5: Storing enum objects in JSON
  WRONG:  {"outcome": EpisodeOutcome.SUCCESS}
  RIGHT:  {"outcome": EpisodeOutcome.SUCCESS.value}  →  "success"

  JSON doesn't know Python enums. Always use .value for serialization.

Mistake 6: Generator consumed twice
  WRONG:
    episodes = store.load()
    count    = len(list(episodes))    # generator consumed here
    for ep in episodes: ...           # empty — nothing left

  RIGHT:
    episodes = list(store.load())     # materialize once
    count    = len(episodes)
    for ep in episodes: ...           # works correctly
```

---

## 11. Recommended Coding Practices

```
1. Write the test BEFORE the implementation
   Create test_notepad_full_sequence_succeeds() first.
   It will fail (red). Implement step() until it passes (green).
   This is TDD — Test Driven Development.

2. Name functions by what they RETURN not what they DO
   WRONG: def process_episode(ep): ...
   RIGHT: def score_episode(ep) -> EpisodeMetrics: ...

3. One responsibility per class
   EpisodeScorer scores. ReportFormatter formats.
   Never: EpisodeScorerAndFormatterAndSaver

4. Make data immutable where possible
   Frozen dataclasses prevent accidental mutation.
   Bugs from accidental mutation are extremely hard to find.

5. Log at boundaries
   Log when episode starts, ends, and at each step.
   Logs are your only window into async code behavior.

6. Properties for computed values
   episode.success is a property (computed from outcome).
   Never store derived values — compute them.
   Stored derived values go stale.
```

---

## 12. How Each Concept Is Reinforced

```
async/await
  SimulationRunner.run_episode() is async
  AgentPolicy.choose_action() is async (for future VLM calls)
  SimulationSession is an async context manager (__aenter__, __aexit__)

State machines
  NotepadEnv, BrowserEnv, FormEnv are all state machines
  LoopState pattern from Assignment 4 — same concept, different domain

dataclasses + frozen
  EnvironmentState, Transition, StepResult, EpisodeStep, Episode
  frozen=True where data should never change

Enums
  TransitionOutcome, EpisodeOutcome (just like ActionType, ChangeType)

Context managers
  SimulationSession (__aenter__, __aexit__) — same pattern as ActionRecorder

JSONL + generators
  EpisodeStore.save() appends one line
  EpisodeStore.load() yields one Episode per line (generator)
  load_successful() filters the generator in one line

Logging
  Structured logs on every transition (episode_started, episode_step, etc.)
  Same structlog pattern throughout

Regex parsing
  VLMPolicy will parse model output — same as parse_action()

Pathlib
  EpisodeStore, SimulationSession — all paths via Path objects

Quality validation
  EpisodeScorer computes quality metrics (like TrajectoryAnalyzer)
  BatchMetrics aggregates across episodes (like QualityReport)

Serialization
  asdict() converts dataclasses → dicts for JSON
  Enum.value converts enum → string for JSON
  Reconstruction: Episode(**dict) rebuilds from JSON

Clean architecture
  Each layer knows nothing about layers above it
  Environments don't know about runners
  Runners don't know about scorers
  Scorers don't know about storage
```

---

## 13. Future Upgrade Ideas

```
Upgrade 1: Connect VLMPolicy to real Qwen model
  Replace the stub with actual LocalModelBackend.predict_action()
  The environment's describe() output IS the observation
  This turns VASH into a real agent evaluation tool

Upgrade 2: Parallel episode execution
  Create N env instances, run asyncio.gather() on all
  10x speedup for batch evaluation

Upgrade 3: Curriculum learning
  Start with easy environments (NotepadEnv)
  Progress to harder ones as success rate improves
  Automatic difficulty scaling

Upgrade 4: Self-play environments
  Two agents compete (one opens files, one deletes them)
  Adversarial evaluation reveals robustness

Upgrade 5: Export successful episodes as training data
  Successful Episode → RecordedStep (Assignment 5 format)
  Pipeline: Run VASH → filter successful → export as JSONL training data
  VASH becomes a synthetic data generator for Phase 16

Upgrade 6: Visual environments
  Replace text descriptions with actual screenshots
  Use PIL to render fake GUI screens
  Test OCR pipeline on synthetic images
```

---

## 14. Connection to Future VisionNav Phases

```
Phase 10 (Advanced Reasoning):
  VASH validates that dynamic planning works
  Run the HardcodedModel through environments
  Verify it correctly identifies DONE state

Phase 12 (Reinforcement Learning):
  VASH environments ARE the training environments for RL
  reward signal from environment IS the GRPO reward
  EpisodeStore saves successful trajectories for GRPO training

Phase 14 (Multi-Agent):
  VASH can run NavigatorAgent and FormAgent on different environments
  Measure: does specialist outperform generalist?

Phase 16 (Dataset Factory):
  Successful VASH episodes become synthetic training samples
  VASH generates 10,000 verified trajectories overnight
  Zero human annotation needed for verified correct paths

Phase 18 (Benchmarks):
  VASH environments ARE the VisionNav-Bench-1000
  Replace simulated with real screenshots for production eval
```

---

## What To Study After This Project

```
1. State machine design patterns (finite state automata)
   → Every complex system is secretly a state machine
   → Learn to see state machines in code you didn't write

2. Property-based testing (Hypothesis library)
   → Instead of specific test cases, generate thousands automatically
   → "For any sequence of valid actions, environment never crashes"

3. Async patterns deeper
   → asyncio.Queue (producer-consumer pattern)
   → asyncio.Semaphore (limit concurrent episodes)
   → These let you run 100 episodes at once safely

4. Python dataclass advanced features
   → __post_init__ for validation
   → field(init=False) for computed fields
   → ClassVar for class-level constants
```

---

## What Phase 16 Evolves Into

```
Phase 16 starts as: manual recording → JSONL → training

After VASH, Phase 16 becomes:

  1. Synthetic generation via VASH
     10,000 episodes/night, zero human time
     Perfect labels (we know exactly what happened)

  2. Human demonstration recording (Assignment 5 system)
     High-quality real examples
     Complex tasks that VASH can't simulate

  3. Real agent self-improvement (Phase 12 RL)
     Production trajectories → scored → added to dataset
     Dataset grows automatically from usage

  4. The pipeline becomes:
     VASH synthetic → formatter → validator → training
     Human demos   → formatter → validator → training
     RL trajectories → formatter → validator → training
     All three streams feed the same training pipeline
```

---

## Skills That Become Important After This Project

```
After VASH you will be ready for:

1. Distributed systems
   Running 10,000 VASH episodes needs a job queue
   Celery, Redis Queues — same patterns you already know, at scale

2. Model evaluation at scale
   Eval harnesses at Anthropic/OpenAI work exactly like VASH
   You now understand the architecture intuitively

3. Synthetic data generation
   VASH is a data generator
   The best datasets mix real + synthetic
   You now know how to build the synthetic side

4. Reinforcement learning environments
   VASH environments follow the Gym interface pattern
   OpenAI Gym, Gymnasium — VASH prepares you for this

5. MLOps and experiment tracking
   VASH results go into WandB
   Tracking experiments is the same as tracking episodes
```

---

**This is your assignment. Build VASH. Make it work. Make it clean. Then come back.**

When you submit — paste:
1. `NotepadEnv.step()` implementation
2. `EpisodeScorer.score_batch()` implementation
3. Test results from `pytest tests/ -v`
4. One batch report printed to terminal

Say **"VASH submitted"** when ready.