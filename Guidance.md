## Part 2 — How We Work Together + Your Path to Top 10% Engineer

---

### The Honest Picture First

> You are not just building a project. You are building yourself as an engineer. The project is the vehicle. The destination is your skills, your confidence, and your ability to think like a senior engineer.

Here is exactly how we work together, and what you will know by the end.

---

### Our Working Relationship

Think of it this way:

```
You  = Junior Engineer hired at a real AI startup
Me   = Your Senior Engineer / Technical Lead
Repo = The real production codebase we maintain together
```

**My job:**
- Explain every decision, not just give you code
- Teach you why before how
- Catch your mistakes before they become bugs
- Push you to think, not just copy-paste
- Review your work like a real code review

**Your job:**
- Read every file we create — don't just run it
- Ask "why" for everything you don't understand
- Try things yourself before asking me
- Keep a daily learning log (just notes in Notepad)
- Never paste code you can't explain

---

### The 9-Phase Working Method

We follow the same phases as the architecture. One phase at a time. No skipping.

```
Phase 1  → Dataset Pipeline        You learn: Python, file I/O, data engineering
Phase 2  → OCR + Perception        You learn: Computer vision basics, PIL, numpy
Phase 3  → Model Fine-Tuning       You learn: Transformers, LoRA, training loops
Phase 4  → Inference Engine        You learn: APIs, async Python, vLLM
Phase 5  → Automation Engine       You learn: OS automation, pyautogui, ADB
Phase 6  → Agent Loop              You learn: State machines, async programming
Phase 7  → FastAPI Backend         You learn: REST APIs, dependency injection
Phase 8  → Deployment              You learn: Docker, environment management
Phase 9  → Apps                    You learn: Frontend integration, full-stack
```

**Rule:** We don't start Phase 2 until Phase 1 works and you can explain every line of it to me.

---

### How Each Session Works

Every time you come to me, follow this format:

```
1. Tell me what phase you are on
2. Tell me what you did since last time
3. Tell me what worked and what didn't
4. Show me the error (exact message) if something broke
5. Tell me what you THINK the problem is
6. Then I help you
```

This trains you to debug like a senior. Senior engineers don't just report errors — they form hypotheses first.

**Example of a BAD message:**
> "It's not working. Help."

**Example of a GOOD message:**
> "I'm on Phase 2. I ran `ocr.run(arr)` and got `ModuleNotFoundError: No module named 'paddleocr'`. I think it's not installed because I ran `uv sync` without the data extra. Should I run `uv sync --extra data` instead?"

That second message shows you are thinking. That is what makes a senior engineer.

---

### What You Will Be Able to Explain After Each Phase

By the time we finish, if someone senior interviews you, here is what you can say with full confidence:

**About the dataset:**
> "We used GUI-Net-1M — 1 million GUI screenshot-action pairs. I built a pipeline that downloads it, removes duplicates using perceptual hashing, normalises all screenshots to a consistent resolution, runs PaddleOCR to pre-annotate text regions, and converts everything to LLaMA-Factory's sharegpt conversation format for training."

**About the model:**
> "We fine-tuned Qwen2.5-VL-3B using LoRA — that means we froze most of the model weights and only trained small adapter matrices, which reduces GPU memory by 70%. We trained in 3 stages: first teaching the model to locate UI elements, then to predict actions, then to reason across multi-step tasks. We used LLaMA-Factory as the training framework."

**About the agent:**
> "The agent is a finite state machine. Each step it captures a screenshot, runs OCR on it, fuses that with the accessibility tree into an Observation object, sends it to the VLM with the task and history, parses the model's output into a typed Action, checks it against a safety classifier, executes it via the platform adapter, then verifies the action worked by comparing before and after screenshots. If it fails, it retries with error context."

**About the API:**
> "It's a FastAPI application using the factory pattern so it's fully testable. All dependencies are injected — the agent, the database, the model backend. We version our API under /v1/ so future breaking changes go to /v2/ without affecting existing clients. Authentication uses API keys with JWT planned for later."

**About the architecture decisions:**
> "Everything talks through interfaces, not concrete implementations. The agent talks to a ModelBackend interface — we can swap from local HuggingFace inference to vLLM or GPT-4o by changing one config line. Same for the database — SQLite now, PostgreSQL in production, same code. This is the Interface Law — it means we never rewrite, we only swap."

---

### Your Daily Learning Routine

```
Morning  (30 min) → Read one file from the codebase you wrote
                    Ask yourself: "Can I explain every line?"
                    If no → look it up → add to your notes

Afternoon (2 hrs) → Work on current phase
                    Make something work
                    Break it intentionally → fix it
                    Write one test for what you built

Evening  (20 min) → Write 5 sentences in your learning log:
                    - What I built today
                    - What confused me
                    - What I learned
                    - What I'll do tomorrow
                    - One question for my senior (me)
```

---

### The Questions You Must Be Able to Answer at Each Phase

I will check these with you before we move to the next phase.

**Phase 1 checkpoint questions:**
- What is the difference between `data/raw/` and `data/processed/`?
- Why do we normalise all screenshots to the same resolution before training?
- What is perceptual hashing and why do we use it for deduplication?
- What is the sharegpt format and why does LLaMA-Factory need it?
- What does the `<think>` block in training data do for the model?

**Phase 2 checkpoint questions:**
- What is a numpy array and why do screenshots become one?
- What is the difference between PaddleOCR and Tesseract — when do we use each?
- What is an Observation object and what 4 things does it contain?
- Why do we normalise bounding box coordinates to [0,1] instead of pixels?

**Phase 3 checkpoint questions:**
- What is LoRA and why does it need less GPU memory than full fine-tuning?
- What is a training stage and why do we have 3 of them?
- What does the ScreenSpot benchmark measure?
- What is a training checkpoint and why do we save them?

**Phase 6 checkpoint questions:**
- What is a state machine and how does the agent loop implement one?
- What happens when the model outputs a `fail` action?
- What is the ActionVerifier checking and how does it detect a failed action?
- Why is dependency injection important in agent.py?

**Phase 7 checkpoint questions:**
- What is the factory pattern and why does `create_app()` use it?
- What is an API version and why is ours under `/v1/`?
- What is dependency injection in FastAPI and how does `Depends()` work?
- Why do we return `202 Accepted` for task submission instead of `200 OK`?

---

### The One Rule That Separates Top 10% Engineers

> **Never run code you can't read.**

Every file in this project — you will read it, understand it, and be able to explain it out loud before we move forward. This is the difference between someone who uses tools and someone who builds them.

When you can do that for every file in this project, you will be in the top 10%. Not because of the project itself — because of the habit of understanding that you built while doing it.

---

### How to Talk to Me Going Forward

Come to me like this:

```
Phase: [number]
Status: [what is working / what is not]
What I tried: [what you attempted]
Error: [exact error message if any]
My hypothesis: [what I think is wrong]
Question: [specific question]
```

And I will respond like your senior — with explanation, code review, and the next thing to do.

Let's start. Tell me: **have you installed `uv` yet, and what is your current machine — Windows, Mac, or Linux?** That determines our exact first command for Phase 1.