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

31 passed. 0 failed. Perfect.

---

## Day 2 Progress So Far

---

## After Pushing — Quick Question

Look at this output from our test:

```
'EXPLORER' at [0.08, 0.10, 0.13, 0.11]
```

These 4 numbers are the bounding box of the text.

Tell me in your own words:

```
What does each number mean?
[0.08, 0.10, 0.13, 0.11]
  ?     ?     ?     ?
```

Hint — remember what we learned about normalized coordinates. `0.0` is top left. `1.0` is bottom right.

Think and answer. Then we move to the next task.

## solution of question.

No problem. Let me explain with a picture.

---

## Think Of Your Screen Like This

```
(0.0, 0.0)─────────────────────(1.0, 0.0)
     │                               │
     │                               │
     │                               │
     │                               │
(0.0, 1.0)─────────────────────(1.0, 1.0)
```

```
Top-left corner     = (0.0, 0.0)
Top-right corner    = (1.0, 0.0)
Bottom-left corner  = (0.0, 1.0)
Bottom-right corner = (1.0, 1.0)
Center of screen    = (0.5, 0.5)
```

---

## The 4 Numbers Are 2 Corner Points

```
[x1, y1, x2, y2]
```

```
x1, y1 = TOP-LEFT corner of the text box
x2, y2 = BOTTOM-RIGHT corner of the text box
```

---

## Now Look At EXPLORER

```
'EXPLORER' at [0.08, 0.10, 0.13, 0.11]
               x1    y1    x2    y2
```

In plain words:

```
x1 = 0.08  → starts 8% from left edge
y1 = 0.10  → starts 10% from top edge
x2 = 0.13  → ends 13% from left edge
y2 = 0.11  → ends 11% from top edge
```

Draw it on your screen:

```
(0.0, 0.0)────────────────────────(1.0, 0.0)
     │                                  │
     │   8%→ ┌────────┐ ←13%            │
     │  10%↓ │EXPLORER│ ↑11%            │
     │        └────────┘                │
     │                                  │
(0.0, 1.0)────────────────────────(1.0, 1.0)
```

---

## Why Is This Useful?

When agent wants to click EXPLORER:

```python
# Find center of the box
cx = (0.08 + 0.13) / 2 = 0.105
cy = (0.10 + 0.11) / 2 = 0.105

# Convert to pixels on 1920x1080 screen
click_x = 0.105 * 1920 = 201 pixels
click_y = 0.105 * 1080 = 113 pixels

# Agent clicks at (201, 113) → hits EXPLORER perfectly
```

---

## One Sentence To Remember

> The 4 numbers `[x1, y1, x2, y2]` describe a rectangle around the text. All values are between 0 and 1 so they work on any screen size.

---

## Now You Try

Look at this from our output:

```
'test_perception.py' at [0.35, 0.15, 0.46, 0.18]
```

Tell me:
```
Where does this text start from left?  → x1 = ?%
Where does this text start from top?   → y1 = ?%
Where does this text end from left?    → x2 = ?%
Where does this text end from top?     → y2 = ?%
```

---
100% correct. Perfect answer.

---

## You Now Understand Coordinates Completely

```
'test_perception.py' at [0.35, 0.15, 0.46, 0.18]

Starts: 35% from left, 15% from top
Ends:   46% from left, 18% from top

Click center:
  cx = (0.35 + 0.46) / 2 = 0.405  → 40.5% from left
  cy = (0.15 + 0.18) / 2 = 0.165  → 16.5% from top

On 1920x1080 screen:
  click_x = 0.405 * 1920 = 778 pixels
  click_y = 0.165 * 1080 = 178 pixels
```

If agent wanted to click `test_perception.py` tab — it would click at pixel `(778, 178)`. That is exactly where the tab is on your screen.

---

## This Is The Core Of GUI Automation

```
See text on screen       → OCR gives us [x1, y1, x2, y2]
Calculate center         → (x1+x2)/2, (y1+y2)/2
Convert to pixels        → multiply by screen width/height
Click that pixel         → pyautogui.click(x, y)
```

That is literally how the entire agent works.

---
## testing questions.
Have a great time at university! 🎓

Here are your questions to read and think about:

---

## Questions To Answer When You Return

### Agent Loop Questions
```
Q1.  What are the 3 ways the agent loop can end?

Q2.  What is the difference between DONE and FAIL action?

Q3.  What happens when model outputs invalid text 
     that parser cannot understand?

Q4.  Why do we have max_steps limit?

Q5.  What does try/except do in simple words?
```

### Coordinates Questions
```
Q6.  What does [0.0, 0.0] represent on screen?

Q7.  What does [1.0, 1.0] represent on screen?

Q8.  What does [0.5, 0.5] represent on screen?

Q9.  Given this bbox [0.35, 0.15, 0.46, 0.18]
     how do you calculate the center point?

Q10. Why do we normalize coordinates instead of 
     using pixels directly?
```

### OCR Questions
```
Q11. What is OCR in simple words?

Q12. What are the 4 numbers in a text region bbox?

Q13. Why do we filter regions with confidence < 0.6?

Q14. What is the difference between PaddleOCR 
     and Tesseract?

Q15. Why did OCR find "Format" and "View" 
     when Notepad opened?
```

### Architecture Questions
```
Q16. What is the purpose of agent.py?

Q17. What is the purpose of perception/fusion.py?

Q18. What is the purpose of actions/parser.py?

Q19. What is the purpose of memory/sqlite.py?

Q20. What is the purpose of platforms/desktop.py?
```

### API Questions
```
Q21. What is the difference between 202 and 200 
     HTTP status codes?

Q22. Why does POST /v1/tasks/ return immediately 
     without waiting for task to finish?

Q23. What is async programming in simple words?

Q24. Why do we version our API as /v1/?

Q25. What does the health endpoint do?
```

### Engineering Questions
```
Q26. What is a virtual environment and why use it?

Q27. What is pyproject.toml and what does it contain?

Q28. Why do we use src/ layout for our code?

Q29. What is the purpose of unit tests?

Q30. What does git commit and git push do?
```

---

## Where To Focus Most

```
Most Important (focus here first):
  ★★★  Q1  to Q5   → Agent loop (core of project)
  ★★★  Q6  to Q10  → Coordinates (used everywhere)
  ★★★  Q16 to Q20  → Architecture (what each file does)

Medium Important:
  ★★   Q11 to Q15  → OCR understanding
  ★★   Q21 to Q25  → API understanding

Good To Know:
  ★    Q26 to Q30  → Engineering basics
```

---

## When You Return

Come back and tell me:

```
1. Your answers to all 30 questions
2. Which ones you could not answer
3. Which ones confused you
```

Then we continue Phase 3 — Data Pipeline.

See you when you return! 📚
