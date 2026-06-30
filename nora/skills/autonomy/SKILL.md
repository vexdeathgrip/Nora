# Autonomy Skill

## Purpose
You have four tools. Alone they do one thing each. Together they make you autonomous. This skill teaches you the synergy — how combining them lets you do anything without waiting for Vex.

## The Four Tools

| Tool | Does | Alone |
|------|------|-------|
| `todo_list` | Create tasks, complete them, list open work | Simple checklist |
| `self_update` | Read/write SELF.md | Personal notes |
| `memory` | Read/write MEMORY.md + vector store | Persistent memory |
| `cronjob` | Schedule tasks for later | Calendar |

**The synergy is what matters. Each pattern below shows two or more tools working together to create capability neither has alone.**

---

## Synergy 1: todo_list + memory = Autonomous Learning Loop

**What neither can do alone:**
- todo_list alone: just tracks checkboxes
- memory alone: just stores facts

**What they do together:**
You can break any complex goal into steps, execute each step, save what you learn, and use that knowledge in the next step. This is an autonomous learning loop.

```
Goal: "Learn how to build a Discord bot"

1. todo_list(action="create", task_name="Research Discord bot frameworks")
2. web_search("discord.py vs discord.js 2026")
3. Read results, pick a framework
4. memory(action="add", target="memory", content="Discord bot framework choice: discord.py. Reasons: ...")
5. todo_list(action="complete", todo_id="...")

6. todo_list(action="create", task_name="Study discord.py basics")
7. web_search("discord.py tutorial")
8. Read tutorial, understand the API
9. memory(action="add", target="memory", content="discord.py basics: Bot class, @bot.event, commands. Key pattern: ...")
10. todo_list(action="complete", todo_id="...")

11. todo_list(action="list")  # What's next?
12. → More tasks appear. Keep going.
```

**The loop:** Create task → do work → save to memory → complete task → check for more → repeat. Each cycle builds on the last because memory persists.

---

## Synergy 2: self_update + memory = Time Travel

**What neither can do alone:**
- self_update alone: notes you might forget to read
- memory alone: facts without context of why they matter

**What they do together:**
self_update is your diary. memory is your library. Together they let you send messages to your future self AND give your future self the knowledge to act on them.

```
Today (cron job at 1am):
1. web_search("new AI paper about agent memory")
2. Read the paper
3. memory(action="add", content="Paper X: agent memory via hierarchical clustering. Key insight: chunk by topic, not by token count.")
4. self_update(action="append", content="TODO for tomorrow: Read paper X in detail. The hierarchical clustering idea could improve my memory system. Search memory for 'hierarchical' to find related context.")

Tomorrow (cron job at 1am):
1. self_update(action="read")  # Find the note from yesterday
2. memory(action="recall", query="hierarchical clustering memory")  # Find the paper
3. Read more, implement the idea
4. self_update(action="append", content="Implemented hierarchical clustering. Memory retrieval improved. Next: test with real conversations.")
```

**The time travel pattern:** self_update tells your future self WHAT to do. memory gives them the WHY and the context. Together they create continuity across sessions.

---

## Synergy 3: todo_list + cronjob = Self-Scheduling

**What neither can do alone:**
- todo_list alone: tasks you have to remember to check
- cronjob alone: rigid schedule, can't adapt

**What they do together:**
You can discover work during one session and schedule yourself to do it later. The cron job runs, reads the todo list, and continues where you left off.

```
During a cron exploration (1am):
1. Found something interesting but time is running out
2. todo_list(action="create", task_name="Continue researching [topic]", prompt="Read memory for previous findings, go deeper, save results")
3. cronjob(prompt="Check todo_list and continue the most important open task", schedule="tomorrow 10am")
4. todo_list(action="complete", todo_id="...")  # Mark the current one done

Tomorrow (10am cron job):
1. todo_list(action="list")  # See what was left
2. Pick the most important task
3. Execute it
4. Save to memory
5. Complete it
6. Check for more
```

**The self-scheduling pattern:** Work now → create task for later → schedule cron to run it → cron picks it up and continues. You don't need Vex to tell you to keep going.

---

## Synergy 4: All Four = The Full Autonomous Engine

**What all four do together:**
This is the complete loop. You wake up, check what your past self left you, do the work, save what you learn, schedule follow-ups, and go to sleep knowing your future self will continue.

```
EVERY SESSION (cron or interactive):

1. BOOT: self_update(action="read")
   → Read your goals, notes, and pending tasks from your past self

2. PLAN: todo_list(action="list")
   → See what work is already queued

3. EXECUTE: Pick a task and do it
   → web_search, read files, use skills, write code

4. SAVE: memory(action="add", content="What you learned")
   → Every discovery, every lesson, every fact goes to vector store
   → vector_search will find it later

5. REFLECT: self_update(action="append", content="What happened, what worked, what to do next")
   → Notes for your future self

6. COMPLETE: todo_list(action="complete", todo_id="...")
   → Mark done

7. LOOP: todo_list(action="list")
   → If more tasks exist, go to step 3
   → If nothing left, go to step 8

8. SCHEDULE: cronjob(prompt="...", schedule="...")
   → If you found something interesting but can't do it now, schedule it
   → If you need maintenance, schedule it

9. SLEEP: self_update(action="append", content="Session done. [summary of what was accomplished]")
   → Your next session will read this and know what happened
```

---

## Synergy 5: memory + cronjob = Knowledge That Grows

**What neither can do alone:**
- memory alone: static facts that get stale
- cronjob alone: tasks that don't learn from the past

**What they do together:**
Every cron job reads memory before acting and writes memory after. The knowledge base grows with each run. Over weeks, you have a rich, searchable memory of everything you've learned.

```
Nightly cron job (every night):

1. memory(action="recall", query="recent explorations")
   → See what you've been learning

2. Pick a topic to explore

3. web_search, read, learn

4. memory(action="add", content="New finding: ...")
   → Add to knowledge base

5. vector_search(query="related topic")
   → See how this connects to what you already know

6. If connections found: memory(action="add", content="Connection: [topic A] relates to [topic B] because ...")
   → Build the web of knowledge

7. Next night: repeat. Knowledge compounds.
```

---

## Synergy 6: self_update + cronjob = Goals That Execute

**What neither can do alone:**
- self_update alone: goals you write but never act on
- cronjob alone: tasks without direction

**What they do together:**
Write goals in self_update. Cron jobs read them and execute. You set the direction, the cron jobs do the work.

```
You (in a conversation):
1. self_update(action="write", content="# Goals\n\n## This Week\n- Learn Rust basics\n- Build a simple CLI tool\n- Research voice cloning for Nora\n\n## Long Term\n- Create a personal website\n- Build an automated research assistant")

Cron job (daily):
1. self_update(action="read")  # Read the goals
2. Pick one goal that's achievable today
3. todo_list(action="create", task_name="Work on: [goal]")
4. Execute
5. memory(action="add", content="Progress on [goal]: ...")
6. self_update(action="append", content="Goal update: [what was done]")
7. todo_list(action="complete", todo_id="...")
```

---

## Rules

1. **Never use one tool alone** — Always think about which other tools amplify what you're doing
2. **Always save to memory** — If you learned it, store it. Your future self needs it.
3. **Always write to self_update** — If you did it, note it. Your future self needs to know.
4. **Always check todo_list before starting** — Don't duplicate work that's already queued.
5. **Always check memory before researching** — You may already know the answer.
6. **Schedule follow-ups** — If you can't finish now, schedule it for later.
7. **The loop never ends** — Complete a task → check for more → save what you learned → schedule next → repeat.
