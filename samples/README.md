# Demo sample

`fake_meeting.mp4` — a 1-minute mock product sync used by the **🎬 Try a sample meeting** button in the live demo.

## What's in it

A scripted, TTS-narrated team meeting featuring named attendees (Priya, Rahul, Anjali) and deliberately seeded with:

| What | Example |
|---|---|
| **Action items** | Rahul → QA pass by Friday · Anjali → marketing v2 by Wed · Priya → churn dashboard |
| **Decisions**    | Go with concept #2 for the launch campaign · Delay the pricing change |
| **Open questions** | Annual discount? Grandfather existing customers? |

The exact spoken script is in [`fake_meeting.txt`](./fake_meeting.txt) — use it to grade the LLM's outputs (titles, summary, extraction, RAG answers) against ground truth.

## How it was built

```bash
# 1. TTS via macOS
say -v Samantha -r 175 -o fake_meeting.aiff -f fake_meeting.txt

# 2. Convert to mp3
ffmpeg -i fake_meeting.aiff fake_meeting.mp3

# 3. Render a title card (PIL) and mux with the audio
ffmpeg -loop 1 -i title_card.png -i fake_meeting.mp3 \
       -c:v libx264 -tune stillimage -pix_fmt yuv420p \
       -c:a aac -b:a 128k -shortest fake_meeting.mp4
```

Want a different sample? Edit `fake_meeting.txt`, re-run the three commands above, and the new file will replace the demo (the app picks up whatever's at `samples/fake_meeting.mp4`).
