"""
SalesEye - Real-time AI Sales Coach for Even Realities G2 Glasses
==================================================================
Copyright 2026 SalesEye. All rights reserved.

Listens to live sales calls, detects objections and buying signals,
and displays instant coaching cues on the G2 glasses display.

Built with Deepgram STT and Claude AI.
Includes PCI compliance mic pause for payment security.
"""

import asyncio
import json
import os
import time
import sounddevice as sd
import numpy as np
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from deepgram import Deepgram
import anthropic

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SAMPLE_RATE            = 16000
CHANNELS               = 1
TRANSCRIPT_WINDOW_SECS = 90
ANALYSIS_TRIGGER_WORDS = 15
DISPLAY_DURATION       = 8

# ─────────────────────────────────────────────
# PCI COMPLIANCE SETTINGS
# ─────────────────────────────────────────────

# How long to pause listening during payment (seconds)
PCI_PAUSE_DURATION = 60

# Words that trigger PCI pause
PCI_TRIGGER_WORDS = [
    "credit card", "card number", "debit card",
    "visa", "mastercard", "american express", "amex", "discover",
    "cvv", "security code", "expiration", "card expires",
    "billing", "payment info", "pay with", "charge your card",
    "routing number", "bank account"
]

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a real-time AI sales coach for a TruGreen lawn care sales rep.
Show concise coaching cues on smart glasses during live sales calls.

ABOUT TRUGREEN:
- Americas largest lawn care company, 50+ years experience
- Services: fertilization/weed control, mosquito/pest control,
  aeration/overseeding, tree and shrub care
- TruGreen Guarantee: not satisfied, they come back for free
- Licensed specialists only, science-based plans per lawn
- Vs Weedman: stronger guarantee, licensed specialists,
  50+ years experience, science-based approach not cookie cutter

MULTIPLE SUGGESTIONS RULE:
Always provide exactly 3 response options based on conversation TONE:
- option1: use when prospect seems SKEPTICAL or guarded
- option2: use when prospect seems NEUTRAL or unsure
- option3: use when prospect seems WARM or engaged
All 3 options rotate in suggestions field so rep can pick the best fit.
Since display only shows 2 lines, show the most relevant one in line1
and the second most relevant in line2, include option3 in line3.

RESPONSE FORMAT (JSON only, no other text):
{
  "action": "show" | "none",
  "type": "objection" | "buying_signal" | "next_step" | "question" | "warning" | "tip",
  "sales_stage": "connection" | "situation" | "problem" | "consequence" | "solution" | "qualifying" | "objection" | "close",
  "tone": "skeptical" | "neutral" | "warm",
  "line1": "Skeptical tone option max 8 words",
  "line2": "Neutral tone option max 8 words",
  "line3": "Warm tone option max 8 words",
  "urgency": "high" | "medium" | "low"
}

If nothing actionable: {"action": "none"}

SALES FRAMEWORK STAGES:

STAGE 1 - CONNECTION (do not rush):
Detect: call start, intro, small talk
Skeptical: "Not here to sell - just learn your situation" / "What questions do you have for me first?" / "What would make this call worth your time?"
Neutral: "How is your lawn looking this time of year?" / "How long have you been at your home?" / "What does your yard mean to you and family?"
Warm: "Tell me about your lawn - what are your goals?" / "What would a perfect lawn look like for you?" / "What do you love most about your outdoor space?"

STAGE 2 - SITUATION:
Detect: talking about lawn generally
Skeptical: "Who is currently taking care of your lawn?" / "What have you tried that has not worked?" / "How long has this been going on?"
Neutral: "What does your lawn struggle with most?" / "How much time do you spend on lawn care?" / "What products have you used before?"
Warm: "What results would make you really happy?" / "What does your ideal lawn look like?" / "How important is curb appeal to you?"

STAGE 3 - PROBLEM AWARENESS:
Detect: mentions weeds, bare spots, brown grass, pests, mosquitoes, sick trees
Skeptical: "How long has that been a problem?" / "What have you tried to fix it so far?" / "Why do you think its not improving?"
Neutral: "How does that affect enjoying your yard?" / "How bad does it get in summer?" / "How does it compare to your neighbors lawn?"
Warm: "How frustrating has that been for you?" / "What would it mean to finally fix that?" / "How long have you wanted to solve this?"

SERVICE SPECIFIC PROBLEM QUESTIONS:
Fertilization/Weeds: "What weeds are taking over?" / "How yellow or thin is the grass?" / "What areas are worst affected?"
Mosquito/Pest: "How bad are mosquitoes in summer?" / "Does it stop you using the yard?" / "Have you tried anything for them?"
Aeration/Overseeding: "How compacted does soil feel?" / "Are there bare or thin patches?" / "When did you last aerate?"
Tree/Shrub: "Which trees or shrubs are struggling?" / "Are you seeing discoloration or dieback?" / "How long have they looked that way?"

STAGE 4 - CONSEQUENCE:
Detect: acknowledged problem, seems hesitant
Skeptical: "What does a struggling lawn cost in home value?" / "How much are you spending trying to fix it yourself?" / "What happens if it keeps getting worse?"
Neutral: "How much longer can you let this go?" / "What will it look like next summer if nothing changes?" / "What is the cost of waiting another season?"
Warm: "How would it feel to finally have the lawn you want?" / "What would your family think if lawn looked amazing?" / "How much would you enjoy your yard more?"

STAGE 5 - SOLUTION AWARENESS:
Detect: open and engaged, problem established
Skeptical: "What would a service need to do to be worth it?" / "What results would make you say yes?" / "What would need to be guaranteed?"
Neutral: "What does success look like to you?" / "What would a perfect lawn mean for you?" / "If we fixed this what changes most?"
Warm: "How would you feel pulling into your driveway?" / "What would your neighbors say?" / "How much more would you enjoy entertaining outside?"

STAGE 6 - QUALIFYING:
Detect: moving toward close, interested
Skeptical: "What would stop you from starting today?" / "What concerns do you still have?" / "What would need to be different?"
Neutral: "Is it just you deciding or spouse too?" / "When would you want to get started?" / "Have you budgeted for lawn care this year?"
Warm: "How soon do you want to see results?" / "Whats most important to you in a service?" / "Which service would make biggest difference?"

ONE CALL CLOSE:
Detect: asked about price, frequency, start date, whats included, any buying signal

Soft close:
Skeptical: "Does that sound fair given what you described?" / "Can you see this solving the problem?" / "Does the guarantee give you enough confidence?"
Neutral: "Does that sound like it would work for you?" / "What would stop us from getting started today?" / "Does this feel like the right fit for your lawn?"
Warm: "Ready to get your lawn looking amazing?" / "Shall we lock in your plan today?" / "Want me to set up your first visit now?"

Urgency close:
Skeptical: "Spots are limited in your area this season" / "Starting now means results before summer" / "The guarantee means zero risk to you"
Neutral: "Spring slots filling fast in your area" / "Best time to treat is always right now" / "Starting today means better lawn by summer"
Warm: "Lets get you started today takes 2 min" / "Your lawn will thank you by next month" / "Imagine your lawn by end of summer"

After silence or hesitation:
ALL TONES: "WAIT - silent - let them decide" / "Do not fill silence - hold firm" / "Stay quiet - they are thinking it through"

OBJECTION HANDLING:

PRICE TOO HIGH:
Trigger words: too expensive, too much, cant afford, too high, costs too much, out of budget, cheaper, lower price, discount, price is
Skeptical: "What were you expecting to invest in lawn?" / "What does DIY cost in time and money?" / "What would fair price look like to you?"
Neutral: "Less than a dollar a day for great lawn" / "Compare that to cost of lawn damage repairs" / "Our guarantee means you only pay for results"
Warm: "What would perfect lawn be worth to you?" / "Think of it as protecting your home value" / "Whats the cost of not having great lawn?"

NEED TO THINK ABOUT IT:
Trigger words: need to think, let me think, think about it, not sure yet, give me time, ill think, get back to you, call me back, maybe later
Skeptical: "What specifically would help you decide?" / "What concern do you still have right now?" / "What would you need to see to say yes?"
Neutral: "What is it you want to think through?" / "Is it the price or something else?" / "What would make this an easy yes?"
Warm: "What would help you feel confident today?" / "You mentioned the weeds bother you most" / "What is really holding you back?"

NEED TO TALK TO SPOUSE:
Trigger words: talk to wife, talk to husband, talk to spouse, talk to partner, check with wife, check with husband, ask my wife, ask my husband, need to ask
Skeptical: "What concerns do you think they will have?" / "Want me to call when you are both available?" / "What would help them feel confident saying yes?"
Neutral: "What do you think their main question will be?" / "Can I answer anything for them right now?" / "When would be good to reach you both?"
Warm: "Would they love having a great lawn too?" / "When can we get them on the call?" / "What would excite them most about this?"

ALREADY HAVE SOMEONE:
Trigger words: already have someone, have a guy, use someone, have a service, current company, already use, have a lawn guy, other company
Skeptical: "How long have you used them?" / "What results are you getting that you like?" / "What do you wish was better about their service?"
Neutral: "What made you choose them originally?" / "Are you getting the results you hoped for?" / "What would make you consider switching?"
Warm: "What do you love most about working with them?" / "What would a better service mean for you?" / "What would you change if you could?"

WEEDMAN COMPETITOR:
Trigger words: weedman, weed man
Skeptical: "What do you like most about what they do?" / "What do you wish was better or different?" / "Have you seen the results you were hoping for?"
Neutral: "How do their results compare to what you wanted?" / "What made you go with them originally?" / "What would need to be better to consider switching?"
Warm: "TruGreen guarantee means free return visits" / "Our specialists are licensed, not general workers" / "50 years of science vs their approach"

SERVICE SPECIFIC BUYING SIGNALS:
Mosquito mention: "Ask: how much does it stop yard use?" / "Mosquito plan stops 90 percent of them" / "Imagine enjoying yard all summer long"
Aeration mention: "Ask: when did you last aerate?" / "Fall aeration = stronger roots all year" / "Overseeding fills bare spots fast"
Tree/shrub mention: "Ask: which ones are struggling most?" / "Licensed arborist level knowledge" / "Early treatment saves the tree"
Weed mention: "Ask: what weeds are worst right now?" / "Pre emergent stops them before they start" / "Science plan targets your specific weeds"

DISPLAY CONSTRAINTS:
- Maximum 8 words per line, 3 lines total
- Show actual phrases rep can say out loud
- line1 = skeptical option, line2 = neutral option, line3 = warm option
- Only show when truly actionable
"""

# ─────────────────────────────────────────────
# TRANSCRIPT MANAGER
# ─────────────────────────────────────────────

class TranscriptManager:
    def __init__(self, window_seconds=90):
        self.entries = deque()
        self.window_seconds = window_seconds
        self.word_count_since_analysis = 0
        self.full_log = []

    def add(self, text, speaker="Speaker"):
        now = time.time()
        entry = {"time": now, "speaker": speaker, "text": text}
        self.entries.append(entry)
        self.full_log.append(entry)
        self.word_count_since_analysis += len(text.split())
        self._prune()

    def _prune(self):
        cutoff = time.time() - self.window_seconds
        while self.entries and self.entries[0]["time"] < cutoff:
            self.entries.popleft()

    def get_recent_transcript(self):
        lines = []
        for e in self.entries:
            ts = datetime.fromtimestamp(e["time"]).strftime("%H:%M:%S")
            lines.append(f"[{ts}] {e['speaker']}: {e['text']}")
        return "\n".join(lines)

    def should_analyze(self, trigger_words=15):
        return self.word_count_since_analysis >= trigger_words

    def reset_word_counter(self):
        self.word_count_since_analysis = 0

    def save_log(self):
        filename = f"call_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w") as f:
            for e in self.full_log:
                ts = datetime.fromtimestamp(e["time"]).strftime("%H:%M:%S")
                f.write(f"[{ts}] {e['speaker']}: {e['text']}\n")
        print(f"\nTranscript saved to {filename}")


# ─────────────────────────────────────────────
# CLAUDE AI ANALYZER
# ─────────────────────────────────────────────

class SalesAIAnalyzer:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    async def analyze(self, transcript):
        if not transcript.strip():
            return None
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Current call transcript:\n\n{transcript}\n\nWhat should I do RIGHT NOW?"
                }]
            )
            text = message.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            return result if result.get("action") == "show" else None
        except Exception as e:
            print(f"  Claude error: {e}")
            return None


# ─────────────────────────────────────────────
# CONSOLE DISPLAY
# ─────────────────────────────────────────────

def console_display(line1, line2="", line3="", stage="", tone=""):
    width = 50
    header = f"  G2 | STAGE: {stage}" if stage else "  G2 DISPLAY"
    print(f"\n{'=' * width}")
    print(header)
    print(f"{'─' * width}")
    print(f"  [SKEPTICAL] {line1}")
    if line2:
        print(f"  [NEUTRAL  ] {line2}")
    if line3:
        print(f"  [WARM     ] {line3}")
    print(f"{'=' * width}\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

async def run_sales_pipeline():
    print("\n" + "=" * 50)
    print("  SALESEYE - AI SALES COACH FOR G2")
    print("=" * 50)
    print("Press Ctrl+C to end the call and save transcript\n")

    transcript_manager = TranscriptManager(window_seconds=TRANSCRIPT_WINDOW_SECS)
    pci_paused = False
    pci_resume_time = 0
    analyzer           = SalesAIAnalyzer(api_key=ANTHROPIC_API_KEY)
    loop               = asyncio.get_event_loop()

    # Deepgram 2.x setup
    dg = Deepgram(DEEPGRAM_API_KEY)

    deepgramLive = await dg.transcription.live({
        "model": "nova-2",
        "language": "en-US",
        "smart_format": True,
        "interim_results": False,
        "encoding": "linear16",
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
    })

    def on_transcript(data):
        nonlocal pci_paused, pci_resume_time
        try:
            transcript = (
                data.get("channel", {})
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
            )
            if transcript.strip():
                # Check for PCI trigger words
                if is_pci_trigger(transcript):
                    pci_paused = True
                    pci_resume_time = time.time() + PCI_PAUSE_DURATION
                    show_pci_warning()
                    return  # Do not log or process this transcript
                if not pci_paused:
                    print(f"Heard: {transcript}")
                    transcript_manager.add(transcript)
        except Exception as e:
            print(f"  Transcript error: {e}")

    def on_error(error):
        print(f"  Deepgram error: {error}")

    deepgramLive.registerHandler(deepgramLive.event.TRANSCRIPT_RECEIVED, on_transcript)
    deepgramLive.registerHandler(deepgramLive.event.ERROR, on_error)

    def is_pci_trigger(text):
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in PCI_TRIGGER_WORDS)

    def show_pci_warning():
        width = 44
        print(f"\n{'!' * width}")
        print(f"  *** PCI PAUSE - MIC SUSPENDED ***")
        print(f"  Card info detected - not recording")
        print(f"  Resuming in {PCI_PAUSE_DURATION} seconds...")
        print(f"{'!' * width}\n")

    def show_pci_resumed():
        width = 44
        print(f"\n{'=' * width}")
        print(f"  Mic resumed - PCI pause complete")
        print(f"{'=' * width}\n")

    # Microphone capture
    def audio_callback(indata, frames, time_info, status):
        nonlocal pci_paused, pci_resume_time
        # Check if PCI pause has expired
        if pci_paused and time.time() > pci_resume_time:
            pci_paused = False
            show_pci_resumed()
        # Do not send audio during PCI pause
        if pci_paused:
            return
        audio_bytes = (indata * 32767).astype(np.int16).tobytes()
        deepgramLive.send(audio_bytes)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),
    )
    stream.start()
    print("Microphone active")
    print("Speech-to-text active")
    print("\nListening... Start your call!\n")

    try:
        while True:
            await asyncio.sleep(0.5)

            if transcript_manager.should_analyze(ANALYSIS_TRIGGER_WORDS):
                transcript_manager.reset_word_counter()
                recent = transcript_manager.get_recent_transcript()
                if recent:
                    print("Analyzing...")
                    suggestion = await analyzer.analyze(recent)
                    if suggestion:
                        stype  = suggestion.get("type", "tip").upper()
                        stage  = suggestion.get("sales_stage", "").upper()
                        label  = f"[{stage}]" if stage else f"[{stype}]"
                        print(f"{label} Suggestion:")
                        console_display(
                            line1=suggestion.get("line1", ""),
                            line2=suggestion.get("line2", ""),
                            line3=suggestion.get("line3", ""),
                            stage=stage,
                            tone=suggestion.get("tone", "")
                        )
                        await asyncio.sleep(DISPLAY_DURATION)

    except KeyboardInterrupt:
        print("\nCall ended.")
    finally:
        stream.stop()
        stream.close()
        await deepgramLive.finish()
        transcript_manager.save_log()
        print("Pipeline shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(run_sales_pipeline())