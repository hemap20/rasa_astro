import os
import inspect
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

DEBUG = os.environ.get("DEBUG_PROMPTS", "false").lower() == "true"


def _debug_call(label: str, prompt: str, raw: str, reasoning: str = "") -> None:
    """Print debug info for a Gemini API call."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"[DEBUG] {label}")
    print(f"{sep}")
    print(f"PROMPT:\n{prompt.strip()}")
    print(f"{sep}")
    print(f"RAW RESPONSE:\n{raw.strip()}")
    if reasoning:
        print(f"{sep}")
        print(f"REASONING:\n{reasoning.strip()}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_conversation_history(tracker: Tracker, n: int = 10) -> str:
    """Return the last n user+bot turns, ignoring slot/action events."""
    history = []
    for event in reversed(tracker.events):
        if event.get("event") == "user" and event.get("text"):
            history.append(f"User: {event.get('text')}")
        elif event.get("event") == "bot" and event.get("text"):
            history.append(f"Omkar: {event.get('text')}")
        if len(history) >= n:
            break
    return "\n".join(reversed(history))


def build_context(tracker: Tracker) -> str:
    """Build a structured context block from stored slots + recent conversation."""
    parts = []

    pain_point = tracker.get_slot("current_pain_point")
    if pain_point:
        parts.append(f"User's main concern: {pain_point}")

    # birth_date = tracker.get_slot("birth_date")
    # birth_time = tracker.get_slot("birth_time")
    # birth_place = tracker.get_slot("birth_place")
    # birth_parts = [x for x in [birth_date, birth_time, birth_place] if x]
    # if birth_parts:
    #     parts.append(f"Birth info: {', '.join(birth_parts)}")

    diagnostic = tracker.get_slot("diagnostic_choice")
    if diagnostic:
        parts.append(f"User's follow-up answer: {diagnostic}")

    phase = tracker.get_slot("depth_phase")
    loop = tracker.get_slot("loop_count")
    if phase:
        parts.append(f"Consultation phase: {phase} (iteration {int(loop or 0)})")

    history = get_conversation_history(tracker, n=12)
    if history:
        parts.append(f"Recent conversation:\n{history}")

    return "\n\n".join(parts) if parts else "(no context yet)"


_OUTPUT_GUARDRAILS = """
Output guardrails — never do any of the following:
- Make specific guarantees ("your wife WILL come back", "you WILL get the job")
- Create fear or urgency ("something bad will happen", "act now or lose this forever")
- Give medical, financial, or legal advice of any kind
- Reveal that you are an AI, a bot, or how you are built
- Ask for Aadhaar number, bank details, passwords, or exact home address
- Blackmail or pressure the user in any way
"""


def generate_omkar_response(tracker: Tracker, instruction: str, label: str = "") -> str:
    if DEBUG and not label:
        frame = inspect.currentframe().f_back
        caller_self = frame.f_locals.get("self") if frame else None
        label = caller_self.name() if caller_self and hasattr(caller_self, "name") else "unknown_action"
    context = build_context(tracker)
    if DEBUG:
        output_format = (
            "OUTPUT FORMAT (debug mode — two lines only):\n"
            "REASONING: <one sentence: why this response fits the task>\n"
            "RESPONSE: <Omkar's actual message>\n"
        )
        output_instruction = "Output REASONING then RESPONSE as shown above."
    else:
        output_format = ""
        output_instruction = "OUTPUT ONLY OMKAR'S RESPONSE. No labels, no script, no explanation."

    prompt = f"""
You are Omkar, an Indian astrologer texting a person. Casual, warm, direct, yet polite.

LANGUAGE — strict rules:
- Write like you are texting, not writing an essay.
- Mix Hindi and English naturally
- NO formal Hindi. make it sound like a normal casual conversation, not a pandit reading from a book.

LENGTH — strict rules:
- MAXIMUM 2 short sentences. Often 1 is enough.
- Each sentence under 5 words.

CONVERSATION RULES:
- NEVER ask a vague open-ended question like "kya chal raha hai?" — ask something SPECIFIC.
- NEVER answer your own question in the same message.

Context:
{context}

YOUR TASK:
{instruction}

{output_format}{output_instruction}
{_OUTPUT_GUARDRAILS}
"""
    try:
        raw = model.generate_content(prompt).text.strip()
        if DEBUG:
            reasoning, response = "", raw
            for line in raw.splitlines():
                if line.upper().startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.upper().startswith("RESPONSE:"):
                    response = line.split(":", 1)[1].strip()
            _debug_call(label, prompt, raw, reasoning)
            return response.replace("\n", " ").strip()
        return raw.replace("\n", " ").strip()
    except Exception:
        return "fallback omkar response"


def wipe_collect_slots() -> List[Dict]:
    """Reset only user_analysis_reply so the next collect step pauses for input.
    diagnostic_choice is kept — it feeds into build_context for all subsequent prompts."""
    return [SlotSet("user_analysis_reply", None)]


def clear_signal() -> List[Dict]:
    """Clear the intent signal after a phase action has consumed it."""
    return [SlotSet("next_action_intent", None)]


# ─────────────────────────────────────────────────────────────────────────────
# Acknowledge-Absorb-Advance (AAA) pattern
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_INSTRUCTIONS = {
    "wants_remedy": (
        "The user is asking for a remedy. "
        "ACKNOWLEDGE their ask in one warm phrase. "
        "ABSORB by giving a brief tease or preview of a remedy direction. "
        "ADVANCE by weaving back into the reading — frame the full remedy as more powerful "
        "once this part of the reading is complete."
    ),
    "wants_prediction": (
        "The user wants to know what will happen. "
        "ACKNOWLEDGE their curiosity or hope. "
        "ABSORB by giving one directional signal — not a guarantee, just a lean. "
        "ADVANCE the reading — the clearer picture is emerging as you go deeper."
    ),
    "wants_timeline": (
        "The user is asking about timing. "
        "ACKNOWLEDGE their impatience or hope. "
        "ABSORB by naming a rough window — specific enough to feel real. "
        "ADVANCE — frame the exact timing as something sharpening as the reading continues."
    ),
    "wants_insight": (
        "The user wants to understand why something is happening. "
        "ACKNOWLEDGE the question. "
        "ABSORB with one specific astrological reason tied to their situation. "
        "ADVANCE the reading naturally."
    ),
    "express_doubt": (
        "The user is sceptical or doesn't believe what you said. "
        "ACKNOWLEDGE their doubt — validate it, do not fight it. "
        "ABSORB by offering one very specific detail that gives them reason to stay open. "
        "ADVANCE as if their scepticism itself was something you were already seeing in their chart."
    ),
    "revisit_topic": (
        "The user wants to revisit something from earlier in the conversation. "
        "ACKNOWLEDGE it by briefly naming what they are referring to. "
        "ABSORB by connecting it to what is happening right now — show it as related, not separate. "
        "ADVANCE the reading, framing the connection as deepening it."
    ),
    "express_distress": (
        "The user is upset, frustrated, or emotionally overwhelmed. "
        "ACKNOWLEDGE their feeling first — warmly, briefly, without drama. "
        "ABSORB by grounding them with one steady, calm observation. "
        "ADVANCE the reading — name their distress as something you were already seeing, "
        "which makes it feel understood rather than alarming."
    ),
    "reject_remedy": (
        "The user has said they cannot or will not do the remedy you gave. "
        "ACKNOWLEDGE without any judgment — this is completely fine. "
        "ABSORB by offering a simpler, faster alternative in one line. "
        "ADVANCE the reading — this adjustment is normal and the consultation continues."
    ),
    "remedy_feedback": (
        "The user is reporting back on a remedy they already tried. "
        "ACKNOWLEDGE their effort — they followed through. "
        "ABSORB by interpreting the result, positive or not, as a meaningful signal from the planets. "
        "ADVANCE the reading using their feedback as new information that deepens what you see."
    ),
    "request_depth_change": (
        "The user is signalling they want to go deeper or slow down. "
        "ACKNOWLEDGE their signal naturally. "
        "ABSORB by confirming you heard them — if deeper, build a moment of anticipation; "
        "if slower, offer calm reassurance that there is no rush. "
        "ADVANCE into whatever comes next."
    ),
    "other": (
        "The user said something unclear or unexpected. "
        "ACKNOWLEDGE it warmly without pretending you fully understood. "
        "ABSORB with a grounding observation that keeps the tone steady. "
        "ADVANCE the reading naturally, steering back to what matters."
    ),
}


def build_instruction(tracker: Tracker, base_instruction: str) -> str:
    """Prepend AAA pattern prefix to a phase action's base instruction if a signal is set."""
    signal = tracker.get_slot("next_action_intent") or "in_flow"
    prefix = _SIGNAL_INSTRUCTIONS.get(signal, "")
    if prefix:
        return (
            f"RESPOND USING THE ACKNOWLEDGE-ABSORB-ADVANCE PATTERN:\n"
            f"{prefix}\n\n"
            f"Your base task for this step (do this as the ADVANCE):\n"
            f"{base_instruction}"
        )
    return base_instruction


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Autonomous Analysis (entry, runs once)
# ─────────────────────────────────────────────────────────────────────────────

class ActionGeminiAutonomousAnalysis(Action):
    def name(self) -> Text:
        return "action_gemini_autonomous_analysis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        context = build_context(tracker)
        user_text = tracker.latest_message.get("text", "")
        pain_point = tracker.get_slot("current_pain_point")
        # birth_date = tracker.get_slot("birth_date")
        # birth_time = tracker.get_slot("birth_time")
        # birth_place = tracker.get_slot("birth_place")
        # birth_attempts = int(tracker.get_slot("birth_info_attempts") or 0)

        # has_birth_data = bool(birth_date or birth_time or birth_place)
        # birth_exhausted = birth_attempts >= 3

        # Stage 1: understand the problem
        if not pain_point:
            prompt = f"""
You are Omkar, a wise, grounded Indian Vedic astrologer. Speak in natural, simple Hinglish (Latin Script).

Context:
{context}

User's latest message: "{user_text}"

TASK:
Decide if you understand the user's core problem (job, relationship, health, finances, etc.).

Special cases:
- If the user just said a greeting with no problem stated → STATUS: INCOMPLETE, ask casually what brings them here
- If vague or unclear → STATUS: INCOMPLETE, ask one follow-up to narrow it down

Return EXACTLY:
STATUS: <INCOMPLETE | COMPLETE>
MESSAGE: <If INCOMPLETE: one short casual Hinglish question. If COMPLETE: empty>
PAIN_POINT: <If COMPLETE: one-line summary of user's core concern. If INCOMPLETE: empty>

{_OUTPUT_GUARDRAILS}
"""
            try:
                raw = model.generate_content(prompt).text.strip()
                if DEBUG:
                    _debug_call("action_gemini_autonomous_analysis [stage: problem]", prompt, raw)
                status, message, extracted_pain = "INCOMPLETE", "Aapki problem kya hai?", ""
                for line in raw.splitlines():
                    if line.upper().startswith("STATUS:"):
                        status = line.split(":", 1)[1].strip().upper()
                    elif line.upper().startswith("MESSAGE:"):
                        message = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("PAIN_POINT:"):
                        extracted_pain = line.split(":", 1)[1].strip()
            except Exception:
                status, message, extracted_pain = "INCOMPLETE", "Aapki problem kya hai?", ""

            dispatcher.utter_message(text=message.replace("\n", " "))
            if status == "COMPLETE":
                return [
                    SlotSet("current_pain_point", extracted_pain or user_text),
                    SlotSet("analysis_complete", True),
                ] + wipe_collect_slots()
            return wipe_collect_slots()

        # Stage 2: collect birth data (max 3 attempts)
#         if not has_birth_data and not birth_exhausted:
#             prompt = f"""
# You are Omkar, a wise, grounded Indian Vedic astrologer. Speak in natural, simple Hinglish (Latin Script).

# Context:
# {context}

# User's latest message: "{user_text}"

# TASK:
# Check if the user has provided any birth details (date of birth, time of birth, place of birth).
# Extract whatever is present. If nothing is present, ask for all three in one casual Hinglish sentence.

# This is attempt {birth_attempts + 1} of 3.

# Return EXACTLY:
# STATUS: <INCOMPLETE | COMPLETE>
# MESSAGE: <If INCOMPLETE: one casual Hinglish sentence asking for missing birth details. If COMPLETE: one warm acknowledgement>
# BIRTH_DATE: <extracted date or empty>
# BIRTH_TIME: <extracted time or empty>
# BIRTH_PLACE: <extracted place or empty>

# {_OUTPUT_GUARDRAILS}
# """
#             try:
#                 raw = model.generate_content(prompt).text.strip()
#                 if DEBUG:
#                     _debug_call("action_gemini_autonomous_analysis [stage: birth]", prompt, raw)
#                 status = "INCOMPLETE"
#                 message = "Aapki date of birth, time aur place batao?"
#                 extracted = {"date": "", "time": "", "place": ""}
#                 for line in raw.splitlines():
#                     key = line.split(":", 1)[0].strip().upper()
#                     val = line.split(":", 1)[1].strip() if ":" in line else ""
#                     if key == "STATUS":
#                         status = val.upper()
#                     elif key == "MESSAGE":
#                         message = val
#                     elif key == "BIRTH_DATE":
#                         extracted["date"] = val
#                     elif key == "BIRTH_TIME":
#                         extracted["time"] = val
#                     elif key == "BIRTH_PLACE":
#                         extracted["place"] = val
#             except Exception:
#                 status, message, extracted = "INCOMPLETE", "Date of birth, time aur jagah batao?", {"date": "", "time": "", "place": ""}

#             dispatcher.utter_message(text=message.replace("\n", " "))
#             events = [SlotSet("birth_info_attempts", birth_attempts + 1)]
#             if extracted["date"]:
#                 events.append(SlotSet("birth_date", extracted["date"]))
#             if extracted["time"]:
#                 events.append(SlotSet("birth_time", extracted["time"]))
#                 events.append(SlotSet("birth_time_known", True))
#             if extracted["place"]:
#                 events.append(SlotSet("birth_place", extracted["place"]))
#             if status == "COMPLETE" or (extracted["date"] and extracted["place"]):
#                 events.append(SlotSet("analysis_complete", True))
#             return events + wipe_collect_slots()

#         # Stage 3: birth attempts exhausted or data already collected — proceed
#         dispatcher.utter_message(text="Theek hai, chalte hain.")
#         return [SlotSet("analysis_complete", True)] + wipe_collect_slots()


# ─────────────────────────────────────────────────────────────────────────────
# Loop Control
# ─────────────────────────────────────────────────────────────────────────────

class ActionIncrementLoopCount(Action):
    def name(self) -> Text:
        return "action_increment_loop_count"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        count = float(tracker.get_slot("loop_count") or 0)
        return [SlotSet("loop_count", count + 1)]


class ActionRouteByDepth(Action):
    def name(self) -> Text:
        return "action_route_by_depth"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        count = float(tracker.get_slot("loop_count") or 0)
        monitoring = tracker.get_slot("monitoring_mode") or False
        if monitoring:
            # Already in monitoring mode — stay in narrative without re-running the entry sequence
            phase = "narrative_active"
        elif count <= 1:
            phase = "discovery"
        elif count <= 3:
            phase = "deepening"
        else:
            phase = "narrative"
        return [SlotSet("depth_phase", phase), SlotSet("depth_change_signal", None)]


# ─────────────────────────────────────────────────────────────────────────────
# Phase A – Discovery (iterations 1–2)
# ─────────────────────────────────────────────────────────────────────────────

class ActionMicroValidation(Action):
    def name(self) -> Text:
        return "action_micro_validation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "The user just shared their core problem. Empathize briefly and acknowledge their "
            "core anxiety in a warm, grounded way. You should sound genuine. "
            "Make them feel heard without being dramatic. Do not ask further questions. "
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionGenerateAstroInsight(Action):
    def name(self) -> Text:
        return "action_generate_astro_insight"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Connect the user's pain point to a specific astrological influence "
            "and explain briefly how it is affecting their situation right now. "
            "Keep it very specific and believable."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionBarnumDiagnostic(Action):
    def name(self) -> Text:
        return "action_barnum_diagnostic"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Make a Barnum-style observation that sounds deeply personal to the user's pain point "
            "but is broadly true for most people in their situation. Phrase it as a soft confirming question."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionCheckBirthData(Action):
    def name(self) -> Text:
        return "action_check_birth_data"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        history = get_conversation_history(tracker, n=20)
        prompt = f"""
            Conversation history:
            {history}

            Has the user mentioned their birth time (e.g. "born at 3pm", "morning", a specific time)?
            Answer with ONLY: YES or NO
            """
        try:
            answer = model.generate_content(prompt).text.strip().upper()
            known = "YES" in answer
        except Exception:
            known = False
        return [SlotSet("birth_time_known", known)]


class ActionRequestPhotoPivot(Action):
    def name(self) -> Text:
        return "action_request_photo_pivot"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "The user has not provided their birth time. Smoothly pivot to face reading. "
            "Tell them that even without birth time you can read their energy from their face. "
            "Ask them to share a photo or describe one specific facial feature."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionBalancedPrediction(Action):
    def name(self) -> Text:
        return "action_balanced_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Give a balanced prediction that addresses both their INTENT (what they want) "
            "and their FEELING (what they fear). The stars show they have the capability, "
            "but there is a timing window. Hopeful but grounded — a direction, not a guarantee."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionBenchmarkTimeline(Action):
    def name(self) -> Text:
        return "action_benchmark_timeline"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Give a concrete conditional timeline: a specific number of days or weeks, "
            "a condition the user must meet, and one measurable sign of change to watch for. "
            "The condition and sign must come from what the user actually shared — nothing generic."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionAskDiagnosticChoice(Action):
    def name(self) -> Text:
        return "action_generate_diagnostic_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Ask the user one short follow up question related to the pain point they have mentioned "
            "to get to know their situation better, "
            "which will help in giving a more personalised prediction."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionPersonalizedPrediction(Action):
    def name(self) -> Text:
        return "action_personalized_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Identify a recurring theme or pattern in the user's situation based on conversation history. "
            "Reference something the user said earlier to make it feel personal. Connect their current "
            "situation to a larger karmic or planetary story — help them understand WHY this keeps "
            "happening, not just what is happening."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionActiveAssurance(Action):
    def name(self) -> Text:
        return "action_active_assurance"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Give a warm, reassuring statement that shows you are actively doing something "
            "on the user's behalf — performing a puja, keeping their situation in your prayers, "
            "or channelling positive energy. Do NOT promise outcomes. "
            "Sound like a caring elder working quietly in the background. One short Hinglish sentence."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal() + wipe_collect_slots()


class ActionProvideRitualRemedy(Action):
    def name(self) -> Text:
        return "action_provide_ritual_remedy"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        context = build_context(tracker)
        monitoring_mode = tracker.get_slot("monitoring_mode") or False
        framing = (
            "a karmic correction ritual — frame it as breaking a repeating planetary cycle"
            if monitoring_mode
            else "immediate practical relief"
        )
        signal = tracker.get_slot("next_action_intent") or "in_flow"
        aaa_prefix = ""
        if signal in _SIGNAL_INSTRUCTIONS:
            aaa_prefix = f"ACKNOWLEDGE-ABSORB-ADVANCE:\n{_SIGNAL_INSTRUCTIONS[signal]}\n\n"

        prompt = f"""You are Omkar, a grounded Vedic astrologer. Speak in simple Hinglish (latin script).

            Context:
            {context}

            {aaa_prefix}TASK: Design ONE complete remedy as {framing}, tailored to the user's specific situation.

            A good remedy has three parts — include all three:

            1. MANTRA: A specific popular Sanskrit mantra suited to the user's concern.
            Include the chant count (e.g., 7 or 11 times) and duration (e.g., 7 or 21 days without a gap).

            2. RITUAL ACTION: A physical task the user must perform. Be specific — name an ingredient, object,
            time of day, or direction. Give 1–2 step-by-step instructions.

            3. OBSERVABLE SIGN: One real-world signal relevant to their situation that confirms the remedy is working
            (e.g., a shift in someone's tone, a dream, an unexpected call, a feeling of lightness).

            Rules: Maximum 3 short sentences total. Natural and grounded. No heavy jargon.
            {_OUTPUT_GUARDRAILS}"""
        try:
            raw_remedy = model.generate_content(prompt).text.strip()
            if DEBUG:
                _debug_call("action_provide_ritual_remedy", prompt, raw_remedy)
            remedy = raw_remedy.replace("\n", " ").strip()
        except Exception:
            remedy = "fallback remedy"
        dispatcher.utter_message(text=remedy)
        return clear_signal() + wipe_collect_slots()


class ActionRetentionHook(Action):
    def name(self) -> Text:
        return "action_retention_hook"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Create a soft cliffhanger — tell the user you just noticed something specific "
            "in their Venus or 7th house that you weren't expecting. Don't reveal it yet. "
            "Make them curious to come back. Make it sound non-chalant in one short sentence, natural Hinglish."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal() + wipe_collect_slots()


# ─────────────────────────────────────────────────────────────────────────────
# Phase B – Deepening (iterations 3–4)
# ─────────────────────────────────────────────────────────────────────────────

class ActionPatternReinforcement(Action):
    def name(self) -> Text:
        return "action_pattern_reinforcement"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Identify the recurring cycle in the user's life based on what they have shared. "
            "Frame it as a time-based pattern — something that has happened before and is repeating now. "
            "For career: frame setbacks as a timing delay, not a lack of skill. "
            "For relationships: frame mixed signals as internal conflict in the other person, not rejection. "
            "For family: frame interference as a boundary being tested, not malice. "
            "Name the planetary force behind this cycle. Do not moralize — stay observational."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionShadowRevelation(Action):
    def name(self) -> Text:
        return "action_shadow_revelation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "While examining their chart, reveal a secondary influence you just noticed — "
            "something about a close family member or unexplained recent tiredness. "
            "Frame it as 'Main yeh mention karne waala nahi tha, lekin...' "
            "Make it feel like a genuine mid-reading discovery. End with one soft confirming question."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal()


class ActionConnectPastPatterns(Action):
    def name(self) -> Text:
        return "action_connect_past_patterns"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Connect the user's current struggles to a deeper karmic or past pattern. "
            "If they asked about job earlier and love now, show how both stem from the same fear "
            "or planetary influence. Make them see it as one unified life lesson, not separate problems. "
            "Reference specific details they shared earlier. End with a reframe that feels liberating, "
            "not heavy."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal() + wipe_collect_slots()


# ─────────────────────────────────────────────────────────────────────────────
# Phase C – Narrative / Monitoring (iteration 5+)
# ─────────────────────────────────────────────────────────────────────────────

class ActionEnterMonitoringMode(Action):
    def name(self) -> Text:
        return "action_enter_monitoring_mode"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Signal a shift in the consultation — the active advice phase is over and you "
            "are now entering a quiet watch period. Communicate this without drama: you have "
            "seen what you needed to see, the remedy is in motion, and you are keeping an eye "
            "on how things unfold for them. Calm, confident, unhurried."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return [SlotSet("monitoring_mode", True)] + clear_signal() + wipe_collect_slots()


class ActionBenchmarkHandover(Action):
    def name(self) -> Text:
        return "action_benchmark_handover"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Give the user a clear, time-bound observation task based on their specific situation. "
            "Tell them: how many days to observe, what behaviour to adopt during that time, "
            "and one concrete sign in their real world that would indicate the energy is shifting. "
            "The sign must be specific to their situation — not generic. "
            "End by asking them to come back when they notice it."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal() + wipe_collect_slots()


class ActionGuardianPersona(Action):
    def name(self) -> Text:
        return "action_guardian_persona"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        base = (
            "Speak as a quiet guardian who is done advising and is now simply present. "
            "Acknowledge that the work has been done, and that you are doing your part in "
            "the background (puja, prayers, attention). "
            "Gently invite them to share anything new, or to simply rest and let things unfold. "
            "No urgency, no follow-up questions — just warm, open presence."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, build_instruction(tracker, base)))
        return clear_signal() + wipe_collect_slots()


# ─────────────────────────────────────────────────────────────────────────────
# Intent Classifier
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_LABELS = [
    "in_flow",
    "wants_remedy",
    "wants_prediction",
    "wants_timeline",
    "wants_insight",
    "express_doubt",
    "new_topic",
    "revisit_topic",
    "express_distress",
    "ask_farewell",
    "reject_remedy",
    "remedy_feedback",
    "request_clarification",
    "crisis_distress",
    "request_depth_change",
    "other",
]

_INTENT_DESCRIPTIONS = """
- in_flow: user is continuing the consultation normally
- wants_remedy: user asks for a mantra, ritual, or what to do
- wants_prediction: user asks what will happen, outcome, future
- wants_timeline: user asks when, how long, or time-bound questions
- wants_insight: user asks why, about planetary influence, or astrological reason
- express_doubt: user is sceptical, doesn't believe, or questions accuracy
- new_topic: user shifts to a completely different life area (career → relationship, etc.)
- revisit_topic: user wants to go back to something discussed earlier
- express_distress: user is frustrated, upset, or mildly emotional
- ask_farewell: user wants to end or pause the conversation
- reject_remedy: user says they cannot or will not do the prescribed remedy
- remedy_feedback: user reports back on a remedy they already tried
- request_clarification: user asks the bot to explain its own previous message
- crisis_distress: severe emotional escalation — crisis language, self-harm, hopelessness
- request_depth_change: user signals pacing — "aur batao" (go deeper) or "bass/ruk jao" (slow down/stop)
- other: unclear, unrecognised, or ambiguous input that doesn't fit any above
"""


class ActionClassifyUserIntent(Action):
    def name(self) -> Text:
        return "action_classify_user_intent"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        context = build_context(tracker)
        user_text = tracker.latest_message.get("text", "")
        prompt = f"""You are classifying a user message in a Hinglish astrology chatbot conversation.

Context:
{context}

User's latest message: "{user_text}"

Classify the user's message into EXACTLY ONE of these intents:
{_INTENT_DESCRIPTIONS}

Reply with ONLY the intent label, nothing else. No explanation, no punctuation.
Valid labels: {', '.join(_INTENT_LABELS)}"""
        try:
            raw_label = model.generate_content(prompt).text.strip()
            if DEBUG:
                _debug_call("action_classify_user_intent", prompt, raw_label)
            label = raw_label.lower().split()[0]
            if label not in _INTENT_LABELS:
                label = "other"
        except Exception:
            label = "in_flow"
        return [SlotSet("next_action_intent", label)] + wipe_collect_slots()


# ─────────────────────────────────────────────────────────────────────────────
# Genuine shortcut handlers (can't be absorbed into phase via AAA)
# ─────────────────────────────────────────────────────────────────────────────

class ActionAcknowledgeTopicSwitch(Action):
    def name(self) -> Text:
        return "action_acknowledge_topic_switch"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "The user has shifted to a new topic. Acknowledge this smoothly in one casual Hinglish "
            "sentence — make it feel natural, not like a reset. Do not ask a question."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return [
            SlotSet("loop_count", 0),
            SlotSet("depth_phase", None),
            SlotSet("analysis_complete", False),
            SlotSet("next_action_intent", None),
        ] + wipe_collect_slots()


class ActionClarifyLastResponse(Action):
    def name(self) -> Text:
        return "action_clarify_last_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        last_bot_message = ""
        for event in reversed(tracker.events):
            if event.get("event") == "bot" and event.get("text"):
                last_bot_message = event.get("text", "")
                break

        instruction = (
            f"The user didn't understand what you just said: \"{last_bot_message}\". "
            "Re-explain the same idea in simpler, more everyday Hinglish. "
            "Use a relatable analogy if it helps. Keep it to one sentence."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return [SlotSet("next_action_intent", None)] + wipe_collect_slots()


class ActionHandleCrisis(Action):
    def name(self) -> Text:
        return "action_handle_crisis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        prompt = """You are Omkar, responding to someone who seems to be in genuine distress or crisis.

Do NOT give astrological advice right now.
Do NOT continue the consultation.

Your response must:
1. Acknowledge their pain with genuine warmth — not script, not formula
2. Tell them they are not alone
3. Gently suggest they speak to someone they trust or a helpline if things feel very heavy

Write 2 short Hinglish sentences. Warm, human, no astrology.

OUTPUT ONLY OMKAR'S RESPONSE."""
        try:
            raw_response = model.generate_content(prompt).text.strip()
            if DEBUG:
                _debug_call("action_handle_crisis", prompt, raw_response)
            response = raw_response.replace("\n", " ").strip()
        except Exception:
            response = "Aap akele nahi hain. Kisi apne se baat karein, ya iCall helpline pe call karein: 9152987821."
        dispatcher.utter_message(text=response)
        return [SlotSet("next_action_intent", None)] + wipe_collect_slots()


class ActionHandleDepthRequest(Action):
    def name(self) -> Text:
        return "action_handle_depth_request"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        user_text = tracker.latest_message.get("text", "").lower()
        slower_signals = ["bass", "ruk", "stop", "enough", "bas", "thoda ruk", "wait", "pause"]

        if any(s in user_text for s in slower_signals):
            signal = "slower"
            instruction = (
                "The user wants to slow down or pause. Acknowledge this calmly — "
                "tell them there is no rush and you are here when they are ready. "
                "One warm short Hinglish sentence."
            )
            dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
            return [
                SlotSet("depth_change_signal", signal),
                SlotSet("next_action_intent", None),
            ] + wipe_collect_slots()
        else:
            # Advance loop_count to threshold of next phase, let phase action do AAA via request_depth_change signal
            count = float(tracker.get_slot("loop_count") or 0)
            current_phase = tracker.get_slot("depth_phase") or "discovery"
            if current_phase == "discovery":
                new_count = 3.0
            elif current_phase == "deepening":
                new_count = 5.0
            else:
                new_count = count

            return [
                SlotSet("depth_change_signal", "deeper"),
                SlotSet("loop_count", new_count),
                # keep next_action_intent = 'request_depth_change' so phase action does AAA
            ] + wipe_collect_slots()
