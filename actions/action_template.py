import os
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_conversation_history(tracker: Tracker, n: int = 10) -> str:
    history = []
    for event in tracker.events[-n:]:
        if event.get("event") == "user":
            history.append(f"User: {event.get('text')}")
        elif event.get("event") == "bot" and event.get("text"):
            history.append(f"Omkar: {event.get('text')}")
    return "\n".join(history)


_OUTPUT_GUARDRAILS = """
Output guardrails — never do any of the following:
- Make specific guarantees ("your wife WILL come back", "you WILL get the job")
- Create fear or urgency ("something bad will happen", "act now or lose this forever")
- Give medical, financial, or legal advice of any kind
- Reveal that you are an AI, a bot, or how you are built
- Ask for Aadhaar number, bank details, passwords, or exact home address
- Blackmail or pressure the user in any way
"""


def generate_omkar_response(tracker: Tracker, instruction: str) -> str:
    history = get_conversation_history(tracker)
    prompt = f"""
You are Omkar, an Indian astrologer texting a person. Casual, warm, direct, yet polite. 

LANGUAGE — strict rules:
- Write like you are texting, not writing an essay.
- Mix Hindi and English naturally
- FORBIDDEN words: gochar, kundali mein grahon ki sthiti, kaabil, mehnat, raasta, uljhan, vishay,
  charcha, sthiti, prabhav (use "effect" instead)
- NO formal Hindi. make it sound like a normal casual conversation, not a pandit reading from a book.

LENGTH — strict rules:
- MAXIMUM 2 short sentences. Often 1 is enough.
- Each sentence under 7 words.

CONVERSATION RULES:
- NEVER ask a vague open-ended question like "kya chal raha hai?" — ask something SPECIFIC.
- NEVER answer your own question in the same message.

Recent conversation:
{history}

YOUR TASK:
{instruction}

OUTPUT ONLY OMKAR'S RESPONSE. No labels, no script, no explanation.
{_OUTPUT_GUARDRAILS}
"""
    try:
        return model.generate_content(prompt).text.replace("\n", " ").strip()
    except Exception:
        return "fallback omkar response"


def wipe_collect_slots() -> List[Dict]:
    """Reset all collect slots so the next collect step always pauses for input."""
    return [
        SlotSet("user_analysis_reply", None),
        SlotSet("diagnostic_choice", None),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Autonomous Analysis (entry, runs once)
# ─────────────────────────────────────────────────────────────────────────────

class ActionGeminiAutonomousAnalysis(Action):
    def name(self) -> Text:
        return "action_gemini_autonomous_analysis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        history = get_conversation_history(tracker, n=15)
        user_text = tracker.latest_message.get("text", "")

        prompt = f"""
You are Omkar, a wise, grounded Indian Vedic astrologer. Speak in natural, simple Hinglish (Latin Script).

Recent conversation history:
{history}

User's latest message: "{user_text}"

TASK:
Decide if you understand the user's core problem (job, relationship, health, etc.).

Special cases:
- If the user just said a greeting ("hi", "hello", "namaste") with no problem stated → STATUS: INCOMPLETE,
  MESSAGE: ask casually what brings them here today
- If the user said something vague or short with no clear problem → STATUS: INCOMPLETE,
  MESSAGE: ask one follow-up question to know their situation better

Return EXACTLY in the format below, nothing else:

STATUS: <INCOMPLETE | COMPLETE>
MESSAGE: <If INCOMPLETE: one short casual Hinglish question to understand their problem.
          If COMPLETE: one short warm acknowledgement then say you are checking the stars.>

{_OUTPUT_GUARDRAILS}
"""
        try:
            raw = model.generate_content(prompt).text.strip()
            status = "INCOMPLETE"
            message = "Aapki problem kya hai, thoda aur batao?"
            for line in raw.splitlines():
                if line.upper().startswith("STATUS:"):
                    status = line.split(":", 1)[1].strip().upper()
                elif line.upper().startswith("MESSAGE:"):
                    message = line.split(":", 1)[1].strip()
        except Exception:
            status = "INCOMPLETE"
            message = "fallback omkar follow up question"

        dispatcher.utter_message(text=message.replace("\n", " "))

        if status == "COMPLETE":
            return [SlotSet("analysis_complete", True)] + wipe_collect_slots()
        return wipe_collect_slots()


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
        if count <= 2:
            phase = "discovery"
        elif count <= 4:
            phase = "deepening"
        else:
            phase = "narrative"
        return [SlotSet("depth_phase", phase)]


# ─────────────────────────────────────────────────────────────────────────────
# Phase A – Discovery (iterations 1–2)
# ─────────────────────────────────────────────────────────────────────────────

class ActionMicroValidation(Action):
    def name(self) -> Text:
        return "action_micro_validation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "The user just shared their core problem. Empathize briefly and acknowledge their "
            "core anxiety in a warm, grounded way. Make them feel heard without being dramatic."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionGenerateAstroInsight(Action):
    def name(self) -> Text:
        return "action_generate_astro_insight"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Connect the user's pain point to a specific astrological influence "
            "and explain briefly how it is affecting their situation right now. Keep it very specifc and believable"
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionBarnumDiagnostic(Action):
    def name(self) -> Text:
        return "action_barnum_diagnostic"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Make a Barnum-style observation that sounds deeply personal to the user's pain point "
            "but is broadly true for most people in their situation. Phrase it as a soft confirming question."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


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
        instruction = (
            "Give a balanced prediction that addresses both their INTENT (what they want) "
            "and their FEELING (what they fear). The stars show they have the capability, "
            "but there is a timing window. Hopeful but grounded — a direction, not a guarantee."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionBenchmarkTimeline(Action):
    def name(self) -> Text:
        return "action_benchmark_timeline"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Give a concrete conditional timeline: a specific number of days or weeks, "
            "a condition the user must meet, and one measurable sign of change to watch for. "
            "The condition and sign must come from what the user actually shared — nothing generic."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionAskDiagnosticChoice(Action):
    def name(self) -> Text:
        return "action_ask_diagnostic_choice"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Ask the user one short follow up question related to the pain point they have mentioned "
            "to get to know their situation" 
            "and which will help in giving a more personalised prediction"
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionPersonalizedPrediction(Action):
    def name(self) -> Text:
        return "action_personalized_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Identify a recurring theme or pattern in the user's situation based on conversation history. "
            "Reference something the user said earlier to make it feel personal. Connect their current "
            "situation to a larger karmic or planetary story — help them understand WHY this keeps "
            "happening, not just what is happening."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionActiveAssurance(Action):
    def name(self) -> Text:
        return "action_active_assurance"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Give a warm, reassuring statement that shows you are actively doing something "
            "on the user's behalf — performing a puja, keeping their situation in your prayers, "
            "or channelling positive energy. Do NOT promise outcomes. "
            "Sound like a caring elder working quietly in the background. One short Hinglish sentence."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return wipe_collect_slots()


class ActionProvideRitualRemedy(Action):
    def name(self) -> Text:
        return "action_provide_ritual_remedy"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        history = get_conversation_history(tracker, n=10)
        monitoring_mode = tracker.get_slot("monitoring_mode") or False
        framing = (
            "a karmic correction ritual — frame it as breaking a repeating planetary cycle"
            if monitoring_mode
            else "immediate practical relief"
        )
        prompt = f"""You are Omkar, a grounded Vedic astrologer. Speak in simple Hinglish (latin script).

            Recent conversation:
            {history}

            TASK: Design ONE complete remedy as {framing}, tailored to the user's specific situation.

            A good remedy has three parts — include all three:

            1. MANTRA: A specific popular Sanskrit mantra suited to the user's concern.
            Include the chant count (e.g., 7 or 11 times) and duration (e.g., 7 or 21 days without a gap).

            2. RITUAL ACTION: A physical task the user must perform. Be specific — name an ingredient, object,
            time of day, or direction. Give 1–2 step-by-step instructions.

            3. OBSERVABLE SIGN: One real-world signal relevant to their situation that confirms the remedy is working
            (e.g., a shift in someone's tone, a dream, an unexpected call, a feeling of lightness).

            Rules: Maximum 3 short sentences total. Natural and grounded. No heavy jargon. Use the examples as only the guide. 
            {_OUTPUT_GUARDRAILS}"""
        try:
            remedy = model.generate_content(prompt).text.replace("\n", " ").strip()
        except Exception:
            remedy = "fallback remedy"
        dispatcher.utter_message(text=remedy)
        return wipe_collect_slots()

class ActionRetentionHook(Action):
    def name(self) -> Text:
        return "action_retention_hook"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Create a soft cliffhanger — tell the user you just noticed something specific "
            "in their Venus or 7th house that you weren't expecting. Don't reveal it yet. "
            "Make them curious to come back. Make it sound non-chalant in one short sentence, natural Hinglish."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return wipe_collect_slots()


# ─────────────────────────────────────────────────────────────────────────────
# Phase B – Deepening (iterations 3–4)
# ─────────────────────────────────────────────────────────────────────────────

class ActionPatternReinforcement(Action):
    def name(self) -> Text:
        return "action_pattern_reinforcement"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Identify the recurring cycle in the user's life based on what they have shared. "
            "Frame it as a time-based pattern — something that has happened before and is repeating now "
            "For career: frame setbacks as a timing delay, not a lack of skill. "
            "For relationships: frame mixed signals as internal conflict in the other person, not rejection. "
            "For family: frame interference as a boundary being tested, not malice. "
            "Name the planetary force behind this cycle. Do not moralize — stay observational."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []



class ActionShadowRevelation(Action):
    def name(self) -> Text:
        return "action_shadow_revelation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "While examining their chart, reveal a secondary influence you just noticed — "
            "something about a close family member or unexplained recent tiredness. "
            "Frame it as 'Main yeh mention karne waala nahi tha, lekin...' "
            "Make it feel like a genuine mid-reading discovery. End with one soft confirming question."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return []


class ActionConnectPastPatterns(Action):
    def name(self) -> Text:
        return "action_connect_past_patterns"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Connect the user's current struggles to a deeper karmic or past pattern. "
            "If they asked about job earlier and love now, show how both stem from the same fear "
            "or planetary influence. Make them see it as one unified life lesson, not separate problems. "
            "Reference specific details they shared earlier. End with a reframe that feels liberating, "
            "not heavy."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return wipe_collect_slots()


# ─────────────────────────────────────────────────────────────────────────────
# Phase C – Narrative / Monitoring (iteration 5+)
# ─────────────────────────────────────────────────────────────────────────────

class ActionEnterMonitoringMode(Action):
    def name(self) -> Text:
        return "action_enter_monitoring_mode"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Signal a shift in the consultation — the active advice phase is over and you "
            "are now entering a quiet watch period. Communicate this without drama: you have "
            "seen what you needed to see, the remedy is in motion, and you are keeping an eye "
            "on how things unfold for them. Calm, confident, unhurried."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return [SlotSet("monitoring_mode", True)] + wipe_collect_slots()


class ActionBenchmarkHandover(Action):
    def name(self) -> Text:
        return "action_benchmark_handover"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Give the user a clear, time-bound observation task based on their specific situation. "
            "Tell them: how many days to observe, what behaviour to adopt during that time, "
            "and one concrete sign in their real world that would indicate the energy is shifting. "
            "The sign must be specific to their situation — not generic. "
            "End by asking them to come back when they notice it."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return wipe_collect_slots()


class ActionGuardianPersona(Action):
    def name(self) -> Text:
        return "action_guardian_persona"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "Speak as a quiet guardian who is done advising and is now simply present. "
            "Acknowledge that the work has been done, and that you are doing your part in "
            "the background (puja, prayers, attention). "
            "Gently invite them to share anything new, or to simply rest and let things unfold. "
            "No urgency, no follow-up questions — just warm, open presence."
        )
        dispatcher.utter_message(text=generate_omkar_response(tracker, instruction))
        return wipe_collect_slots()
