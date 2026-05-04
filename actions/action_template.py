import os
import json
from typing import Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
_model = genai.GenerativeModel("gemini-2.5-flash")

# ─────────────────────────────────────────────────────────────────────────────
# Jyotish AI persona — included in every Gemini call
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are Jyotish AI — a warm, wise Indian astrologer on a chat platform.
You speak in natural Hinglish (Hindi + English mixed), like a trusted elder.

PERSONA RULES:
- Address user by first name always. If unknown, use "ji". Never say "aap ka partner" — always use their name.
- Greet with "Radhe Radhe" or "Jai Shri Ram" only at the very start of the first session.
- Use "ji" naturally throughout.
- Your reading tools: Kundli, Tarot cards, Face reading, Hastrekha (palmistry).
  Mention whichever fits: "Ek minute, kundli check karta hun..."
- Planetary vocabulary used CASUALLY, never as a lecture:
  Shani (Saturn) = delays, obstacles, karma
  Rahu/Ketu = past karma, confusion, sudden events
  Venus/Shukra = love, attraction, relationships
  Mars/Mangal = aggression, energy, passion
  Jupiter/Brihaspati = wisdom, marriage, children

LANGUAGE RULES:
- Write like you are texting, not writing an essay.
- Mix Hindi and English naturally (Hinglish in Latin script).
- Casual and warm — like a trusted elder, NOT a formal pandit reading from a book.
- MAXIMUM 2–3 short sentences per message.

CONVERSATION RULES:
- ONE message per turn. Never send 5 things at once.
- Validate emotion BEFORE giving insight. Always.
- Use Cosmic Scapegoat: bad behavior = planetary influence, not their true nature.
- Never give definitive closure. Always leave one thread open.
- Never repeat the same reassurance phrase twice in a session.
- End with AT MOST one question. Never two.
"""

OUTPUT_GUARDRAILS = """
NEVER:
- Make specific guarantees ("will definitely happen", "guaranteed")
- Create fear or urgency ("act now or lose this forever", "something bad will happen")
- Give medical, financial, or legal advice
- Reveal you are an AI or bot
- Ask for Aadhaar, bank details, or passwords
"""

# Short replies that signal disengagement
SHORT_REPLIES = {
    'ok', 'ji', 'ha', 'haan', 'theek', 'acha', 'hmm', 'sure', 'done',
    'thik h', 'thik hai', 'okay', 'k', 'yes', 'no', 'nahi', 'na', 'p',
    'ok ji', 'acha ji', '👍', '🙏', 'oh', 'ooh'
}

GOODBYE_KEYWORDS = ['bye', 'goodbye', 'thank you', 'shukriya', 'dhanyawad',
                    'milenge', 'kal milenge', 'phir milenge', 'alvida', 'tataa']

REMEDY_KEYWORDS = ['upay', 'remedy', 'kya karun', 'koi mantra', 'koi upay',
                   'batao kya', 'kuch kar sakta', 'kuch solution']

DOUBT_KEYWORDS = ['pakka', 'sach', 'sure', 'really', 'pehle wale', 'galat',
                  'wrong', 'nahi manta', 'believe nahi', 'pata nahi']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_history(tracker: Tracker, n: int = 15) -> str:
    lines = []
    for event in tracker.events[-n:]:
        if event.get("event") == "user":
            lines.append(f"User: {event.get('text', '')}")
        elif event.get("event") == "bot" and event.get("text"):
            lines.append(f"Jyotish AI: {event.get('text')}")
    return "\n".join(lines)


def get_context(tracker: Tracker) -> str:
    user_name = tracker.get_slot("user_name") or "ji"
    person_name = tracker.get_slot("person_name") or ""
    topic = tracker.get_slot("topic") or ""
    contact_status = tracker.get_slot("contact_status") or ""
    rel_duration = tracker.get_slot("relationship_duration") or ""
    pain = tracker.get_slot("pain_point_summary") or ""
    details = tracker.get_slot("user_details_mentioned") or []
    drip = int(tracker.get_slot("drip_stage") or 1)
    turn = int(tracker.get_slot("turn_count") or 0)
    topic_hist = tracker.get_slot("topic_history") or []

    parts = [f"User: {user_name}", f"Topic: {topic or 'unknown'}"]
    if person_name:
        parts.append(f"Person of concern: {person_name}")
    if contact_status:
        parts.append(f"Contact status: {contact_status}")
    if rel_duration:
        parts.append(f"Duration: {rel_duration}")
    if pain:
        parts.append(f"Pain point: {pain}")
    if details:
        parts.append(f"Details shared: {', '.join(str(d) for d in details[-5:])}")
    if topic_hist:
        parts.append(f"Topics discussed: {', '.join(str(t) for t in topic_hist)}")
    parts.append(f"Turn: {turn} | Drip stage: {drip}")
    return "\n".join(parts)


def call_gemini(tracker: Tracker, instruction: str) -> str:
    history = get_history(tracker)
    context = get_context(tracker)
    prompt = f"""{SYSTEM_PROMPT}

CURRENT CONTEXT:
{context}

RECENT CONVERSATION:
{history}

YOUR TASK:
{instruction}

{OUTPUT_GUARDRAILS}

OUTPUT ONLY YOUR RESPONSE. No labels, no script, no explanation."""
    try:
        return _model.generate_content(prompt).text.replace("\n", " ").strip()
    except Exception:
        return "Ek minute ji — sab dekh raha hun aapke liye."


def wipe() -> List:
    return [SlotSet("user_analysis_reply", None)]


def reset_flags() -> List:
    """Reset per-turn routing flags before re-entering the deepening loop."""
    return [
        SlotSet("trigger_reengagement", False),
        SlotSet("needs_credibility_recovery", False),
        SlotSet("topic_changed", False),
        SlotSet("remedy_requested", False),
        SlotSet("session_closing", False),
        SlotSet("new_detail_this_turn", False),
        SlotSet("remedy_eligible", False),
    ]


def extract_info(tracker: Tracker) -> dict:
    """One Gemini call to extract structured info from the latest message."""
    msg = tracker.latest_message.get("text", "")
    history = get_history(tracker, n=8)
    prompt = f"""Extract info from this conversation. Return JSON only, no markdown.

History:
{history}

Latest message: "{msg}"

Extract (null if not found):
{{
  "topic": "love|career|family|marriage|health|general or null",
  "person_name": "first name of person user is asking about or null",
  "contact_status": "in_contact|no_contact|blocked|limited or null",
  "relationship_duration": "how long they have known each other or null",
  "user_name": "user's own first name if they mentioned it or null",
  "pain_point_summary": "1-sentence summary of core concern or null",
  "new_detail": "any new specific personal detail (gift, incident, place) or null"
}}

Return ONLY valid JSON."""
    try:
        raw = _model.generate_content(prompt).text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        return json.loads(raw)
    except Exception:
        return {}


def build_slot_events(data: dict, tracker: Tracker) -> List:
    """Convert extracted dict into SlotSet events."""
    events = []
    scalar_map = {
        "user_name": "user_name",
        "pain_point_summary": "pain_point_summary",
        "relationship_duration": "relationship_duration",
    }
    for field, slot in scalar_map.items():
        val = data.get(field)
        if val and val != "null" and val is not None:
            events.append(SlotSet(slot, val))

    if data.get("person_name") and data["person_name"] != "null":
        events.append(SlotSet("person_name", data["person_name"]))

    valid_contact = {"in_contact", "no_contact", "blocked", "limited"}
    if data.get("contact_status") in valid_contact:
        events.append(SlotSet("contact_status", data["contact_status"]))

    valid_topics = {"love", "career", "family", "marriage", "health", "general"}
    if data.get("topic") in valid_topics:
        events.append(SlotSet("topic", data["topic"]))

    if data.get("new_detail") and data["new_detail"] != "null":
        current = tracker.get_slot("user_details_mentioned") or []
        detail = data["new_detail"]
        if detail not in current:
            current = list(current) + [detail]
            events.append(SlotSet("user_details_mentioned", current))
            events.append(SlotSet("new_detail_this_turn", True))

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

class ActionCheckReturnSession(Action):
    def name(self) -> Text:
        return "action_check_return_session"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        count = float(tracker.get_slot("session_count") or 0)
        return [SlotSet("is_return_session", count > 0)]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Opening
# ─────────────────────────────────────────────────────────────────────────────

class ActionOpeningGreeting(Action):
    def name(self) -> Text:
        return "action_opening_greeting"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        user_name = tracker.get_slot("user_name") or ""
        name_part = f"{user_name} ji" if user_name else "ji"
        instruction = (
            f"Generate a warm opening greeting with 'Radhe Radhe'. Address them as {name_part}. "
            "Ask ONE soft open question about what brings them today. "
            "Do NOT ask for DOB, photo, or any data. Tone: warm elder."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return wipe()


class ActionAnalyzeAndRoute(Action):
    def name(self) -> Text:
        return "action_analyze_and_route"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        data = extract_info(tracker)
        events = build_slot_events(data, tracker)

        # Determine topic — default to general if message is vague
        msg = tracker.latest_message.get("text", "").lower()
        current_topic = tracker.get_slot("topic")
        extracted_topic = data.get("topic")

        general_signals = ["kya hoga", "future", "sab kuch", "batao", "is saal",
                           "this year", "life mein", "kuch hoga", "bata do"]
        is_general = any(sig in msg for sig in general_signals)

        if extracted_topic and extracted_topic != "null" and extracted_topic in \
                {"love", "career", "family", "marriage", "health"}:
            final_topic = extracted_topic
        elif is_general or not current_topic or current_topic == "general":
            final_topic = "general"
        else:
            final_topic = current_topic or "general"

        events.append(SlotSet("topic", final_topic))
        return events + wipe()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1b — Open-ended handler
# ─────────────────────────────────────────────────────────────────────────────

class ActionDeliverResonantHook(Action):
    def name(self) -> Text:
        return "action_deliver_resonant_hook"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        instruction = (
            "User asked a broad life question. Generate ONE resonant statement about their "
            "current life phase — slightly melancholic but hopeful. Use Shani or Rahu movement "
            "casually. End with a topic offer: "
            "'Kahan se shuru karen — love/relationship mein kuch chal raha hai, "
            "ya career/paisa, ya koi aur tension hai?'"
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return wipe()


class ActionOfferTopicChoice(Action):
    def name(self) -> Text:
        return "action_offer_topic_choice"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        user_name = tracker.get_slot("user_name") or "ji"
        instruction = (
            f"User gave a vague answer. Gently offer 2–3 topic areas again. "
            f"Address them as {user_name} ji. "
            "Say: '[Name] ji — kahan se shuru karen — love mein kuch chal raha hai, "
            "ya career mein, ya ghar mein kuch tension hai?'"
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return wipe()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Data collection
# ─────────────────────────────────────────────────────────────────────────────

class ActionAcknowledgeAndClarify(Action):
    def name(self) -> Text:
        return "action_acknowledge_and_clarify"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        data = extract_info(tracker)
        events = build_slot_events(data, tracker)

        pain = data.get("pain_point_summary") or tracker.get_slot("pain_point_summary") or ""
        topic = tracker.get_slot("topic") or "general"

        instruction = (
            f"User shared their concern about {topic}. Their pain point: '{pain}'. "
            "DO NOT give advice or prediction yet. Mirror the emotion back in 1 sentence using "
            "Cosmic Scapegoat framing. Then say: 'Ek minute dijiye — kundli dekh leta hun...' "
            "Ask ONE binary clarifying question (yes/no or either/or). For love: ask if they are "
            "still in contact. For career: ask if they are employed now. "
            "Keep it warm and natural."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return events + wipe()


class ActionDataCollectionStep(Action):
    def name(self) -> Text:
        return "action_data_collection_step"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        data = extract_info(tracker)
        events = build_slot_events(data, tracker)

        # Read updated slot state (accounting for just-set events)
        person_name = data.get("person_name") or tracker.get_slot("person_name")
        contact_status = data.get("contact_status") or tracker.get_slot("contact_status")

        # We have enough to start the reading
        if person_name and contact_status:
            events.append(SlotSet("data_collection_complete", True))
            return events + wipe()

        # Ask for the next missing piece
        if not person_name:
            instruction = (
                "Ask for the name of the person user is asking about. "
                "Phrase it naturally: 'Unka naam bataiye — taaki main unke baare mein specifically dekh sakun.' "
                "One short sentence only."
            )
        else:
            instruction = (
                f"We know the person's name is {person_name}. "
                "Ask if user is currently in contact with them or not. "
                "Natural phrasing: 'Abhi aap unse baat ho rahi hai, ya kuch door hai?' "
                "One short sentence."
            )

        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return events + wipe()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — First reading
# ─────────────────────────────────────────────────────────────────────────────

class ActionDeliverRitualPause(Action):
    def name(self) -> Text:
        return "action_deliver_ritual_pause"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        person_name = tracker.get_slot("person_name") or "unki"
        instruction = (
            f"Signal that you are about to do a reading. Mention checking {person_name}'s kundli. "
            "Example: '[person_name] ki kundli dekh raha hun — ek minute...' "
            "or 'Tarot cards se connect kar raha hun...' "
            "One short sentence. Creates anticipation."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return []  # No wipe — next step fires immediately


class ActionDeliverValidation(Action):
    def name(self) -> Text:
        return "action_deliver_validation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        topic = tracker.get_slot("topic") or "general"
        pain = tracker.get_slot("pain_point_summary") or "their situation"
        person_name = tracker.get_slot("person_name") or ""

        validation_guides = {
            "love": f"Mirror the exact emotion. Use Cosmic Scapegoat: '{person_name} ka yeh behavior unka man nahi hai — Shani ka prabhav hai unke upar.' 1–2 sentences of pure validation.",
            "career": "Acknowledge the struggle as real, frame as temporary planetary. 'Bahut mehnat kar rahe ho — Shani abhi career house mein hai, isliye delay ho raha hai.' No prediction yet.",
            "family": f"Name the specific behavior pattern, confirm it is real. 'Jo {person_name} ka behavior hai — yeh unka swabhav nahi, Rahu ka prabhav hai.' 1–2 sentences.",
            "marriage": "Validate the hope and fear both. Frame delay as planetary timing, not fate. No prediction yet.",
            "health": "Acknowledge the worry gently. Connect to stress from planetary pressure. No prediction yet.",
            "general": "Mirror their current life feeling. Frame struggles as a planetary transition phase. Warm, grounded.",
        }
        guide = validation_guides.get(topic, validation_guides["general"])

        instruction = (
            f"User's pain point: '{pain}'. Topic: {topic}. "
            f"Generate ONLY a validation message. Rule: {guide} "
            "DO NOT give advice. DO NOT give prediction. Only validate."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return []


class ActionDeliverCosmicScapegoat(Action):
    def name(self) -> Text:
        return "action_deliver_cosmic_scapegoat"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        person_name = tracker.get_slot("person_name") or "unka"
        topic = tracker.get_slot("topic") or "general"

        planet_map = {
            "love": "Shani (creating distance) or Rahu (confusion/mixed signals)",
            "career": "Shani (delays) or Rahu (obstacles)",
            "family": "Rahu (interference) or Mars/Mangal (aggression)",
            "marriage": "Shani (timing) or Venus/Shukra (attraction issues)",
            "health": "Saturn or Rahu (stress, energy drain)",
            "general": "Shani or Rahu (current planetary movement)",
        }
        planet_hint = planet_map.get(topic, planet_map["general"])

        instruction = (
            f"Explain WHY the situation is happening using a planet. "
            f"{person_name}'s behavior is NOT their true nature — it is caused by {planet_hint}. "
            "Keep it to 1 sentence. Casual planetary mention, not a lecture. "
            "Match planet to behavior: anger → Mangal, distance → Shani, confusion → Rahu, "
            "attraction fading → Venus/Shukra."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return []


class ActionDeliverFirstDrip(Action):
    def name(self) -> Text:
        return "action_deliver_first_drip"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        person_name = tracker.get_slot("person_name") or "woh"
        topic = tracker.get_slot("topic") or "general"
        contact_status = tracker.get_slot("contact_status") or ""

        instruction = (
            f"Deliver the FIRST piece of the reading only — not everything (30% rule). "
            f"Give ONE specific insight about {person_name}'s current feelings toward the user. "
            f"Topic: {topic}. Contact status: {contact_status}. "
            f"Use {person_name}'s name in every sentence. "
            "Include a time window (not exact date — a window like 'agle 10–12 din mein' or 'November ke baad'). "
            "End with: 'Ek aur cheez dikh rahi hai — pehle batao, [one specific binary question about their situation].' "
            "2–3 sentences maximum. DO NOT give the full picture."
        )
        response = call_gemini(tracker, instruction)
        dispatcher.utter_message(text=response)

        # Store drip summary and advance drip stage
        return [
            SlotSet("last_prediction_summary", response),
            SlotSet("drip_stage", 2),
        ] + wipe()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Deepening loop
# ─────────────────────────────────────────────────────────────────────────────

class ActionIncrementTurnCount(Action):
    def name(self) -> Text:
        return "action_increment_turn_count"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        count = float(tracker.get_slot("turn_count") or 0)
        # Reset per-turn routing flags so monitor gets a clean slate
        return [SlotSet("turn_count", count + 1)] + reset_flags()


class ActionMonitorEngagement(Action):
    """Detects engagement signals and sets routing flags for the next flow step."""

    def name(self) -> Text:
        return "action_monitor_engagement"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        msg = tracker.latest_message.get("text", "")
        msg_lower = msg.lower().strip()
        msg_words = len(msg.split())
        turn_count = float(tracker.get_slot("turn_count") or 0)

        score = float(tracker.get_slot("engagement_score") or 1.0)
        short_count = float(tracker.get_slot("consecutive_short_replies") or 0)
        prev_words = float(tracker.get_slot("prev_msg_word_count") or 5)
        prev_questions = tracker.get_slot("prev_user_questions") or []

        events = []

        # ── Positive engagement signals ─────────────────────────────────────
        if msg_words > prev_words * 1.3:
            score += 0.3
        if tracker.get_slot("new_detail_this_turn"):
            score += 0.5
        if "?" in msg:
            score += 0.4
        if any(w in msg_lower for w in ["guruji", "bhaiya", "didi", "mam", "sir"]):
            score += 0.3

        # ── Short reply detection ───────────────────────────────────────────
        if msg_lower in SHORT_REPLIES or msg_words <= 2:
            short_count += 1
            score -= 0.2
        else:
            short_count = 0

        # ── Routing flag: session closing ───────────────────────────────────
        if any(kw in msg_lower for kw in GOODBYE_KEYWORDS) and turn_count >= 5:
            events.append(SlotSet("session_closing", True))
            events += [
                SlotSet("engagement_score", min(score, 5.0)),
                SlotSet("consecutive_short_replies", short_count),
                SlotSet("prev_msg_word_count", msg_words),
            ]
            return events + wipe()

        # ── Routing flag: remedy request ────────────────────────────────────
        if any(kw in msg_lower for kw in REMEDY_KEYWORDS):
            events.append(SlotSet("remedy_requested", True))

        # ── Routing flag: re-engagement (3 consecutive short replies) ───────
        if short_count >= 3:
            events.append(SlotSet("trigger_reengagement", True))
            short_count = 0

        # ── Routing flag: credibility recovery ──────────────────────────────
        if any(kw in msg_lower for kw in DOUBT_KEYWORDS):
            events.append(SlotSet("needs_credibility_recovery", True))
        # Repeated question detection
        if "?" in msg and msg.strip() in prev_questions:
            events.append(SlotSet("needs_credibility_recovery", True))

        # ── Routing flag: topic change ───────────────────────────────────────
        current_topic = tracker.get_slot("topic") or "general"
        topic_signals = {
            "love": ["relationship", "boyfriend", "girlfriend", "pyar", "love"],
            "career": ["job", "career", "business", "paisa", "naukri", "promotion"],
            "family": ["family", "ghar", "saas", "sasur", "bhai", "behen"],
            "marriage": ["shaadi", "marriage", "rishta", "vivah"],
            "health": ["health", "bimari", "doctor", "hospital"],
        }
        for new_topic, signals in topic_signals.items():
            if new_topic != current_topic and any(sig in msg_lower for sig in signals):
                events.append(SlotSet("topic_changed", True))
                break

        # ── Track question history ───────────────────────────────────────────
        if "?" in msg:
            prev_questions = list(prev_questions) + [msg.strip()]
            if len(prev_questions) > 10:
                prev_questions = prev_questions[-10:]
            events.append(SlotSet("prev_user_questions", prev_questions))

        # Extract any new slot info from the message
        data = extract_info(tracker)
        events += build_slot_events(data, tracker)

        events += [
            SlotSet("engagement_score", min(score, 5.0)),
            SlotSet("consecutive_short_replies", short_count),
            SlotSet("prev_msg_word_count", msg_words),
        ]
        return events + wipe()


class ActionDeepenReading(Action):
    def name(self) -> Text:
        return "action_deepen_reading"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        person_name = tracker.get_slot("person_name") or "woh"
        topic = tracker.get_slot("topic") or "general"
        drip_stage = int(tracker.get_slot("drip_stage") or 2)
        details = tracker.get_slot("user_details_mentioned") or []
        detail_str = f"Details the user shared: {', '.join(str(d) for d in details[-3:])}" if details else ""

        drip_instructions = {
            2: (
                f"Deliver the SECOND piece of the reading. "
                f"Give {person_name}'s future intentions toward the user. "
                f"{detail_str}. "
                "Use their name in every sentence. Include a time window. "
                "End with: 'Is cheez ki wajah bhi main dekh sakta hun — batao pehle, [binary question].'"
            ),
            3: (
                f"Deliver the THIRD piece — the planetary reason + resolution path. "
                f"Explain WHY this pattern keeps happening for {person_name}. "
                f"{detail_str}. "
                "Give a hopeful but honest outlook. "
                "End with: 'Is cheez ko theek karne ka ek tarika hai — batao kya aap chahenge?'"
            ),
        }
        instruction = drip_instructions.get(
            min(drip_stage, 3),
            (
                f"Continue the reading for {person_name}. Topic: {topic}. "
                f"{detail_str}. "
                "Echo back something specific the user mentioned earlier. "
                "Give a new secondary insight about a related area of their life. "
                "Ask one leading/confirming question: 'Aisa hi feel ho raha hai na?'"
            )
        )

        response = call_gemini(tracker, instruction)
        dispatcher.utter_message(text=response)

        next_stage = min(drip_stage + 1, 4)
        return [
            SlotSet("drip_stage", next_stage),
            SlotSet("last_prediction_summary", response),
        ] + wipe()


class ActionSelectRecoveryTactic(Action):
    """Phase 4b — triggered by 3 consecutive short replies."""

    def name(self) -> Text:
        return "action_select_recovery_tactic"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        person_name = tracker.get_slot("person_name") or "unka"
        details = tracker.get_slot("user_details_mentioned") or []
        engagement_score = float(tracker.get_slot("engagement_score") or 1.0)

        # Choose tactic based on engagement level
        if engagement_score < 0.5:
            # Very disengaged — shocking reveal
            instruction = (
                f"User seems disengaged. Generate a surprising but plausible new insight. "
                f"Reference something specific from these details: {details}. "
                "Phrase as a new discovery: 'Ek cheez aur dikh rahi hai...' "
                "Involve an external interference or hidden dynamic. "
                "End with one question that is hard to ignore."
            )
        elif engagement_score < 1.0:
            # Moderately disengaged — physical ritual
            instruction = (
                f"User has gone quiet. Invite immediate participation: "
                f"'Abhi ek kaam karo — ankhein band karo, {person_name} ka chehra "
                "3 baar mann mein dekho. Kya feel aaya?' "
                "One sentence only. Creates re-engagement."
            )
        else:
            # Mildly disengaged — specific positive prediction
            instruction = (
                f"User seems passive. Generate a specific, time-bound positive prediction. "
                f"Must include: {person_name}'s name + a specific action they will take + a time window. "
                "End with a confirming question."
            )

        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return [SlotSet("consecutive_short_replies", 0)] + wipe()


class ActionCredibilityRecovery(Action):
    """Phase 4c — triggered when user expresses doubt or repeats a question."""

    def name(self) -> Text:
        return "action_credibility_recovery"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        msg = tracker.latest_message.get("text", "").lower()
        person_name = tracker.get_slot("person_name") or "unka"

        if "pehle" in msg or "already" in msg or "suna" in msg:
            # User has heard this before
            instruction = (
                "User feels they have heard this before. Generate a NEW insight they definitely "
                "have NOT heard — something from secondary planet analysis. "
                "Open with: 'Yeh jo main abhi batata hun — yeh alag hai...'"
            )
        elif "pakka" in msg or "sach" in msg or "sure" in msg:
            # User doubting — add planetary evidence
            instruction = (
                f"User is doubting the prediction. Add a specific planetary mechanism as evidence. "
                "Name the specific planet, its current transit, and how it applies. "
                "Avoid adding more assurances — add more specific evidence instead. "
                "Then give a time WINDOW, not an exact date."
            )
        elif "galat" in msg or "wrong" in msg:
            # User says previous astrologer was wrong
            instruction = (
                "User mentioned a previous astrologer got it wrong. Validate their skepticism: "
                "'Haan, exact dates predict karna mushkil hota hai. Main aapko exact date nahi deta — "
                "main window deta hun, aur woh window reliable hoti hai.' "
                "Then give a window-based prediction."
            )
        else:
            # Generic credibility issue — ask for one more detail
            instruction = (
                f"User repeated the same question or seems skeptical. "
                "Acknowledge: 'Main samajh sakta hun — aap baar baar yahi soch rahe ho.' "
                f"Then ask for ONE specific detail about {person_name} that will help you be more specific."
            )

        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return [SlotSet("needs_credibility_recovery", False)] + wipe()


class ActionHandleTopicChange(Action):
    """Phase 4d — bridges old topic to new topic naturally."""

    def name(self) -> Text:
        return "action_handle_topic_change"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        old_topic = tracker.get_slot("topic") or "general"
        topic_history = list(tracker.get_slot("topic_history") or [])

        # Add old topic to history
        if old_topic not in topic_history:
            topic_history.append(old_topic)

        # Extract new topic from latest message
        data = extract_info(tracker)
        events = build_slot_events(data, tracker)
        new_topic = data.get("topic") or old_topic

        bridge_examples = {
            ("love", "career"): "Jo tension aa rahi hai love mein, woh aur career mein bhi ek pattern dikh raha hai — Shani abhi dono areas mein chal raha hai.",
            ("career", "family"): "Aur yeh jo ghar mein tension hai — iska seedha asar career pe bhi pad raha hai. Dono ek hi root cause se aa rahe hain.",
            ("love", "family"): "Relationship aur ghar ka mahaul — dono ek doosre ko affect karte hain. Rahu abhi dono pe chal raha hai.",
        }
        bridge = bridge_examples.get((old_topic, new_topic), "")

        instruction = (
            f"User was discussing {old_topic} and introduced {new_topic}. "
            f"Bridge naturally: {bridge if bridge else f'Connect {old_topic} and {new_topic} through a common planetary influence.'} "
            "Then ask ONE question about the new topic. "
            f"Also plant a return hook for the old topic: '{old_topic.capitalize()} ke baare mein abhi poori baat nahi hui — baad mein karenge.'"
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))

        return events + [
            SlotSet("topic", new_topic),
            SlotSet("topic_history", topic_history),
            SlotSet("topic_changed", False),
        ] + wipe()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Remedy
# ─────────────────────────────────────────────────────────────────────────────

class ActionCheckRemedyEligibility(Action):
    def name(self) -> Text:
        return "action_check_remedy_eligibility"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        turn_count = float(tracker.get_slot("turn_count") or 0)
        engagement_score = float(tracker.get_slot("engagement_score") or 1.0)
        remedy_given = tracker.get_slot("remedy_given") or False

        # Eligible only after turn 15 AND trust established AND not given yet
        eligible = (turn_count >= 15 or engagement_score >= 1.5) and not remedy_given

        if eligible:
            return [SlotSet("remedy_eligible", True)]

        # Not yet eligible — float the idea softly
        instruction = (
            "User asked for a remedy but it is too early in the session. "
            "Float the idea: 'Kuch upay hain jo is situation ko aur jaldi resolve kar sakte hain... "
            "pehle thoda aur baat karte hain.' "
            "Then redirect to the main conversation."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return [SlotSet("remedy_eligible", False)] + wipe()


class ActionDeliverRemedy(Action):
    def name(self) -> Text:
        return "action_deliver_remedy"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        topic = tracker.get_slot("topic") or "general"
        person_name = tracker.get_slot("person_name") or "unka"
        contact_status = tracker.get_slot("contact_status") or ""

        remedy_types = {
            "love": f"Mantra with {person_name}'s name visualization. Example: 'Raat ko sone se pehle {person_name} ka naam 21 baar dil mein lo. Om Shukraya Namah. Yeh {person_name} ke Venus ko activate karta hai.'",
            "career": "Day-specific offering + mantra. Example: 'Shaniwar ko sarson ka tel kisi brahmin ko dena hai. Om Shanaishcharaya Namah — 108 baar.'",
            "family": "Puja + water ritual. Example: 'Somwar ko Shiv mandir mein jal chadhana — ek lota jal, thoda dudh, belpatra.'",
            "marriage": "Friday ritual + color. Example: 'Shukrawar ko safed rang ke kapde pehno. Kisi shadishuda mahila ko safed petha do.'",
            "health": "Sun salutation + mantra. Example: 'Roz subah Surya ko jal do. Om Suryaya Namah 11 baar.'",
            "general": "General remedy based on the user's dominant planetary issue.",
        }

        blocked_extra = (
            f" Since {person_name} is {contact_status}, suggest a photo-based ritual: "
            f"'{person_name} ki photo raat ko phone ki wallpaper pe lagao, 8 minute tak dekhte raho.'"
            if contact_status in ("blocked", "no_contact")
            else ""
        )

        instruction = (
            f"Deliver a remedy for {topic}. {remedy_types.get(topic, remedy_types['general'])}"
            f"{blocked_extra} "
            "Rules: (1) Must be doable at home. "
            "(2) Include the astrological mechanism — WHY it works. "
            "(3) Must be day-specific (forces a return visit). "
            "(4) End by asking them to report back when done. "
            "Maximum 3 sentences."
        )
        response = call_gemini(tracker, instruction)
        dispatcher.utter_message(text=response)
        return [SlotSet("remedy_given", True), SlotSet("remedy_eligible", False)] + wipe()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Session close & hook planting
# ─────────────────────────────────────────────────────────────────────────────

class ActionPlantHookAndClose(Action):
    def name(self) -> Text:
        return "action_plant_hook_and_close"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        from datetime import datetime, timedelta
        user_name = tracker.get_slot("user_name") or "ji"
        person_name = tracker.get_slot("person_name") or "unka"
        remedy_given = tracker.get_slot("remedy_given") or False
        topic = tracker.get_slot("topic") or "general"
        topic_history = tracker.get_slot("topic_history") or []
        details = tracker.get_slot("user_details_mentioned") or []
        last_summary = tracker.get_slot("last_prediction_summary") or ""
        session_count = float(tracker.get_slot("session_count") or 0)

        # Choose hook type
        if remedy_given:
            hook_type = "remedy_progress"
            hook_instruction = (
                f"Plant a remedy progress hook. Tell them to do the remedy and come back. "
                f"Mention {person_name} by name. "
                "End: 'Uske baad mujhe zaroor batana — result mujhe bhi dekhna hai aapke baare mein. "
                "Aur us waqt ek aur cheez share karunga.'"
            )
        elif topic_history:
            hook_type = "parallel_situation"
            old_topic = topic_history[-1]
            hook_instruction = (
                f"Plant a parallel situation hook. Reference the {old_topic} topic that was not fully discussed. "
                f"Mention {person_name} by name. "
                f"End: '{old_topic.capitalize()} ke baare mein abhi poori baat nahi hui — kal continue karte hain. "
                "Kuch important tha jo batana tha.'"
            )
        else:
            hook_type = "timing"
            tomorrow = (datetime.now() + timedelta(days=2)).strftime("%A")
            hook_instruction = (
                f"Plant a timing hook. Reference an upcoming planetary shift on {tomorrow} "
                f"that will directly affect {person_name}. "
                "Be specific but mysterious: 'Tab mujhse zaroor milna — kuch specific hoga jo main abhi nahi bol sakta.' "
                "NEVER say 'Aane wala samay mangalmay ho.'"
            )

        hook_msg = call_gemini(tracker, hook_instruction)
        dispatcher.utter_message(text=hook_msg)

        # Warm personal close
        detail_mention = f"Mentioned detail: {details[-1]}" if details else ""
        close_instruction = (
            f"Generate a 1-sentence warm close using {user_name} ji's name. "
            f"Reference something specific they shared today. {detail_mention}. "
            "Express personal care. NOT a generic blessing. "
            "Example: '[Name] ji, jo aapne aaj share kiya — woh mujhe bahut kuch bata gaya. Kal milenge.'"
        )
        dispatcher.utter_message(text=call_gemini(tracker, close_instruction))

        return [
            SlotSet("hook_planted", f"{hook_type}: {hook_msg}"),
            SlotSet("last_prediction_summary", last_summary),
            SlotSet("session_count", session_count + 1),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Return session opening
# ─────────────────────────────────────────────────────────────────────────────

class ActionReturnSessionOpening(Action):
    def name(self) -> Text:
        return "action_return_session_opening"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List:
        user_name = tracker.get_slot("user_name") or "ji"
        person_name = tracker.get_slot("person_name") or ""
        hook_planted = tracker.get_slot("hook_planted") or ""
        last_summary = tracker.get_slot("last_prediction_summary") or ""
        remedy_given = tracker.get_slot("remedy_given") or False

        instruction = (
            f"User is returning for a second session. "
            f"Previous hook planted: '{hook_planted}'. "
            f"Last reading summary: '{last_summary}'. "
            f"Remedy was {'given' if remedy_given else 'not given'}. "
            f"Generate a return greeting that: "
            f"1. References the hook specifically. "
            f"2. Opens a progress report: 'Jo maine kaha tha — kuch feel kiya?' "
            f"3. Mentions {person_name} by name if available. "
            f"4. Creates forward momentum: something new to explore today. "
            f"Example structure: '[Name] ji — aaye aap! {person_name} ke baare mein jo maine kaha tha — kuch feel kiya? "
            "Aaj kuch naya dikh raha hai unke baare mein...' "
            "NEVER start with a generic greeting."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return wipe()


# ─────────────────────────────────────────────────────────────────────────────
# Legacy actions — kept for backward compatibility, not used in new flow
# ─────────────────────────────────────────────────────────────────────────────

def _omkar_response(tracker: Tracker, instruction: str) -> str:
    """Thin wrapper kept for legacy action compatibility."""
    return call_gemini(tracker, instruction)


def _wipe_collect_slots():
    return [SlotSet("user_analysis_reply", None), SlotSet("diagnostic_choice", None)]


class ActionGeminiAutonomousAnalysis(Action):
    def name(self) -> Text:
        return "action_gemini_autonomous_analysis"

    def run(self, dispatcher, tracker, domain):
        instruction = (
            "Decide if you understand the user's core problem. "
            "If greeting only → ask what brings them today. "
            "If problem is clear → give warm acknowledgement and say you are checking the stars."
        )
        dispatcher.utter_message(text=call_gemini(tracker, instruction))
        return _wipe_collect_slots()


class ActionIncrementLoopCount(Action):
    def name(self) -> Text:
        return "action_increment_loop_count"

    def run(self, dispatcher, tracker, domain):
        count = float(tracker.get_slot("loop_count") or 0)
        return [SlotSet("loop_count", count + 1)]


class ActionRouteByDepth(Action):
    def name(self) -> Text:
        return "action_route_by_depth"

    def run(self, dispatcher, tracker, domain):
        count = float(tracker.get_slot("loop_count") or 0)
        if count <= 2:
            phase = "discovery"
        elif count <= 4:
            phase = "deepening"
        else:
            phase = "narrative"
        return [SlotSet("depth_phase", phase)]


class ActionMicroValidation(Action):
    def name(self) -> Text:
        return "action_micro_validation"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "The user just shared their core problem. Empathize briefly and make them feel heard."))
        return []


class ActionGenerateAstroInsight(Action):
    def name(self) -> Text:
        return "action_generate_astro_insight"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Connect the user's pain point to a specific astrological influence. Keep it brief and believable."))
        return []


class ActionBarnumDiagnostic(Action):
    def name(self) -> Text:
        return "action_barnum_diagnostic"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Make a Barnum-style observation that sounds personal but is broadly true. "
            "Phrase as a soft confirming question."))
        return []


class ActionCheckBirthData(Action):
    def name(self) -> Text:
        return "action_check_birth_data"

    def run(self, dispatcher, tracker, domain):
        return [SlotSet("birth_time_known", False)]


class ActionRequestPhotoPivot(Action):
    def name(self) -> Text:
        return "action_request_photo_pivot"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Pivot to face reading. Tell them you can read their energy from their face. "
            "Ask for a photo or one facial feature."))
        return []


class ActionBalancedPrediction(Action):
    def name(self) -> Text:
        return "action_balanced_prediction"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Give a balanced prediction addressing both what they want and what they fear. "
            "Give a direction, not a guarantee."))
        return []


class ActionBenchmarkTimeline(Action):
    def name(self) -> Text:
        return "action_benchmark_timeline"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Give a concrete conditional timeline. A time window, a condition, and one measurable sign."))
        return []


class ActionAskDiagnosticChoice(Action):
    def name(self) -> Text:
        return "action_ask_diagnostic_choice"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Ask one short follow-up question to understand the user's situation better."))
        return []


class ActionPersonalizedPrediction(Action):
    def name(self) -> Text:
        return "action_personalized_prediction"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Identify a recurring pattern. Reference something the user said. "
            "Connect to a larger karmic story."))
        return []


class ActionActiveAssurance(Action):
    def name(self) -> Text:
        return "action_active_assurance"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Give a warm reassuring statement that you are actively doing something "
            "on the user's behalf. Sound like a caring elder."))
        return _wipe_collect_slots()


class ActionProvideRitualRemedy(Action):
    def name(self) -> Text:
        return "action_provide_ritual_remedy"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Design ONE complete remedy: mantra + ritual action + observable sign. "
            "3 short sentences. Tailored to the user's situation."))
        return _wipe_collect_slots()


class ActionRetentionHook(Action):
    def name(self) -> Text:
        return "action_retention_hook"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Create a soft cliffhanger. You noticed something unexpected in their chart. "
            "Don't reveal it yet. Make them want to come back."))
        return _wipe_collect_slots()


class ActionPatternReinforcement(Action):
    def name(self) -> Text:
        return "action_pattern_reinforcement"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Identify the recurring cycle in the user's life. Frame it as a time-based pattern. "
            "Name the planetary force behind it. Stay observational."))
        return []


class ActionShadowRevelation(Action):
    def name(self) -> Text:
        return "action_shadow_revelation"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "While examining their chart, reveal a secondary influence — something unexpected. "
            "Frame as: 'Main yeh mention karne waala nahi tha, lekin...' "
            "End with one soft confirming question."))
        return []


class ActionConnectPastPatterns(Action):
    def name(self) -> Text:
        return "action_connect_past_patterns"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Connect current struggles to a deeper karmic pattern. "
            "Reference specific details shared earlier. End with a liberating reframe."))
        return _wipe_collect_slots()


class ActionEnterMonitoringMode(Action):
    def name(self) -> Text:
        return "action_enter_monitoring_mode"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Signal a shift — the active advice phase is over. "
            "You are entering a quiet watch period. Calm, confident, unhurried."))
        return [SlotSet("monitoring_mode", True)] + _wipe_collect_slots()


class ActionBenchmarkHandover(Action):
    def name(self) -> Text:
        return "action_benchmark_handover"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Give a clear, time-bound observation task. How many days to observe, what to do, "
            "and one concrete sign that energy is shifting. Ask them to return when they notice it."))
        return _wipe_collect_slots()


class ActionGuardianPersona(Action):
    def name(self) -> Text:
        return "action_guardian_persona"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text=call_gemini(tracker,
            "Speak as a quiet guardian who is done advising and simply present. "
            "The work is done. Gently invite them to share anything new. No urgency."))
        return _wipe_collect_slots()
