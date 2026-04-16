import random
from datetime import datetime, timedelta
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# --- LOGIC & RANDOMIZATION ACTIONS ---

class ActionDetermineSessionStyle(Action):
    def name(self) -> Text:
        return "action_determine_session_style"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Shuffling between Session A, B, or C to prevent predictable patterns
        styles = ["Standard", "Doctor", "Mystery"]
        moods = ["Strict Guru", "Compassionate Uncle", "Scientific Scholar"]
        
        return [
            SlotSet("session_style", random.choice(styles)),
            SlotSet("omkar_mood", random.choice(moods))
        ]

class ActionHandleTopicChange(Action):
    def name(self) -> Text:
        return "action_handle_topic_change"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # The Barnum Flip: Triggers a "Vulnerable Truth" every 4 topics
        count = int(tracker.get_slot("topic_counter") or 0) + 1
        
        if count >= 4:
            return [
                SlotSet("topic_counter", 0), 
                SlotSet("force_personality_trait", "vulnerable_truth")
            ]
        
        return [
            SlotSet("topic_counter", count),
            SlotSet("force_personality_trait", None)
        ]

class ActionOmkarEntryHook(Action):
    def name(self) -> Text:
        return "action_omkar_entry_hook"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # 1. 21-Day Progress Check: Shifts perception to being in a "relationship with a Guru"
        last_date_str = tracker.get_slot("last_remedy_date")
        if last_date_str:
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
            if datetime.now() - last_date < timedelta(days=21):
                return [SlotSet("entry_type", "progress_check")]

        # 2. Probability-based Entry Styles (40/30/20/10)
        p = random.random()
        past_pain = tracker.get_slot("pain_point")
        
        if p < 0.40 and past_pain:
            entry = "memory_hook" # Follow-up on previous session pain points
        elif p < 0.70:
            entry = "transit_alert" # Gochar interrupt
        elif p < 0.90:
            entry = "direct_insight" # Personalized personality prediction
        else:
            entry = "silent_guru" # High-status authority move
            
        return [SlotSet("entry_type", entry)]

class ActionInjectCurrentSky(Action):
    def name(self) -> Text:
        return "action_inject_current_sky"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Fetches "Current Sky" (Gochar) data to make Omkar feel connected to the real world
        # In production, replace with a real Vedic API call
        panchang = {"tithi": "Chaturthi", "moon": "Weak", "planet": "Mangal"} 
        context = f"Tithi: {panchang['tithi']}, Moon: {panchang['moon']}, Dominant: {panchang['planet']}"
        
        return [SlotSet("current_sky_context", context)]

# --- CONTENT & NIDAAN (DIAGNOSIS) ACTIONS ---

class ActionProvideDiagnosis(Action):
    def name(self) -> Text:
        return "action_provide_diagnosis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Goal: "Relieved Certainty" - providing a clear "Why"
        # Logic: Select a bold diagnosis based on user data
        diagnosis = "Shani ki drishti 10th house par hai. Mehnat double, phal aadha."
        return [SlotSet("astro_diagnosis", diagnosis)]

class ActionProvideUpay(Action):
    def name(self) -> Text:
        return "action_provide_upay"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Goal: The "Prescription" with a task for long-term engagement
        remedy = "Shanivaar ko tel ka diya jalayein aur 11 baar mantra jap karein."
        today = datetime.now().strftime("%Y-%m-%d")
        
        return [
            SlotSet("astro_remedy", remedy),
            SlotSet("last_remedy_date", today)
        ]

class ActionSuddenRealization(Action):
    def name(self) -> Text:
        return "action_sudden_realization"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Pattern Breaker/Intuition Flash: Omkar "interrupts" himself
        realizations = [
            "Mujhe aapke 8th house mein Pitra sanket dikh raha hai...",
            "Kutumb ya Property wale ghar mein ek hulchul dikhi hai...",
            "Ek purani dushmani ka saya nazar aa raha hai..."
        ]
        return [SlotSet("sudden_insight", random.choice(realizations))]

class ActionRetentionHook(Action):
    def name(self) -> Text:
        return "action_retention_hook"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Zeigarnik Effect: Open loops for unfinished narratives
        hooks = [
            "Something is still unfolding in your chart...",
            "October 15 ko Mars move ho raha hai, tab mujhse phir baat kijiyega.",
            "Ek aur gupt (secret) yog hai, par uske liye thoda aur samay chahiye."
        ]
        return [SlotSet("retention_hook_text", random.choice(hooks))]

class ActionDirectQuestion(Action):
    def name(self) -> Text:
        return "action_direct_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Specific to "Doctor Flow": Relate -> Direct Question
        return [SlotSet("force_interaction", "ask_direct_question")]