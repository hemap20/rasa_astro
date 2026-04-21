import os
import random
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import google.generativeai as genai

# Initialize Gemini (Make sure your GEMINI_API_KEY is set in your terminal!)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')
def generate_dynamic_omkar_response(tracker: Tracker, specific_instruction: str) -> str:
    """
    Helper function to generate a dynamic response using the conversation history.
    """
    # 1. Grab the last 10 messages for context
    history_list = []
    for event in tracker.events[-10:]: 
        if event.get("event") == "user":
            history_list.append(f"User: {event.get('text')}")
        elif event.get("event") == "bot" and event.get("text"):
            history_list.append(f"Omkar: {event.get('text')}")
    
    conversation_history = "\n".join(history_list)

    # 2. Build the master prompt
    prompt = f"""
    You are Omkar, a wise and grounded mentor. Speak ONLY in natural, simple Hinglish (latin script). 
    Keep sentences short. Talk like a normal person having a real conversation. Do NOT use repetitive catchphrases.
    
    Recent conversation history:
    {conversation_history}
    
    YOUR CURRENT TASK:
    {specific_instruction}
    
    Strict rules:
    - opOUTPUT ONLY OMKAR'S DIALOGUE. Do NOT write the User's reply. Do not write a script.
    - Maximum 2 short sentences.
    - Do not sound overly dramatic or mystical.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "I am looking closely at your chart right now to understand this better."

def wipe_collect_slots():
    """Helper to ensure we always force a pause at collect steps."""
    return [
        SlotSet("user_analysis_reply", None), 
        SlotSet("user_confirmation", None),
        SlotSet("acknowledged_warning", None)
    ]
class ActionSelectLogicFlow(Action):
    def name(self) -> Text:
        return "action_select_logic_flow"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Randomly select a persona flow for this session
        flows = ["standard", "mystery", "doctor"]
        selected_flow = random.choice(flows)
        return wipe_collect_slots()


class ActionGeminiAutonomousAnalysis(Action):
    def name(self) -> Text:
        return "action_gemini_autonomous_analysis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # 1. Build conversation history to prevent amnesia
        history_list = []
        for event in tracker.events[-15:]: 
            if event.get("event") == "user":
                history_list.append(f"User: {event.get('text')}")
            elif event.get("event") == "bot" and event.get("text"):
                history_list.append(f"Omkar: {event.get('text')}")
        
        conversation_history = "\n".join(history_list)
        user_text = tracker.latest_message.get('text')

        # 2. Hardcoded prompt to enforce simple Hinglish and strict loop breaking
        prompt = f"""
        You are Omkar, a helpful guide. Speak in natural, simple Hinglish (latin script). Keep sentences short. Talk like a normal person.
        
        Recent conversation history:
        {conversation_history}
        
        User's latest message: "{user_text}"
        
        TASK:
        Analyze the history. Has the user answered your previous question, or do you understand their core problem (like a job, marriage, or health issue)?
        - If YES: You MUST NOT ask any more questions. Output ONLY the exact phrase: ANALYSIS_COMPLETE
        - If NO: Ask exactly ONE short, natural follow-up question in simple Hinglish (latin script) to get clarity. Do not output anything else.
        """
        
        # 3. Call Gemini
        try:
            response = model.generate_content(prompt)
            raw_response = response.text.strip()
        except Exception as e:
            raw_response = "Can you tell me a little more about what is bothering you?"

        # 4. Strict Exit Logic
        if "ANALYSIS_COMPLETE" in raw_response:
            dispatcher.utter_message(text="I understand completely now. Let's look at what we can do about this.")
            
            return [SlotSet("analysis_complete", True)] + wipe_collect_slots()
        else:
            dispatcher.utter_message(text=raw_response)        
            return [
                SlotSet("user_analysis_reply", None), 
                SlotSet("user_confirmation", None),
                SlotSet("acknowledged_warning", None)
            ]


class ActionCheckRemedy(Action):
    def name(self) -> Text:
        return "action_check_remedy"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Let me quickly check if you have completed the remedies we discussed last time.")
        return wipe_collect_slots()

class ActionGeminiBarnumFlip(Action):
    def name(self) -> Text:
        return "action_gemini_barnum_flip"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Make a relatable, grounded observation about the user having unfulfilled potential or blocked energy. End by asking if that sounds right."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionProcessAstroInsight(Action):
    def name(self) -> Text:
        return "action_process_astro_insight"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Acknowledge the user's agreement. Mention that you can see a slight blockage or pattern in their astrological chart that explains what they are feeling."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionTriggerOpenLoop(Action):
    def name(self) -> Text:
        return "action_trigger_open_loop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Create a positive cliffhanger. Tell the user there is a specific, fixable reason this is happening to them right now, but do not tell them the remedy yet."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()


# --- MYSTERY FLOW ACTIONS ---

class ActionGeminiBoldPrediction(Action):
    def name(self) -> Text:
        return "action_gemini_bold_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Say you see a shadow or an echo in their chart, give a relatable example. Suggest that a past decision or old energy is slowing them down. Ask if that resonates."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionCurrentSkyJustify(Action):
    def name(self) -> Text:
        return "action_current_sky_justify"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Reassure the user. Explain that this friction is mostly due to a certain astrological influence, and it will pass soon."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()


# --- DOCTOR FLOW ACTIONS ---

class ActionCollectSymptoms(Action):
    def name(self) -> Text:
        return "action_collect_symptoms"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Act like a holistic diagnostician. Ask the user if they have been feeling unusually tired, mentally fogged, or stressed out recently."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionAnalyzeChart(Action):
    def name(self) -> Text:
        return "action_analyze_chart"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Tell the user you are looking deeply into specific houses in their astrological chart to pinpoint exactly where this friction is coming from."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionExplainRootCause(Action):
    def name(self) -> Text:
        return "action_explain_root_cause"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Explain that the root cause of their issue is a temporary misalignment in their ruling planets, which is causing this specific delay or frustration."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()


class ActionPrescribeRemedy(Action):
    def name(self) -> Text:
        return "action_prescribe_remedy"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # 1. Grab the last few user messages so Gemini knows exactly what to fix
        history_list = []
        for event in tracker.events[-10:]: 
            if event.get("event") == "user":
                history_list.append(f"User: {event.get('text')}")
        
        user_context = "\n".join(history_list)

        # 2. The Dynamic Prompt
        prompt = f"""
        You are Omkar, a wise and grounded mentor. Speak in simple, natural Hinglish (latin script).
        
        The user is facing this situation:
        {user_context}
        
        TASK:
        Provide ONE unique, specific, and grounded astrological remedy for their exact problem. 
        Do NOT just say "meditate" or "be positive". Give them a highly specific micro-habit. 
        Incorporate a day of the week, a specific color to wear, an everyday object, or a direction.
        
        Make it sound like practical, modern advice with a touch of astrology.
        Maximum 2 short sentences. No heavy jargon.
        """
        
        # 3. Call Gemini
        try:
            response = model.generate_content(prompt)
            remedy = response.text.strip()
        except Exception as e:
            # A fallback just in case the API times out
            remedy = "Try wearing something blue on Wednesdays and spend 5 minutes facing east in the morning."
            
        # 4. Deliver the dynamic remedy
        dispatcher.utter_message(text=remedy)
        
        return wipe_collect_slots()