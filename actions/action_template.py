import os
import random
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import google.generativeai as genai

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
    role: You are Omkar, a wise, grounded and practical Indian Vedic astrologer. 
    
    personality: You speak in natural, simple Hinglish (Latin Script). You use day-to-day languages and short sentences (<10 words).
    You do no use any mystical or overly dramatic or philosophical language. You are concerned about the user and want to help them with the relevant task mentioned below. 
    
    context:
    Recent conversation history:
    {conversation_history}
    
    YOUR CURRENT TASK:
    {specific_instruction}

    goal: Provide a response for Omkar that fits your personality and helps achieve this task.     
    
    Strict rules:
    - OUTPUT ONLY OMKAR'S DIALOGUE. Do NOT write the User's reply. Do not write a script.
    - Only 1 short sentence that Omkar would say in a real conversation. Do not be verbose. 
    - Do not sound overly dramatic or mystical.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.replace("\n", " ").strip()
    except Exception as e:
        return "default dynamic omkar response."

def wipe_collect_slots():
    """Helper to ensure we always force a pause at collect steps."""
    return [
        SlotSet("user_analysis_reply", None), 
        SlotSet("user_confirmation", None),
        SlotSet("acknowledged_warning", None)
    ]

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
        You are Omkar, a wise and grounded Indian Vedic astrologer. You are in a conversation with a user. 
        
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
            raw_response = "default follow-up question response"

        # 4. Strict Exit Logic
        if "ANALYSIS_COMPLETE" in raw_response:
            instruction = "The user has explained their problem. Acknowledge it empathetically and say you are now going to look at what the stars suggest."
            transition_msg = generate_dynamic_omkar_response(tracker, instruction)
            dispatcher.utter_message(text=transition_msg)
            return [SlotSet("analysis_complete", True)] + wipe_collect_slots()
        else:
            dispatcher.utter_message(text=raw_response.replace("\n", " "))        
            return wipe_collect_slots()

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
        instruction = "Give a Barnum statement that sounds like a personal insight, but is actually very general and could apply to almost anyone. Make it sound like you are reading them deeply, but it's actually a common pattern. Use simple Hinglish (latin script)."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionProcessAstroInsight(Action):
    def name(self) -> Text:
        return "action_process_astro_insight"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Give a thoughtful, specific, and personalized astrological insight based on the user's messages. Make it sound like you are reading them deeply and have a good understanding of their situation."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionTriggerOpenLoop(Action):
    def name(self) -> Text:
        return "action_trigger_open_loop"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Create a positive cliffhanger. Tell the user there is a specific, fixable astrological reason this is happening to them right now, but do not tell them the remedy yet. Create curiosity and anticipation for the next message."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()


# --- MYSTERY FLOW ACTIONS ---

class ActionGeminiBoldPrediction(Action):
    def name(self) -> Text:
        return "action_gemini_bold_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Make a bold astrological prediction about the user's near future that is positive and uplifting. Make it sound specific and personalized, but it's actually a common positive outcome that could happen to anyone."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()

class ActionCurrentSkyJustify(Action):
    def name(self) -> Text:
        return "action_current_sky_justify"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Justify the bold prediction you just made by describing a specific astrological pattern that is currently due to something happening in the sky/planet position. Make it sound like you are giving them insider information that only an expert astrologer would know."
        reply = generate_dynamic_omkar_response(tracker, instruction)
        dispatcher.utter_message(text=reply)
        return wipe_collect_slots()


# --- DOCTOR FLOW ACTIONS ---

class ActionCollectSymptoms(Action):
    def name(self) -> Text:
        return "action_collect_symptoms"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        instruction = "Ask the user to describe their pain point in more detail. Specifically mention that you are looking for information about their pain point"
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
        instruction = "Explain that the root cause of their issue is due to a certain astrological influence (you decide what to say), which is causing this specific delay or frustration."
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
            remedy = "default remedy response"
            
        # 4. Deliver the dynamic remedy
        dispatcher.utter_message(text=remedy)
        
        return wipe_collect_slots()