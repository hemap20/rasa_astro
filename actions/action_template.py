import os
import random 
import google.generativeai as genai
from jinja2 import Template 
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from .db import get_user_profile, update_user_profile
from typing import Optional, Any, Dict, List, Text

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

class ActionGeminiAutonomousAnalysis(Action):
    def name(self):
        return "action_gemini_autonomous_analysis"

    async def run(self, dispatcher, tracker, domain):
        # Load the Jinja template
        with open("prompts/astro_analysis_step.jinja2") as f:
            template = Template(f.read())
        
        # Render with current context
        prompt = template.render(
            user_text=tracker.latest_message.get('text'),
            slots=tracker.slots,
            current_flow_name=tracker.active_loop_name # Note: May be None in Rasa Flows
        )
        
        # Fixed the undefined call_gemini function
        response = model.generate_content(prompt)
        raw_response = response.text 
        
        # Split the text into a list of sentences based on newlines
        messages = [msg.strip() for msg in raw_response.split('\n') if msg.strip()]
        for message in messages[:3]: # Ensure max 3 messages
            dispatcher.utter_message(text=message)        
        
        # Fixed the undefined 'response' variable reference
        if "ANALYSIS_COMPLETE" in raw_response:
            return [SlotSet("analysis_complete", True)]
        return []

class ActionSelectLogicFlow(Action):
    def name(self) -> str:
        return "action_select_logic_flow"

    async def run(self, dispatcher, tracker, domain):
        # Shuffling the Logic Flow (Session A, B, or C)
        flow_choice = random.choices(
            ["standard", "doctor", "mystery"], 
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        # Entry Style Selection
        entry_rand = random.random()
        if entry_rand < 0.4:
            entry_style = "memory_hook"
        elif entry_rand < 0.7:
            entry_style = "transit_alert"
        elif entry_rand < 0.9:
            entry_style = "direct_insight"
        else:
            entry_style = "silent_guru"

        return [
            SlotSet("logic_flow", flow_choice),
            SlotSet("entry_style", entry_style) 
        ]

class ActionProcessAstroInsight(Action):
    def name(self) -> str:
        return "action_process_astro_insight"

    async def run(self, dispatcher, tracker, domain):
        user_id = tracker.sender_id
        profile = get_user_profile(user_id, user_id)
        
        # Increment interaction for pattern tracking
        profile.interaction_count += 1
        
        # 1. Bold Prediction Logic (Requirement 4)
        # 2. Every 4 topic changes, do a Barnum Flip (Requirement 7)
        do_barnum = (profile.interaction_count % 4 == 0)
        
        system_prompt = f"""
        You are Omkar, a high-status Vedic Astrologer. 
        Tone: Authoritative, Empathetic, Hinglish.
        Current Context: User is worried about {tracker.get_slot('current_pain_point')}.
        
        Rules:
        - If 'do_barnum' is True, start with a Vulnerable Truth: "Aap dil ke saaf hain, par log fayda uthate hain."
        - Use Bold Predictions: "You've felt this pressure for months, right?" (Past-tense certainty).
        - Don't jump to planets immediately; offer sympathy first.
        - Use the Zeigarnik Effect: Hint at something in the '10th house' but don't reveal it yet.
        """
        
        response = model.generate_content(system_prompt)
        dispatcher.utter_message(text=response.text)
        
        update_user_profile(user_id, profile)
        
        # Changed 'pain_point_count' to 'interaction_counter' to match domain.yml
        return [SlotSet("interaction_counter", profile.interaction_count)]

class ActionCheckRemedy(Action):
    def name(self) -> str:
        return "action_check_remedy"

    async def run(self, dispatcher, tracker, domain):
        user_id = tracker.sender_id
        profile = get_user_profile(user_id, user_id)
        
        if profile.active_remedy:
            # Memory Layer (Requirement: Progress Check)
            remedy_name = profile.active_remedy.get('name')
            dispatcher.utter_message(text=f"Namaste! Kya aapne wo {remedy_name} wala upay kiya? Kya asar mehsoos hua?")
        else:
            dispatcher.utter_message(text="Namaste! Grahon ki chaal jhoot nahi bolti. Batiye aaj kya duvidha hai?")
        
        return []