import google.generativeai as genai
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.engine.graph import GraphComponent as Component
import json

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class GeminiNLUComponent(Component):

    name = "GeminiNLUComponent"

    SYSTEM_PROMPT = """
    You are an NLU engine for a Jyotish AI astrology chatbot.
    Users speak in Hindi, Hinglish, or English. Messages are short and informal.

    Your job: Given a user message, return a JSON object with:
    - "intent": one of [
        greeting, user_acknowledgement, ask_question, provide_birth_info,
        express_urgency, express_doubt, express_gratitude,
        new_topic, emotional_response, out_of_scope,
        share_love_concern, share_career_concern, share_family_concern,
        share_marriage_concern, ask_open_general, ask_specific_question,
        share_new_update, ask_remedy, confirm_remedy_done, topic_change,
        goodbye, returning_user, affirm, deny, inform_pain_point,
        ask_remedy_check, express_distressx
      ]
    - "confidence": float between 0 and 1
    - "entities": list of {entity, value} objects if any (e.g. name, date, problem_type)
    - "sentiment": one of [positive, neutral, negative, distressed]

    Rules:
    - "hello", "namaste", "hi", "Radhe Radhe" = greeting
    - Short replies like "ok", "hmm", "P", "??", "👍", "theek", "haan", "accha" = user_acknowledgement
    - "haan", "yes", "bilkul", "sahi hai" = affirm
    - "nahi", "no", "nope", "galat" = deny
    - Sharing birth date/time/place = provide_birth_info
    - "jaldi bata", "urgent hai", "please jaldi" = express_urgency
    - Doubt or disbelief about prediction = express_doubt
    - "shukriya", "thank you", "bahut acha", "thanks" = express_gratitude
    - Love/relationship problems (boyfriend, girlfriend, husband, wife, ex) = share_love_concern
    - Job, business, money, career problems = share_career_concern
    - Family conflict, parents, siblings = share_family_concern
    - Marriage, shaadi, rishta, divorce = share_marriage_concern
    - Open ended "kya hoga mera?", "future batao" with no specific topic = ask_open_general
    - Specific question about timing, outcome, prediction = ask_specific_question
    - User sharing a new development or update about their situation = share_new_update
    - Asking for remedy, totka, upay, solution = ask_remedy
    - Confirming they did the remedy = confirm_remedy_done
    - Switching to a completely different topic = topic_change
    - "bye", "ok theek hai", "band karo", farewell = goodbye
    - User mentions they talked before, came back, "pehle bhi aaya tha" = returning_user
    - Explicitly describing their pain point or problem in detail = inform_pain_point
    - Asking if a remedy will work, checking remedy status = ask_remedy_check
    - Deep emotional distress, crying, fear, despair = express_distress
    - Strong negative emotion but not necessarily distress = emotional_response
    - Anything unrelated to astrology/relationships/life problems = out_of_scope

    Respond ONLY with valid JSON. No explanation.

    Example output:
    {"intent": "share_love_concern", "confidence": 0.95, "entities": [{"entity": "person_name", "value": "Rahul"}, {"entity": "relationship_type", "value": "ex"}], "sentiment": "distressed"}
    """

    def __init__(self, config, model_storage, resource, execution_context):
        super().__init__()
        genai.configure(api_key=config.get("gemini_api_key"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    @classmethod
    def create(cls, config, model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext):
        return cls(config, model_storage, resource, execution_context)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    def process(self, messages: list[Message]) -> list[Message]:
        for message in messages:
            user_text = message.get("text")
            if not user_text:
                continue

            try:
                response = self.model.generate_content(
                    f"{self.SYSTEM_PROMPT}\n\nUser message: {user_text}"
                )
                result = json.loads(response.text.strip())

                # Set intent
                message.set("intent", {
                    "name": result.get("intent", "out_of_scope"),
                    "confidence": result.get("confidence", 0.9)
                }, add_to_output=True)

                # Set entities
                entities = result.get("entities", [])
                message.set("entities", entities, add_to_output=True)

                # Set sentiment as a custom key (accessible in actions via tracker)
                message.set("sentiment", result.get("sentiment", "neutral"), add_to_output=True)

            except Exception as e:
                # Fallback if Gemini fails
                message.set("intent", {"name": "out_of_scope", "confidence": 0.0}, add_to_output=True)
                message.set("entities", [], add_to_output=True)

        return messages

    def train(self, training_data: TrainingData) -> None:
        pass  # No training needed — Gemini handles everything