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
    You are an NLU engine for an astrology chatbot.
    Users speak in Hindi, Hinglish, or English. Messages are short and informal.

    Your job: Given a user message, return a JSON object with:
    - "intent": one of [greeting, user_acknowledgement, ask_question, provide_birth_info, 
                        express_urgency, express_doubt, express_gratitude, 
                        new_topic, emotional_response, out_of_scope]
    - "confidence": float between 0 and 1
    - "entities": list of {entity, value} objects if any (e.g. name, date, problem_type)
    - "sentiment": one of [positive, neutral, negative, distressed]

    Rules:
    - Any form of greeting "hello" "namaste" "hi = greeting
    - Short replies like "ok", "hmm", "P", "??", "👍", "theek" = user_acknowledgement
    - Questions about love, job, money, health = ask_question
    - Sharing birth date/time/place = provide_birth_info
    - "jaldi bata", "urgent hai" = express_urgency
    - Doubt or disbelief = express_doubt
    - "shukriya", "thank you", "bahut acha" = express_gratitude
    - Switching to a completely new problem = new_topic
    - Crying, fear, hopelessness = emotional_response
    - Anything unrelated to astrology = out_of_scope

    Respond ONLY with valid JSON. No explanation.

    Example output:
    {"intent": "ask_question", "confidence": 0.95, "entities": [{"entity": "problem_type", "value": "marriage"}], "sentiment": "distressed"}
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