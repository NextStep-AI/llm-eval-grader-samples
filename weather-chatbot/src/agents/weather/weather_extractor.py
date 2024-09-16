import os
import inspect
from openai import AzureOpenAI

from src.context import Context
from src.clients.weather import WeatherType

UNKNOWN_CATEGORY = "UNKNOWN"


class WeatherExtractor:
    """
    Class for extracting what information about the weather
    a user wants to know so that an API call can be made.
    """

    def extract(self, context: Context):

        message_history = context.get_messages()

        # Grabbing the most recent two messages so that we can update
        # the category as needed, without managing change detection
        recent_history = message_history[-2:]

        if len(message_history) == 0:
            return

        flattened_history = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])

        system_prompt = inspect.cleandoc(f"""
            Your task is to try and determine what type of question or questions a user is asking about the weather
            based on a following conversation transcript and classify it into one of the following
            categories: {[enum.name for enum in WeatherType]}. If it's unclear what category to choose
            or the user hasn't asked any questions about the weather simply return {UNKNOWN_CATEGORY}.
            Conversation transcript:
            ```
            {flattened_history}
            ```
            """)

        messages = [{"role": "system", "content": system_prompt}]

        response = AzureOpenAI().chat.completions.create(
            temperature=0,
            model=os.environ["OPENAI_DEPLOYMENT_NAME"],
            messages=messages)

        weather_category_response = response.choices[0].message.content

        if UNKNOWN_CATEGORY in weather_category_response.upper():
            return

        for enum in WeatherType:
            if enum.name in weather_category_response.upper():
                context.weather_category = enum
