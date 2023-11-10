"""
This is a discord bot for generating texts, images, files and audio using OpenAI's GPT-4-Turbo, Dall-E 3 and other models

Author: Stefan Rial
YouTube: https://youtube.com/@StefanRial
GitHub: https://https://github.com/StefanRial/AlexBot
E-Mail: mail.stefanrial@gmail.com
"""
from datetime import datetime
import io
import json
import urllib
import urllib.request
import discord
from openai import OpenAI
from configparser import ConfigParser
from discord import app_commands

config_file = "config.ini"
config = ConfigParser(interpolation=None)
config.read(config_file)

SERVER_ID = config["discord"]["server_id"]
DISCORD_API_KEY = config["discord"][str("api_key")]
OPENAI_ORG = config["openai"][str("organization")]
OPENAI_API_KEY = config["openai"][str("api_key")]

GUILD = discord.Object(id=SERVER_ID)

SYSTEM_MESSAGE = config["bot"]["system_message"]
HISTORY_LENGTH = config["bot"]["history_length"]

FILE_PATH = config["settings"][str("file_path")]
FILE_NAME_FORMAT = config["settings"][str("file_name_format")]

tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image_with_dalle",
            "description": "generates an image using Dall-E and returns it as a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "prompt for the image to be generated"
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_python_script",
            "description": "creates and returns a python script as a main.py file",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "prompt for the script to be created"
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_voice_message",
            "description": "creates and returns a voice message with the prompt as text",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "what the voice message should say"
                    }
                },
                "required": ["prompt"],
            },
        },
    }
]

ai = OpenAI(api_key=OPENAI_API_KEY)


def download_image(url: str):
    file_name = f"{datetime.now().strftime(FILE_NAME_FORMAT)}.jpg"
    full_path = f"{FILE_PATH}{file_name}"
    urllib.request.urlretrieve(url, full_path)
    return file_name


def create_voice_message(prompt):
    response = ai.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=prompt
    )

    file_name = f"{datetime.now().strftime(FILE_NAME_FORMAT)}.mp3"
    full_path = f"{FILE_PATH}{file_name}"

    response.stream_to_file(full_path)

    return full_path


def create_python_script(prompt):
    response = ai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Only respond with the generated code, formatted for a .py file."},
            {"role": "user", "content": prompt}
        ]
    )

    script = response.choices[0].message.content
    file_name = f"{datetime.now().strftime(FILE_NAME_FORMAT)}.py"
    full_path = f"{FILE_PATH}{file_name}"
    with open(full_path, "w") as file:
        file.write(script)

    return full_path


def trim_conversation_history(history, max_length=int(HISTORY_LENGTH)):
    if len(history) > max_length:
        history = history[-max_length:]
    return history


def generate_image_with_dalle(prompt):
    response = ai.images.generate(
        prompt = prompt,
        model = "dall-e-3",
        quality="hd",
        response_format="url"
    )
    image_data = response.data[0]
    image_url = image_data.url

    print(image_url)
    return "files/" + download_image(image_url)


class Client(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.conversation_history = []

    async def setup_hook(self):
        self.tree.copy_global_to(guild=GUILD)
        await self.tree.sync(guild=GUILD)

    async def on_message(self, message):
        author = message.author
        embed_files = []
        if message.author == self.user:
            return

        input_content = message.content
        print(f"{message.author}: {input_content}")

        self.conversation_history.append({"role": "system", "content": f"The user is {author.display_name}. {SYSTEM_MESSAGE}"})

        if message.attachments:
            attachment = message.attachments[0]
            if attachment.filename.endswith('.py'):
                file_content = io.BytesIO()
                await attachment.save(file_content)
                file_content.seek(0)
                text_data = file_content.read().decode('utf-8')
                self.conversation_history.append({"role": "user", "content": input_content + ", existing file content: " + text_data})
        else:
            self.conversation_history.append({"role": "user", "content": input_content})

        self.conversation_history = trim_conversation_history(self.conversation_history)

        try:
            response = ai.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=self.conversation_history,
                tools=tools,
                tool_choice="auto"
            )

            assistant_response = response.choices[0].message
            tool_calls = assistant_response.tool_calls

            if tool_calls:
                available_functions = {
                    "generate_image_with_dalle": generate_image_with_dalle,
                    "create_python_script": create_python_script,
                    "create_voice_message": create_voice_message,
                }
                self.conversation_history.append(assistant_response)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(
                        prompt=function_args.get("prompt")
                    )
                    self.conversation_history.append(
                        {"tool_call_id": tool_call.id,
                         "role": "tool",
                         "name": function_name,
                         "content": "The file has been created and is attached to your next message"
                         }
                    )
                    embed_files.append(discord.File(function_response))

                response = ai.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=self.conversation_history
                )
                assistant_response = response.choices[0].message
            else:
                self.conversation_history.append({"role": "assistant", "content": assistant_response.content})
            assistant_response = assistant_response.content

        except AttributeError:
            assistant_response = "It looks like you might have to update your openai package. You can do that with ```pip install --upgrade openai```"
        except ImportError:
            assistant_response = "You might not have all required packages installed. Make sure you install the openai and discord package"

        if assistant_response is not None:
            parts = [assistant_response[i:i + 2000] for i in range(0, len(assistant_response), 2000)]
            for index, part in enumerate(parts):
                try:
                    print(f"E.D.I.T.H.: {part}")
                    if len(embed_files) > 0:
                        await message.channel.send(content=part, files=embed_files)
                    else:
                        await message.channel.send(part)
                except discord.errors.Forbidden:
                    print("E.D.I.T.H.: I am not able to send a message. Do I have the correct permissions on your server?")


intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = Client(intents=intents)

client.run(DISCORD_API_KEY)
