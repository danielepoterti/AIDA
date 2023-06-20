#!/usr/bin/env python
# pylint: disable=unused-argument, wrong-import-position
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import random
import os
import pickle
import AIDAkeys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain import PromptTemplate
from langchain.memory import  ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from telegram import __version__ as TG_VER
from telegram.constants import ChatAction

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
# from telegram import (ChatAction)
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Percorso del file JSON delle credenziali
cred = credentials.Certificate(AIDAkeys.firebaseCertificate)

# Inizializza l'app Firebase
firebase_admin.initialize_app(cred,{
    'databaseURL': AIDAkeys.databaseURL
})

os.environ['OPENAI_API_KEY'] = AIDAkeys.openAIkeyAndrea
embeddings = OpenAIEmbeddings()
persist_directory = 'ChromaDB_Bicocca_ALTERNATIVE_DEF'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


prompt=ChatPromptTemplate(
    input_variables=['context', 'question'], 
    output_parser=None,
    partial_variables={}, 
    messages=[
        SystemMessagePromptTemplate(
            prompt = PromptTemplate(
                input_variables=['context'], 
                output_parser=None, 
                partial_variables={}, 
                template=AIDAkeys.template, 
                template_format='f-string', 
                validate_template=True), 
                additional_kwargs={}), 
                HumanMessagePromptTemplate(
                    prompt = PromptTemplate(
                        input_variables=['question'], 
                        output_parser=None, 
                        partial_variables={}, 
                        template='{question}', 
                        template_format='f-string', 
                        validate_template=True), 
                        additional_kwargs={})])

def processThought(thought):
  return thought



# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user

    ref_mem = db.reference('/chats/'+str(user.id)+'/memory/')
    snapshot_mem = ref_mem.get()

    

    if snapshot_mem is not None :
        await update.message.reply_html(
            f"""Ciao {user.first_name}!
Come posso aiutarti?
        """,          
        )

    else:

        chat_mem = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k = 5)
        
   
        ref_mem.set(pickle.dumps(chat_mem).hex())
        

        await update.message.reply_html(
            rf"""Ciao {user.mention_html()}!
Sono un intelligenza artificiale ...""",
            
        )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    user = update.effective_user
    ref_mem = db.reference('/chats/'+str(user.id)+'/memory/')
    

    chat_mem = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k = 5)
    
    
    ref_mem.set(pickle.dumps(chat_mem).hex())
    

    chat_id = update.effective_chat.id
    message_id = update.message.message_id
    bot = context.bot
    bot.delete_message(chat_id, message_id)

    await update.message.reply_text("😵‍💫")
    await update.message.reply_text("Possiamo parlare di un altro argomento, mi sono dimenticato di tutto ciò che mi hai detto.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    waitingEmoji = ["🤔", "💭", "🔎", "💬"]

    await update.message.reply_text(random.choice(waitingEmoji))

    ref_mem = db.reference('/chats/'+str(user.id)+'/memory/')
    

    snapshot_mem = ref_mem.get()
    

    memory = pickle.loads(bytes.fromhex(snapshot_mem))
    

    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
                                           verbose = True,
                                           retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k":8}),
                                           memory=memory,
                                           chain_type = "stuff",
                                           condense_question_llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo'),
                                           #condense_question_prompt = CONDENSE_QUESTION_PROMPT,
                                           combine_docs_chain_kwargs={'prompt': prompt}
                                           )
    
    llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')

    tools = [
    Tool(
        name = "Bicocca QA System",
        func=qa.run,
        description="""useful for when you need to answer questions about courses at the University of Milano-Bicocca. It allows you to find information into document of degree courses or exams belonging to a degree course.
        This tool is useful when the user asks for informations about University of Milano-Bicocca aspects. Input should be a question."""
    ),
    Tool(
    name = "Thought Processing",
    description = """This is useful for when you have a thought that you want to use in a task,
    but you want to make sure it's formatted correctly.
    Input is your thought and self-critique and output is the processed thought.""",
    func =  processThought,
  )

    ]

    agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                         verbose=True,
                         memory =  memory,
                         agent_kwargs = {                         
                                "input_variables": ["input", "agent_scratchpad", "chat_history"],
                          },
                         handle_parsing_errors=True,
                         )

    
    print(update.message.text)
    
    

    agent.agent.llm_chain.prompt.template = AIDAkeys.templateAgent

    response = agent.run(input=str(update.message.text))

    ref_mem.set(pickle.dumps(agent.memory).hex())

    message_id = update.effective_message.message_id
    
    await context.bot.delete_message(chat_id=chat_id, message_id=message_id+1)

    await update.message.reply_text(response)

    del qa
    del agent


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(AIDAkeys.telegramBOTtoken).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()