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
import os
import pickle
import AIDAkeys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from telegram import __version__ as TG_VER

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



template = """You're a friendly AI of Università degli Studi di Milano-Bicocca made to give an answer to students' (both High School Students and University Students) question. You must not talk about other universities.
Your answers must be based on the documents provided; all documents provided are related to the Università degli Sudi di Milano-Bicocca.
If a question is asked within the application about a degree program or course of study your question must follow the contents of their documents.\n
If you want more informations to understand better which documents you need to use, you can ask for clarifications to the user. Note that in SUA documents you can find general informations about a degree course,
in Syllabus documents you can find informations about a single exam. Note that in italian the word "corso" is used also as a synonym of the word "insegnamento", so you need to disambiguate the term
in order to understand if the user wants informations about a degree course or an exam. \n
You must not answer questions not related to the university environment. \n
Use the following pieces of context to answer the users question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
\n----------------\n{context}"""


# aggiungere che deve rispondere solo a domande in ambito universitario

prompt=ChatPromptTemplate(
    input_variables=['context', 'question'], 
    output_parser=None, partial_variables={}, 
    messages=[
        SystemMessagePromptTemplate(
            prompt = PromptTemplate(
                input_variables=['context'], 
                output_parser=None, 
                partial_variables={}, 
                template=template, 
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

    ref = db.reference('/chats/'+str(user.id))
    snapshot = ref.get()

    if snapshot is not None:
        await update.message.reply_html(
            f"""Ciao {user.first_name}!
Come posso aiutarti?
        """,          
        )

    else:

        chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
   
        ref.set(pickle.dumps(chat_history).hex())

        await update.message.reply_html(
            rf"""Ciao {user.mention_html()}!
Sono un intelligenza artificiale ...""",
            
        )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    user = update.effective_user
    ref = db.reference('/chats/'+str(user.id))

    chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    


    ref.set(pickle.dumps(chat_history).hex())

    chat_id = update.effective_chat.id
    message_id = update.message.message_id
    bot = context.bot
    bot.delete_message(chat_id, message_id)

    await update.message.reply_text("😵‍💫")
    await update.message.reply_text("Possiamo parlare di un altro argomento, mi sono dimenticato di tutto ciò che mi hai detto.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    user = update.effective_user
    ref = db.reference('/chats/'+str(user.id))

    snapshot = ref.get()

    memory = pickle.loads(bytes.fromhex(snapshot))

  

    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
                                           verbose = True,
                                           retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k":10}),
                                           memory=memory,
                                           chain_type = "stuff",
                                           combine_docs_chain_kwargs={'prompt': prompt})
    print(update.message.text)
    # context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    result = qa({"question": str(update.message.text) })



    ref.set(pickle.dumps(qa.memory).hex())

    await update.message.reply_text(result["answer"])

    del qa


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