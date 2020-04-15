# t.me/appart_ai_bot

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, ConversationHandler, MessageHandler, Filters
import logging
from emoji import emojize
import requests
import telegram
import pickle
import pandas as pd
import os
import numpy as np
import csv
from datetime import datetime
import re

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Stages
FIRST, TYPING_REPLY, TYPING_REPLY_AREA, TYPING_REPLY_ROOM, TYPING_REPLY_FURNISHED, TYPING_REPLY_TYPE, TYPING_REPLY_COMPLETE, TYPING_REPLY_AREA_COMPLETE, TYPING_REPLY_ROOM_COMPLETE, TYPING_REPLY_FURNISHED_COMPLETE, TYPING_REPLY_TYPE_COMPLETE, TYPING_REPLY_FLOOR_COMPLETE, TYPING_REPLY_BALCON_COMPLETE, TYPING_REPLY_GARDIEN_COMPLETE, TYPING_REPLY_PARKING_COMPLETE, TYPING_REPLY_MACHINEALAVER_COMPLETE, TYPING_REPLY_AMERICAINE_COMPLETE, TYPING_REPLY_QUALITE_COMPLETE, PREDICT_AREA_CATEGORY, IMPORT= range(20)
# Callback data
START, PREDICT_AREA, VERIFY, BOP, CREDITS, COMMENT_CHOISIR, ANALYSE_RAPIDE, ANALYSE_COMPLETE, COMPARE_PARIS, COMPARE_COURONNE, PREDICT_AREA_ONE, ANALYSE_IMPORT, INFORMATIONS = range(13)

##Machine learning algorithm ####################################################################################
os.chdir('/home/ec2-user/')
fast_model = pickle.load(open("model_fast_city.sav", 'rb'))
complete_model = pickle.load(open("model_complete_city.sav", 'rb'))
predict_area_model = pickle.load(open("predict_area.sav", 'rb'))
dictionary = {'type': 1.0, 'area': 0.0, 'room': 0.0, 'furnished': 0.0, 'ville_75002': 0.0, 'ville_75003': 0.0, 'ville_75004': 0.0, 'ville_75005': 0.0, 'ville_75006': 0.0, 'ville_75007': 0.0, 'ville_75008': 0.0, 'ville_75009': 0.0, 'ville_75010': 0.0, 'ville_75011': 0.0, 'ville_75012': 0.0, 'ville_75013': 0.0, 'ville_75014': 0.0, 'ville_75015': 0.0, 'ville_75016': 0.0, 'ville_75017': 0.0, 'ville_75018': 0.0, 'ville_75019': 0.0, 'ville_75020': 0.0, 'ville_92100': 0.0, 'ville_92110': 0.0, 'ville_92120': 0.0, 'ville_92130': 0.0, 'ville_92170': 0.0, 'ville_92200': 0.0, 'ville_92240': 0.0, 'ville_92300': 0.0, 'ville_93100': 0.0, 'ville_93170': 0.0, 'ville_93200': 0.0, 'ville_93210': 0.0, 'ville_93260': 0.0, 'ville_93300': 0.0, 'ville_93310': 0.0, 'ville_93400': 0.0, 'ville_93500': 0.0, 'ville_94160': 0.0, 'ville_94200': 0.0, 'ville_94220': 0.0, 'ville_94250': 0.0, 'ville_94270': 0.0, 'ville_94300': 0.0}
dictionary_complete = {'type': 1.0, 'area': 0.0, 'room': 0.0, 'furnished': 0.0, 'floor': 0.0, 'parking': 0.0, 'balcon': 0.0, 'qualite': 0.0, 'ameri': 0.0, 'gardien': 0.0, 'machine_a_laver': 0.0, 'ville_75002': 0.0, 'ville_75003': 0.0, 'ville_75004': 0.0, 'ville_75005': 0.0, 'ville_75006': 0.0, 'ville_75007': 0.0, 'ville_75008': 0.0, 'ville_75009': 0.0, 'ville_75010': 0.0, 'ville_75011': 0.0, 'ville_75012': 0.0, 'ville_75013': 0.0, 'ville_75014': 0.0, 'ville_75015': 0.0, 'ville_75016': 0.0, 'ville_75017': 0.0, 'ville_75018': 0.0, 'ville_75019': 0.0, 'ville_75020': 0.0, 'ville_92100': 0.0, 'ville_92110': 0.0, 'ville_92120': 0.0, 'ville_92130': 0.0, 'ville_92170': 0.0, 'ville_92200': 0.0, 'ville_92240': 0.0, 'ville_92300': 0.0, 'ville_93100': 0.0, 'ville_93170': 0.0, 'ville_93200': 0.0, 'ville_93210': 0.0, 'ville_93260': 0.0, 'ville_93300': 0.0, 'ville_93310': 0.0, 'ville_93400': 0.0, 'ville_93500': 0.0, 'ville_94160': 0.0, 'ville_94200': 0.0, 'ville_94220': 0.0, 'ville_94250': 0.0, 'ville_94270': 0.0, 'ville_94300': 0.0}
dictionary_predict_area = {'rent': 0, 'furnished': 0, 'ville_75002': 0, 'ville_75003': 0, 'ville_75004': 0, 'ville_75005': 0, 'ville_75006': 0, 'ville_75007': 0, 'ville_75008': 0, 'ville_75009': 0, 'ville_75010': 0, 'ville_75011': 0, 'ville_75012': 0, 'ville_75013': 0, 'ville_75014': 0, 'ville_75015': 0, 'ville_75016': 0, 'ville_75017': 0, 'ville_75018': 0, 'ville_75019': 0, 'ville_75020': 0, 'ville_92100': 0, 'ville_92110': 0, 'ville_92120': 0, 'ville_92130': 0, 'ville_92170': 0, 'ville_92200': 0, 'ville_92240': 0, 'ville_92300': 0, 'ville_93100': 0, 'ville_93170': 0, 'ville_93200': 0, 'ville_93210': 0, 'ville_93260': 0, 'ville_93300': 0, 'ville_93310': 0, 'ville_93400': 0, 'ville_93500': 0, 'ville_94160': 0, 'ville_94200': 0, 'ville_94220': 0, 'ville_94250': 0, 'ville_94270': 0, 'ville_94300': 0}

# KEYBOARDS ##################################################################################

keyboard_start = [
    [InlineKeyboardButton(emojize(":compass:  Analyser la carte des superficies ", use_aliases=True), callback_data=str(PREDICT_AREA))],
    [InlineKeyboardButton(emojize(":bar_chart:  Estimer un loyer", use_aliases=True),callback_data=str(VERIFY))],
    [InlineKeyboardButton(emojize(":robot:  A propos", use_aliases=True), callback_data=str(CREDITS))]
    ]

keyboard_bop = [
    [InlineKeyboardButton(emojize(":dog:  Plus de fun", use_aliases=True), callback_data=str(BOP)),
     InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]
]

keyboard_credits = [
    [InlineKeyboardButton(emojize(":man:  Linkedin", use_aliases=True), url="https://www.linkedin.com/in/antoinebaena/")],
    [InlineKeyboardButton(emojize(":dog:  Avoir du fun", use_aliases=True), callback_data=str(BOP)),
     InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]
    ]

keyboard_verify = [
    [InlineKeyboardButton(emojize(":mag:  Importer une annonce depuis LouerAgile", use_aliases=True), callback_data=str(IMPORT))],
    [InlineKeyboardButton(emojize(":zap:  Analyse rapide", use_aliases=True), callback_data=str(ANALYSE_RAPIDE)),
    InlineKeyboardButton(emojize(":star:  Analyse complète", use_aliases=True), callback_data=str(ANALYSE_COMPLETE))],
    [InlineKeyboardButton(emojize(":face_with_monocle:  Comment choisir", use_aliases=True), callback_data=str(COMMENT_CHOISIR)),
     InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]
    ]

keyboard_verify2 = [
    [InlineKeyboardButton(emojize(":mag:  Importer une annonce depuis LouerAgile", use_aliases=True), callback_data=str(IMPORT))],
    [InlineKeyboardButton(emojize(":zap:  Analyse rapide", use_aliases=True), callback_data=str(ANALYSE_RAPIDE)),
    InlineKeyboardButton(emojize(":star:  Analyse complète", use_aliases=True), callback_data=str(ANALYSE_COMPLETE))],
    [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]
    ]

keyboard_lost = [[InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

# COMMANDES ##################################################################################"
users = dict()

def start(update, context):
    user = update.message.from_user
    logger.info("User %s started the conversation.", user.first_name)
    users[update.effective_chat.id] = dict()
    users[update.effective_chat.id]['id'] = update.effective_chat.id
    users[update.effective_chat.id]['first_name'] = user['first_name']
    users[update.effective_chat.id]['language_code'] = user['language_code']
    users[update.effective_chat.id]['date'] = datetime.now()

    reply_markup = InlineKeyboardMarkup(keyboard_start)

    update.message.reply_text(
        emojize("Salut ! Que puis-je faire pour t'aider {}?".format(user['first_name']), use_aliases=True),
        reply_markup=reply_markup
    )
    return FIRST

def start_over(update, context):
    query = update.callback_query
    query.answer()

    reply_markup = InlineKeyboardMarkup(keyboard_start)
    query.edit_message_text(
        text=emojize("Salut ! Que puis-je faire pour t'aider ?", use_aliases=True),
        reply_markup=reply_markup
    )
    return FIRST

def get_url():
    contents = requests.get('https://random.dog/woof.json').json()
    url = contents['url']
    return url

def bop(update, context):
    query = update.callback_query
    query.answer()
    query.edit_message_text(text="Fun envoyé :)")
    url = get_url()
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=url)

    reply_markup = InlineKeyboardMarkup(keyboard_bop)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="On fait quoi maintenant ?",
        reply_markup=reply_markup
    )
    return FIRST

def credits(update, context):
    query = update.callback_query
    query.answer()

    reply_markup = InlineKeyboardMarkup(keyboard_credits)
    query.edit_message_text(
        text=emojize("<b> A propos de Appart-ai</b> \n \n<b>Description</b> \nTu cherches à louer un appartement ou une maison à Paris ou aux alentours ? Je suis là pour t'aider à mieux comprendre le marché de la location immobilière à Paris ! :house: \n \n<b>Auteur</b> \nAntoine Baena, économiste. \nENSAE Institut Polytechnique de Paris.  \n \n<b>Outils</b> \nAppart-ai a été développé en Python en s'appuyant sur l'API de Telegram. La partie machine learning a été développée avec pandas, scikit-learn et nltk. Les modèles de prédiction retenus sont des regressions Gradient Boosting (XGBoost). \n \n<b>Données</b> \n- Annonces de locations immobilières exhaustives scrapées en temps réel depuis Juillet 2019 sur 8 sites, dont SeLoger, Leboncoin et Pap. \n- Données géographiques de OpenStreetMap. \n- data.gouv.fr, Positions géographiques des stations du réseau RATP. \n \nChatbot créé avec amour à Paris en avril 2020.",
                     use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup
    )
    return FIRST

# PREDICT AREA #########################################################################################

keyboard_start_predict_area = [[InlineKeyboardButton(emojize(":rocket:  Lancer", use_aliases=True), callback_data=str(PREDICT_AREA_ONE))],
                               [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))
                               ]]

def start_predict_area(update, context):
    query = update.callback_query
    query.answer()
    reply_markup = InlineKeyboardMarkup(keyboard_start_predict_area)
    query.edit_message_text(
        text=emojize("<b>:compass:  Analyser la carte des superficies</b> \n \nLa première étape pour trouver un appartement, c'est de décider où le chercher. Celà dépend de tes préférences personnelles et de ton budget. \n \nPour t'aider à faire ce choix, je peux t'indiquer quelle est la superficie que tu pourrais avoir dans chaque arrondissement de Paris et communes limitrophes, en fonction de ton budget et de si tu souhaites un appart meublé ou non.", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup
    )
    return FIRST

def predict_area_budget(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize("Tout d'abord, peux-tu m'indiquer ton budget (charges comprises) ?", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML
    )
    return PREDICT_AREA_CATEGORY

def predict_area_budget_2(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize(":robot: Bip bop ! Je suis capable de t'aider que pour des budgets compris entre 400€ et 4000€\n \n :arrow_right: Peux-tu m'indiquer ton budget (charges comprises) ?", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML
    )
    return PREDICT_AREA_CATEGORY

reply_keyboard_furnished = [['Meublé', 'Non meublé']]
keyboard_furnished = ReplyKeyboardMarkup(reply_keyboard_furnished, one_time_keyboard=True)

def predict_area_furnished(update, context):
    users[update.effective_chat.id]['rent'] = update.message.text
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("Ton budget est <b>{}€</b> par mois, charges comprises. \n \n :arrow_right: Est-ce que tu souhaites un appartement meublé ou non meublé ?".format(users[update.effective_chat.id]['rent']), use_aliases=True),
        parse_mode = telegram.ParseMode.HTML,
        reply_markup=keyboard_furnished
    )
    return PREDICT_AREA_CATEGORY

reply_keyboard_paris_couronne = [['Uniquement Paris', 'Paris et les communes limitrophes']]
keyboard_paris_couronne = ReplyKeyboardMarkup(reply_keyboard_paris_couronne, one_time_keyboard=True)

def predict_area_paris_couronne(update, context):
    users[update.effective_chat.id]['furnished'] = update.message.text
    if users[update.effective_chat.id]['furnished'] == "Meublé":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("Tu souhaites un appartement <b>meublé</b>. \n \n :arrow_right: Pour pouvoir t'aider au mieux, j'ai besoin de savoir si tu souhaites connaitre les superficies que tu pourrais avoir uniquement à Paris, ou Paris et les communes limitrophes ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_paris_couronne
        )
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("Tu souhaites un appartement <b> non meublé</b>. \n \n :arrow_right: Pour pouvoir t'aider au mieux, j'ai besoin de savoir si tu souhaites connaitre les superficies que tu pourrais avoir uniquement à Paris, ou Paris et les communes limitrophes ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_paris_couronne
        )
    return PREDICT_AREA_CATEGORY

keyboard_compare_paris = [
                [InlineKeyboardButton(emojize(":compass:  Comparer avec les villes limitrophes", use_aliases=True), callback_data=str(COMPARE_COURONNE))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(PREDICT_AREA_ONE))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def predict_area_paris(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_paris)

    personal_dictionary = {**dictionary_predict_area}
    personal_dictionary['rent'] = float(users[update.effective_chat.id]['rent'])
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")

    predictions = dict()
    for i in list_of_city[0:20]:
        i_id = dico_key_values_city[i]
        if i == "1er - Louvre":
            predictions[i] = int(predict_area_model.predict(pd.DataFrame([personal_dictionary])))
        else:
            personal_dictionary[i_id] = float(1)
            predictions[i] = int(predict_area_model.predict(pd.DataFrame([personal_dictionary])))
            personal_dictionary[i_id] = float(0)

    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i,y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("A Paris, le loyer médian est 1189€, et la superficie médiane est 34m2 (997€ dans les communes limitrophes, et 37m2). Seuls 5% des appartements ont un loyer inférieur à 580€, et seuls 1% moins de 450€. \n \nSi tu indiques un budget trop faible par rapport au marché locatif parisien, tu risques d'obtenir des valeurs abérantes car l'algorithme ne va pas savoir comment gérer ta requête car elle est impossible. ", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://www.hotel-design-secret-de-paris.com/blog/wp-content/uploads/2013/11/arrondissement-paris.gif")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("<b>Caractéristiques de ta recherche </b> :  \n \n<b>Budget </b>: {}  \n<b>Meublé </b>: {} \n \n Les superficies sont classées de la plus élevée à la plus basse. \n \n 	:small_red_triangle:{} : {}m2 \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2  ".format(users[update.effective_chat.id]['rent'], users[update.effective_chat.id]['furnished'], *ordre_list), use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)

    try:
        open('analyse_area_log.csv', 'r')
        with open('analyse_area_log.csv', 'a') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].values())
    except FileNotFoundError:
        with open('analyse_area_log.csv', 'a+') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].keys())
            w.writerow(users[update.effective_chat.id].values())
    return PREDICT_AREA_CATEGORY

keyboard_compare_paris = [
                [InlineKeyboardButton(emojize(":compass:  Comparer avec uniquement Paris", use_aliases=True), callback_data=str(COMPARE_PARIS))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(PREDICT_AREA_ONE))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def predict_area_couronne(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_paris)

    personal_dictionary = {**dictionary_predict_area}
    personal_dictionary['rent'] = float(users[update.effective_chat.id]['rent'])
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")

    predictions = dict()
    for i in list_of_city:
        i_id = dico_key_values_city[i]
        if i == "1er - Louvre":
            predictions[i] = int(predict_area_model.predict(pd.DataFrame([personal_dictionary])))
        else:
            personal_dictionary[i_id] = float(1)
            predictions[i] = int(predict_area_model.predict(pd.DataFrame([personal_dictionary])))
            personal_dictionary[i_id] = float(0)

    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i,y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":compass: <b>Comparaison avec les arrondissements de Paris uniquement</b> \n \n :bulb: Quelques informations sont à avoir pour bien comprendre l'outil qui t'est proposé, son intêret et ses limites. \n \nA Paris, le loyer médian est 1189€, et la superficie médiane est 34m2 (997€ dans les communes limitrophes, et 37m2). Seuls 5% des appartements ont un loyer inférieur à 580€, et seuls 1% moins de 450€. \n \n:face_with_monocle: Si tu indiques un budget trop faible par rapport au marché locatif parisien, tu risques d'obtenir des valeurs abérantes car l'algorithme ne va pas savoir comment gérer ta requête car elle est impossible. ", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://zupimages.net/up/20/16/h0ni.jpeg")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(" :star: <b>Caractéristiques de ta recherche </b> :  \n \n<b>Budget </b>: {}  \n<b>Meublé </b>: {} \n \n Les superficies sont classées de la plus élevée à la plus basse. \n \n 	:small_red_triangle:{} : {}m2 \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n 	:small_red_triangle:{} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2  \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n       {} : {}m2    \n :small_red_triangle_down: {} : {}m2    \n :small_red_triangle_down: {} : {}m2    \n :small_red_triangle_down: {} : {}m2    \n :small_red_triangle_down: {} : {}m2    \n :small_red_triangle_down: {} : {}m2    \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2  \n :small_red_triangle_down: {} : {}m2".format(users[update.effective_chat.id]['rent'], users[update.effective_chat.id]['furnished'], *ordre_list), use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)

    try:
        open('analyse_area_log.csv', 'r')
        with open('analyse_area_log.csv', 'a') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].values())
    except FileNotFoundError:
        with open('analyse_area_log.csv', 'a+') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].keys())
            w.writerow(users[update.effective_chat.id].values())
    return PREDICT_AREA_CATEGORY


# VERIFY  ########################################################################################

def verify(update, context):
    query = update.callback_query
    query.answer()
    reply_markup = InlineKeyboardMarkup(keyboard_verify)
    query.edit_message_text(
        text=emojize("<b>:bar_chart:  Estimer un loyer</b> \n \nTu as trouvé une annonce de location d'appart qui t'interesse à Paris ou aux alentours ? Mon algorithme de machine learning est capable de t'indiquer comment le loyer de cet appart se positionne par rapport au marché.", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup
    )
    return FIRST

def comment_choisir(update, context):
    query = update.callback_query
    query.answer()
    query.edit_message_text(text=emojize("<b>:bar_chart:  Estimer un loyer</b> \n \n En s'appuyant sur une base de données contenant l'ensemble des appartements loués à Paris et dans les communes limotrphes depuis Juillet 2019, des algorithmes de machine learning spécialement entrainés sont capables de prédire le 'vrai' loyer de l'appartement que tu envisages de louer. Si le loyer qu'on te demande est en dessous, tu sauras que tu fais une bonne affaire ! Au contraite, si le loyer est au dessus, il y a de grandes chances que tu paies trop cher.", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML
    )
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("Je te propose deux niveaux d'analyse, en fonction du temps ou du nombre d'informations dont tu disposes. \n \n :zap: <b>Une analyse rapide en 4 questions</b> : \n- L'arrondissement ou la ville \n - La taille (en m2) \n - Le nombre de pièces \n - Si c'est meublé \n \n :star: <b>Une analyse complète en 11 questions</b>: \n - Les 4 questions précédentes \n - Numéro de l'étage \n - Présence d'un balcon \n - Présence d'un concierge \n - Place de parking ou garage attitré \n - Place de parking comprise   \n - Présence d'un lave-linge \n - Présence d'une cuisine américaine ou d'une douche italienne. \n - Appartement neuf ou ecemment rénové, voire de standing \n \nL'estimation avancée permet d'améliorer de 15% la précision du loyer estimé. \n \n :mag: Si tu préfères, à la place de rentrer manuellement les informations, tu peux m'envoyer le lien d'une annonce provenant de LouerAgile, et je m'occuperais de récupérer automatiquement toutes les informations disponibles.", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML)

    reply_markup = InlineKeyboardMarkup(keyboard_verify2)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("Ton choix est fait ? :grinning:", use_aliases=True),
        reply_markup=reply_markup
    )
    return FIRST

# ANALYSE RAPIDE #####################################################################################################
reply_keyboard_paris_couronne = [['Paris', 'Communes limitrophes']]
keyboard_markup_paris_couronne = ReplyKeyboardMarkup(reply_keyboard_paris_couronne, one_time_keyboard=True)

def analyse_rapide_ville_1(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize(":zap: C'est parti pour une estimation rapide du loyer. \n \n :arrow_right: L'appartement se situe à Paris ou dans les communes limitrophes ?", use_aliases=True),
        reply_markup=keyboard_markup_paris_couronne,
        parse_mode=telegram.ParseMode.HTML
    )
    return TYPING_REPLY

reply_keyboard_arrondissements = [['1er - Louvre', '2ème - Bourse'],
                                  ['3ème - Temple', "4ème - Hotel de Ville"],
                                  ["5ème - Panthéon","6ème - Luxembourg"],
                                  ["7ème - Palais-Bourbon","8ème - Elysée"],
                                  ["9ème - Opera","10ème - Enclos Saint-Laurent"],
                                  ["11ème - Popincourt","12ème - Reuilly"],
                                  ["13ème - Gobelins","14ème - Observatoire"],
                                  ["15ème - Vaugigard","16ème - Passy"],
                                  ["17ème - Batignolles-Monceau", "18ème - Butte-Montmartre"],
                                  ["19ème - Buttes-Chaumont", "20ème - Ménilmontant"]]
keyboard_markup_arrondissements = ReplyKeyboardMarkup(reply_keyboard_arrondissements, one_time_keyboard=True)

def analyse_rapide_ville_1_paris(update, context):
    context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo="https://www.hotel-design-secret-de-paris.com/blog/wp-content/uploads/2013/11/arrondissement-paris.gif")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize("L'appartement se situe à <b>Paris</b>. \n \n :arrow_right: Puis-je savoir quel arrondissement ?", use_aliases=True),
        reply_markup=keyboard_markup_arrondissements,
        parse_mode=telegram.ParseMode.HTML
    )
    return TYPING_REPLY

reply_markup_petite_couronne = [['Neuilly-Sur-Seine', 'Boulogne-Billancourt'],
                                  ['Clichy', "Montrouge"],
                                  ["Issy-les-Moulineaux","Aubervilliers"],
                                  ["Vanves","Malakoff"],
                                  ["Levallois-Perret","Montreuil"],
                                  ["Bagnolet","Saint-Denis"],
                                  ["Les Lilas", "Gentilly"],
                                  ["Pré Saint-Gervais", "Saint-Ouen"],
                                  ["Pantin", "Saint-Mandé"],
                                  ["Ivry Sur Seine", "Charenton-le-Pont"],
                                  ["Kremlin-Bicêtre", "Vincennes"]]

keyboard_markup_petite_couronne = ReplyKeyboardMarkup(reply_markup_petite_couronne, one_time_keyboard=True)

def analyse_rapide_ville_1_petite_couronne(update, context):
    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://zupimages.net/up/20/16/h0ni.jpeg")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize("L'appartement se situe dans une <b>commune limitrophe de Paris</b>. \n \n :arrow_right: Puis-je savoir quelle ville ?", use_aliases=True),
        reply_markup=keyboard_markup_petite_couronne,
        parse_mode=telegram.ParseMode.HTML
    )
    return TYPING_REPLY

list_of_city = ['1er - Louvre', '2ème - Bourse',
                '3ème - Temple', "4ème - Hotel de Ville",
                "5ème - Panthéon", "6ème - Luxembourg",
                "7ème - Palais-Bourbon", "8ème - Elysée",
                "9ème - Opera", "10ème - Enclos Saint-Laurent",
                "11ème - Popincourt", "12ème - Reuilly",
                "13ème - Gobelins", "14ème - Observatoire",
                "15ème - Vaugigard", "16ème - Passy",
                "17ème - Batignolles-Monceau", "18ème - Butte-Montmartre",
                "19ème - Buttes-Chaumont", "20ème - Ménilmontant",
                'Neuilly-Sur-Seine', 'Boulogne-Billancourt',
                'Clichy', "Montrouge",
                "Issy-les-Moulineaux", "Aubervilliers",
                "Vanves", "Malakoff",
                "Levallois-Perret", "Montreuil",
                "Bagnolet", "Saint-Denis",
                 "Les Lilas", "Gentilly",
                "Pré Saint-Gervais", "Saint-Ouen",
                "Pantin", "Saint-Mandé",
                "Ivry Sur Seine", "Charenton-le-Pont",
                "Kremlin-Bicêtre", "Vincennes"]

id_of_city = ['ville_75001', 'ville_75002',
                'ville_75003', "ville_75004",
                "ville_75005", "ville_75006",
                "ville_75007", "ville_75008",
                "ville_75009", "ville_75010",
                "ville_75011", "ville_75012",
                "ville_75013", "ville_75014",
                "ville_75015", "ville_75016",
                "ville_75017", "ville_75018",
                "ville_75019", "ville_75020",
                'ville_92200', 'ville_92100',
                'ville_92110', "ville_92120",
                "ville_92130", "ville_93300",
                "ville_92170", "ville_92240",
                "ville_92300", "ville_93100",
                "ville_93170", "ville_93210",
                "ville_93260", "ville_94250",
                "ville_93310", "ville_93400",
                "ville_93500", "ville_94160",
                "ville_94200", "ville_94220",
                "ville_94270", "ville_94300"]

dico_key_values_city = dict(zip(list_of_city, id_of_city))
dico_key_values_city2 = dict(zip(id_of_city, list_of_city))

def analyse_rapide_area(update, context):
    users[update.effective_chat.id]['key_ville'] = update.message.text
    id_ville = dico_key_values_city[users[update.effective_chat.id]['key_ville']]
    update.message.reply_text(emojize("L'appartement est à <b>{}</b>. \nJ'identifie que le code postal est <b>{}</b>. \n \n :arrow_right: Combien de mètres carrés mesure cet appartement ?".format(users[update.effective_chat.id]['key_ville'], dico_key_values_city[users[update.effective_chat.id]['key_ville']][6:]), use_aliases=True), parse_mode=telegram.ParseMode.HTML)
    return TYPING_REPLY_AREA

small_numeric_keyboard2 = [["1", "2", "3", "4"]]
small_numeric_keyboard = ReplyKeyboardMarkup(small_numeric_keyboard2, one_time_keyboard=True, resize_keyboard=True)

def analyse_rapide_area_error(update, context):
        update.message.reply_text(emojize(":robot: Bip bop ! La valeur entrée dois être comprise entre 9 et 200. Seule la valeur doit être envoyée, par exemple '18'.  \n \n :arrow_right: Combien de mètres carrés mesure cet appartement ?".format(users[update.effective_chat.id]['key_ville'], dico_key_values_city[users[update.effective_chat.id]['key_ville']][6:]), use_aliases=True), parse_mode=telegram.ParseMode.HTML)
        return TYPING_REPLY_AREA

def analyse_rapide_room(update, context):
    users[update.effective_chat.id]['area'] = update.message.text
    update.message.reply_text(text=emojize("La taille de l'appartement est de <b>{} m2</b>. \n \n:arrow_right: Combien de pièces il y a ? \n \nPrécision : la cuisine, la salle de bain et les toilettes ne comptent pas.".format(users[update.effective_chat.id]['area']), use_aliases=True), parse_mode=telegram.ParseMode.HTML,  reply_markup=small_numeric_keyboard)
    return TYPING_REPLY_ROOM

reply_keyboard_furnished = [['Meublé', 'Non meublé']]
keyboard_furnished = ReplyKeyboardMarkup(reply_keyboard_furnished, one_time_keyboard=True)

def analyse_rapide_furnished(update, context):
    users[update.effective_chat.id]['room'] = update.message.text
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("L'appartement a <b>{} pièces</b>. \n \n :arrow_right: Est-il loué meublé ?".format(users[update.effective_chat.id]['room']), use_aliases=True),
        parse_mode = telegram.ParseMode.HTML,
        reply_markup=keyboard_furnished
    )
    return TYPING_REPLY_FURNISHED

keyboard_end = [[InlineKeyboardButton(emojize(":compass:  Comparer dans Paris ", use_aliases=True), callback_data=str(COMPARE_PARIS))],
                [InlineKeyboardButton(emojize(":compass:  Comparer avec les villes limitrophes", use_aliases=True), callback_data=str(COMPARE_COURONNE))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(VERIFY))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def resultat(update, context):
    users[update.effective_chat.id]['furnished'] = update.message.text

    personal_dictionary = {**dictionary}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])

    if users[update.effective_chat.id]['key_ville'] != "1er - Louvre":
        personal_dictionary[dico_key_values_city[users[update.effective_chat.id]['key_ville']]] = float(1)
    users[update.effective_chat.id]['result'] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))

    reply_markup = InlineKeyboardMarkup(keyboard_end)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":robot: Bip bop ! J'ai fini de calculer l'estimation rapide du loyer de l'appartement que tu m'as décris. \n \n <b>Localisation </b>: {}  \n <b>Superficie </b>: {}  \n <b>Nombre de pièces </b>: {}  \n <b>Meublé </b>: {} \n \n:zap: L'appartement que tu as décris a un loyer estimé de <b>{}€</b> par mois (charges comprises).".format(users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'], users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], users[update.effective_chat.id]['result']), use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup
    )

    try:
        open('analyse_rapide_log.csv', 'r')
        with open('analyse_rapide_log.csv', 'a') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].values())
    except FileNotFoundError:
        with open('analyse_rapide_log.csv', 'a+') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].keys())
            w.writerow(users[update.effective_chat.id].values())
    return TYPING_REPLY_FURNISHED


keyboard_compare_paris = [
                [InlineKeyboardButton(emojize(":compass:  Comparer avec les villes limitrophes", use_aliases=True), callback_data=str(COMPARE_COURONNE))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(VERIFY))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_rapide_compare_paris(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_paris)

    personal_dictionary = {**dictionary}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])

    predictions = dict()
    for i in list_of_city[0:20]:
        i_id = dico_key_values_city[i]
        if i == "1er - Louvre":
            predictions[i] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))
        else:
            personal_dictionary[i_id] = float(1)
            predictions[i] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))
            personal_dictionary[i_id] = float(0)

    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i, y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://www.hotel-design-secret-de-paris.com/blog/wp-content/uploads/2013/11/arrondissement-paris.gif")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":compass: <b>Comparaison avec les arrondissements de Paris </b> \n \nSi l'appartement que tu m'as décris n'étais pas à {} mais ailleurs dans Paris, quel serait son prix ? C'est ce que je te présente ici. \n \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \nLes loyers estimés sont classées du plus élevé au plus faible. \n \n 	:small_red_triangle:{} : {}€ \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€ ".format(users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'], users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], *ordre_list),
                     use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return TYPING_REPLY_FURNISHED

keyboard_compare_couronne = [
                [InlineKeyboardButton(emojize(":compass:  Comparer dans Paris ", use_aliases=True), callback_data=str(COMPARE_PARIS))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(VERIFY))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_rapide_compare_couronne(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_couronne)

    personal_dictionary = {**dictionary}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])

    predictions = dict()
    for i in list_of_city[20:]:
        i_id = dico_key_values_city[i]
        personal_dictionary[i_id] = float(1)
        predictions[i] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))
        personal_dictionary[i_id] = float(0)

    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i, y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://zupimages.net/up/20/16/h0ni.jpeg")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":compass: <b>Comparaison avec les villes limitrophes à Paris </b> \n \nSi l'appartement que tu m'as décris n'étais pas à {} mais ailleurs dans Paris, quel serait son prix ? C'est ce que je te présente ici. \n \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \nLes loyers estimés sont classées du plus élevé au plus faible. \n \n 	:small_red_triangle:{} : {}€ \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€ ".format(users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'], users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], *ordre_list),
                     use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return TYPING_REPLY_FURNISHED



##  ANALYSE COMPLETE #######################################################################################################################

def analyse_complete_ville_1(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize(":star: C'est parti pour une estimation complète du loyer. \n \n :arrow_right: L'appartement se situe à Paris ou dans une commune limitrophe ?", use_aliases=True),
        reply_markup=keyboard_markup_paris_couronne,
        parse_mode=telegram.ParseMode.HTML
    )
    return TYPING_REPLY_COMPLETE

def analyse_complete_ville_1_paris(update, context):
    context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo="https://www.hotel-design-secret-de-paris.com/blog/wp-content/uploads/2013/11/arrondissement-paris.gif")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize("L'appartement se situe à <b>Paris</b>. \n \n :arrow_right: Puis-je savoir quel arrondissement ?", use_aliases=True),
        reply_markup=keyboard_markup_arrondissements,
        parse_mode=telegram.ParseMode.HTML
    )
    return TYPING_REPLY_COMPLETE

def analyse_complete_ville_1_petite_couronne(update, context):
    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://zupimages.net/up/20/16/h0ni.jpeg")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text= emojize("L'appartement se situe dans une <b>commune limitrophe de Paris</b>. \n \n :arrow_right: Puis-je savoir quelle ville ?", use_aliases=True),
        reply_markup=keyboard_markup_petite_couronne,
        parse_mode=telegram.ParseMode.HTML
    )
    return TYPING_REPLY_COMPLETE

def analyse_complete_area(update, context):
    users[update.effective_chat.id]['key_ville'] = update.message.text
    id_ville = dico_key_values_city[users[update.effective_chat.id]['key_ville']]
    update.message.reply_text(emojize("L'appartement est à <b>{}</b>. \nJ'identifie que le code postal est <b>{}</b>. \n \n :arrow_right: Combien de mètres carrés mesure cet appartement ?".format(users[update.effective_chat.id]['key_ville'], dico_key_values_city[users[update.effective_chat.id]['key_ville']][6:]), use_aliases=True), parse_mode=telegram.ParseMode.HTML)
    return TYPING_REPLY_AREA_COMPLETE

def analyse_complete_area_error(update, context):
        update.message.reply_text(emojize(":robot: Bip bop ! La valeur entrée dois être comprise entre 9 et 200. Seule la valeur doit être envoyée, par exemple '18'.  \n \n :arrow_right: Combien de mètres carrés mesure cet appartement ?".format(users[update.effective_chat.id]['key_ville'], dico_key_values_city[users[update.effective_chat.id]['key_ville']][6:]), use_aliases=True), parse_mode=telegram.ParseMode.HTML)
        return TYPING_REPLY_AREA_COMPLETE

def analyse_complete_room(update, context):
    users[update.effective_chat.id]['area'] = update.message.text
    update.message.reply_text(text=emojize("La taille de l'appartement est de <b>{} m2</b>. \n \n:arrow_right: Combien de pièces il y a ? \n \nPrécision : la cuisine, la salle de bain et les toilettes ne comptent pas.".format(users[update.effective_chat.id]['area']), use_aliases=True), parse_mode=telegram.ParseMode.HTML,  reply_markup=small_numeric_keyboard)
    return TYPING_REPLY_ROOM_COMPLETE

reply_keyboard_furnished = [['Meublé', 'Non meublé']]
keyboard_furnished = ReplyKeyboardMarkup(reply_keyboard_furnished, one_time_keyboard=True)

def analyse_complete_furnished(update, context):
    users[update.effective_chat.id]['room'] = update.message.text
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("L'appartement a <b>{} pièces</b>. \n \n :arrow_right: Est-il loué meublé ?".format(users[update.effective_chat.id]['room']), use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=keyboard_furnished
    )
    return TYPING_REPLY_FURNISHED_COMPLETE

small_numeric_keyboard3 = [["RDC", "1", "2", "3"],
                           ["4", "5", "6", "7"],
                           ["8 ou plus"]]
small_numeric_keyboard_reply = ReplyKeyboardMarkup(small_numeric_keyboard3, one_time_keyboard=True, resize_keyboard=True)

def analyse_complete_floor(update, context):
    users[update.effective_chat.id]['furnished'] = update.message.text
    if users[update.effective_chat.id]['furnished'] == "Meublé":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement est <b>meublé</b>. \n \n :arrow_right: A quel étage se situe-t'il ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=small_numeric_keyboard_reply)
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement est <b>non meublé</b>. \n \n :arrow_right: A quel étage se situe-t'il ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=small_numeric_keyboard_reply)
    return TYPING_REPLY_BALCON_COMPLETE

reply_keyboard_basic = [['Oui', 'Non']]
keyboard_basic = ReplyKeyboardMarkup(reply_keyboard_basic, one_time_keyboard=True)

def analyse_complete_balcon(update, context):
    users[update.effective_chat.id]['floor_brut'] = update.message.text
    if users[update.effective_chat.id]['floor_brut'] == "8 ou plus":
        users[update.effective_chat.id]['floor'] = "8"
    if users[update.effective_chat.id]['floor_brut'] == "RDC":
        users[update.effective_chat.id]['floor'] = "0"
    else:
        users[update.effective_chat.id]['floor'] = users[update.effective_chat.id]['floor_brut']

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize("L'appartement est à <b>l'étage {}</b>. \n \n :arrow_right: A-t'il un balcon ?".format(users[update.effective_chat.id]['floor']), use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=keyboard_basic)
    return TYPING_REPLY_BALCON_COMPLETE

def analyse_complete_gardien(update, context):
    users[update.effective_chat.id]['balcon'] = update.message.text
    if users[update.effective_chat.id]['balcon'] == "Oui":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement a un <b>balcon</b>. \n \n :arrow_right: Y-a-t'il un gardien ou concierge ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement n'a <b>pas d balcon</b>. \n \n :arrow_right: Y-a-t'il un concierge ou gardien ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    return TYPING_REPLY_GARDIEN_COMPLETE

def analyse_complete_parking(update, context):
    users[update.effective_chat.id]['gardien'] = update.message.text
    if users[update.effective_chat.id]['gardien'] == "Oui":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement a un <b>concierge ou gardien</b>. \n \n :arrow_right: Y-a-t'il une place de parking ou un garage attitré ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement n'a <b>pas de concierge ou gardien</b>. \n \n :arrow_right: Y-a-t'il une place de parking ou un garage attitré ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    return TYPING_REPLY_PARKING_COMPLETE

def analyse_complete_machinealaver(update, context):
    users[update.effective_chat.id]['parking'] = update.message.text
    if users[update.effective_chat.id]['parking'] == "Oui":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement a une <b> place de parking ou garage attitré</b> \n \n :arrow_right: Y-a-t'il un lave-linge déja présent ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement n'a <b>pas de place de parking ou garage attitré</b> \n \n :arrow_right: Y-a-t'il un lave-linge déja présent ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    return TYPING_REPLY_MACHINEALAVER_COMPLETE

def analyse_complete_americaine(update, context):
    users[update.effective_chat.id]['machine_a_laver'] = update.message.text
    if users[update.effective_chat.id]['machine_a_laver'] == "Oui":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement a un <b> lave-linge</b>. \n \n :arrow_right: Cet appartement dispose-t'il d'une cuisine américaine, ou d'une douche à l'italienne ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement n'a <b>pas de lave-linge</b>. \n \n :arrow_right: Cet appartement dispose-t'il d'une cuisine américaine, ou d'une douche à l'italienne ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    return TYPING_REPLY_AMERICAINE_COMPLETE

def analyse_complete_qualite(update, context):
    users[update.effective_chat.id]['ameri'] = update.message.text
    if users[update.effective_chat.id]['ameri'] == "Oui":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement a une <b>cuisine à l'américaine ou une douche à l'italienne</b>. \n \n :arrow_right: Dernière question ! Cet appartement est-il neuf ou rénové recemment, voire s'agit-il d'un appartement de standing ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize("L'appartement n'a <b>pas cuisine à l'américaine ou de douche à l'italienne</b>. \n \n :arrow_right: Dernière question ! Cet appartement est-il neuf ou rénové recemment, voire s'agit-il d'un appartement de standing ?", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=keyboard_basic)
    return TYPING_REPLY_QUALITE_COMPLETE

keyboard_compare_complet = [
                [InlineKeyboardButton(emojize(":compass:  Comparer avec Paris", use_aliases=True), callback_data=str(COMPARE_PARIS))],
                [InlineKeyboardButton(emojize(":compass:  Comparer avec les villes limitrophes", use_aliases=True), callback_data=str(COMPARE_COURONNE))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(VERIFY))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def resultat_complete(update, context):
    users[update.effective_chat.id]['qualite'] = update.message.text

    personal_dictionary = {**dictionary_complete}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])
    personal_dictionary['floor'] = float(users[update.effective_chat.id]['floor'])
    personal_dictionary['balcon'] = float(users[update.effective_chat.id]['balcon'] == "Oui")
    personal_dictionary['gardien'] = float(users[update.effective_chat.id]['gardien'] == "Oui")
    personal_dictionary['parking'] = float(users[update.effective_chat.id]['parking'] == "Oui")
    personal_dictionary['machine_a_laver'] = float(users[update.effective_chat.id]['machine_a_laver'] == "Oui")
    personal_dictionary['ameri'] = float(users[update.effective_chat.id]['ameri'] == "Oui")
    personal_dictionary['qualite'] = float(users[update.effective_chat.id]['qualite'] == "Oui")

    if users[update.effective_chat.id]['key_ville'] != "1er - Louvre":
        personal_dictionary[dico_key_values_city[users[update.effective_chat.id]['key_ville']]] = float(1)
    users[update.effective_chat.id]['result'] = int(complete_model.predict(pd.DataFrame([personal_dictionary])))

    reply_markup = InlineKeyboardMarkup(keyboard_compare_complet)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":robot: Bip bop ! J'ai fini de calculer l'estimation complète du loyer recommandé de l'appartement que tu m'as décris. \n \n<b>Localisation </b>: {}  \n<b>Superficie </b>: {}  \n<b>Nombre de pièces </b>: {}  \n <b>Meublé </b>: {} \n<b>Etage : </b> {} \n<b>Balcon</b>: {} \n<b>Concierge </b>: {} \n<b>Parking </b>: {} \n<b>Lave-linge </b>: {} \n<b>Cuisine américaine </b>: {} \n<b>Neuf/rénové </b>: {} \n \n :star: L'appartement que tu as décris a un loyer estimé de <b>{}€</b> par mois (charges comprises).".format(users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'], users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], users[update.effective_chat.id]['floor_brut'], users[update.effective_chat.id]['balcon'], users[update.effective_chat.id]['gardien'], users[update.effective_chat.id]['parking'], users[update.effective_chat.id]['machine_a_laver'], users[update.effective_chat.id]['ameri'],  users[update.effective_chat.id]['qualite'], users[update.effective_chat.id]['result']), use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)

    try:
        open('analyse_complete_log.csv', 'r')
        with open('analyse_complete_log.csv', 'a') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].values())
    except FileNotFoundError:
        with open('analyse_complete_log.csv', 'a+') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].keys())
            w.writerow(users[update.effective_chat.id].values())

    return TYPING_REPLY_QUALITE_COMPLETE

keyboard_compare_paris = [
                [InlineKeyboardButton(emojize(":compass:  Comparer avec les villes limitrophes", use_aliases=True), callback_data=str(COMPARE_COURONNE))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(VERIFY))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_complete_compare_paris(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_paris)

    personal_dictionary = {**dictionary_complete}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])
    personal_dictionary['floor'] = float(users[update.effective_chat.id]['floor'] == "Oui")
    personal_dictionary['balcon'] = float(users[update.effective_chat.id]['balcon'] == "Oui")
    personal_dictionary['gardien'] = float(users[update.effective_chat.id]['gardien'] == "Oui")
    personal_dictionary['parking'] = float(users[update.effective_chat.id]['parking'] == "Oui")
    personal_dictionary['machine_a_laver'] = float(users[update.effective_chat.id]['machine_a_laver'] == "Oui")
    personal_dictionary['ameri'] = float(users[update.effective_chat.id]['ameri'] == "Oui")
    personal_dictionary['qualite'] = float(users[update.effective_chat.id]['qualite'] == "Oui")

    predictions = dict()
    for i in list_of_city[0:20]:
        i_id = dico_key_values_city[i]
        if i == "1er - Louvre":
            predictions[i] = int(complete_model.predict(pd.DataFrame([personal_dictionary])))
        else:
            personal_dictionary[i_id] = float(1)
            predictions[i] = int(complete_model.predict(pd.DataFrame([personal_dictionary])))
            personal_dictionary[i_id] = float(0)
            
    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i,y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://zupimages.net/up/20/16/h0ni.jpeg")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":compass: <b>Comparaison avec les arrondissements de Paris </b> \n \nSi l'appartement que tu m'as décris n'étais pas à {} mais ailleurs dans Paris, quel serait son prix ? C'est ce que je te présente ici. \n \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n <b> Etage : </b> {} \n <b> Balcon </b>: {} \n <b> Concierge </b>: {} \n <b> Parking </b>: {} \n <b> Lave-linge </b>: {} \n <b> Cuisine américaine </b>: {} \n <b> Neuf/rénové </b>: {} \n \nLes loyers estimés sont classées du plus élevé au plus faible. \n \n 	:small_red_triangle:{} : {}€ \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€ ".format(users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'], users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], users[update.effective_chat.id]['floor'], users[update.effective_chat.id]['balcon'], users[update.effective_chat.id]['gardien'], users[update.effective_chat.id]['parking'], users[update.effective_chat.id]['machine_a_laver'], users[update.effective_chat.id]['ameri'],  users[update.effective_chat.id]['qualite'], *ordre_list),
                     use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return TYPING_REPLY_QUALITE_COMPLETE

keyboard_compare_couronne = [
                [InlineKeyboardButton(emojize(":compass:  Comparer dans Paris ", use_aliases=True), callback_data=str(COMPARE_PARIS))],
                [InlineKeyboardButton(emojize(":bar_chart:  Autre estimation", use_aliases=True), callback_data=str(VERIFY))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_complete_compare_couronne(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_couronne)

    personal_dictionary = {**dictionary_complete}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])
    personal_dictionary['floor'] = float(users[update.effective_chat.id]['floor'] == "Oui")
    personal_dictionary['balcon'] = float(users[update.effective_chat.id]['balcon'] == "Oui")
    personal_dictionary['gardien'] = float(users[update.effective_chat.id]['gardien'] == "Oui")
    personal_dictionary['parking'] = float(users[update.effective_chat.id]['parking'] == "Oui")
    personal_dictionary['machine_a_laver'] = float(users[update.effective_chat.id]['machine_a_laver'] == "Oui")
    personal_dictionary['ameri'] = float(users[update.effective_chat.id]['ameri'] == "Oui")
    personal_dictionary['qualite'] = float(users[update.effective_chat.id]['qualite'] == "Oui")

    predictions = dict()
    for i in list_of_city[20:]:
        i_id = dico_key_values_city[i]
        personal_dictionary[i_id] = float(1)
        predictions[i] = int(complete_model.predict(pd.DataFrame([personal_dictionary])))
        personal_dictionary[i_id] = float(0)

    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i,y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_photo(chat_id=update.effective_chat.id, photo="https://www.hotel-design-secret-de-paris.com/blog/wp-content/uploads/2013/11/arrondissement-paris.gif")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":compass: <b>Comparaison avec les villes limitrophes de Paris</b> \n \nSi l'appartement que tu m'as décris n'étais pas à {} mais ailleurs dans Paris, quel serait son prix ? C'est ce que je te présente ici. \n \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n <b> Etage : </b> {} \n <b> Balcon </b>: {} \n <b> Concierge </b>: {} \n <b> Parking </b>: {} \n <b> Lave-linge </b>: {} \n <b> Cuisine américaine </b>: {} \n <b> Neuf/rénové </b>: {}\n \nLes loyers estimés sont classées du plus élevé au plus faible. \n \n 	:small_red_triangle:{} : {}€ \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€ ".format(users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'], users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], users[update.effective_chat.id]['floor'], users[update.effective_chat.id]['balcon'], users[update.effective_chat.id]['gardien'], users[update.effective_chat.id]['parking'], users[update.effective_chat.id]['machine_a_laver'], users[update.effective_chat.id]['ameri'],  users[update.effective_chat.id]['qualite'], *ordre_list),
                     use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return TYPING_REPLY_QUALITE_COMPLETE

# IMPORT ###############################################################################

keyboard_import = [
    [InlineKeyboardButton(emojize(":information_source:  Plus d'informations", use_aliases=True), callback_data=str(INFORMATIONS)),
    InlineKeyboardButton(emojize(":back:  Retour", use_aliases=True), callback_data=str(VERIFY))],
    [InlineKeyboardButton(emojize(":link:  Visiter LouerAgile.fr", use_aliases=True), url="https://www.loueragile.fr/")]]

def analyse_import(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_import)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":mag:  <b>Importer une annonce depuis LouerAgile</b> \n \n Si tu le souhaites, je peux récupérer automatiquement pour toi toutes les informations disponibles d'une annonce de location. \n \nCopie-colle dans le chat le lien d'une annonce provenant de LouerAgile. Le lien doit ressembler à www.LouerAgile.fr/alert_result?token=...", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return ANALYSE_IMPORT

keyboard_import2 = [
    [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_import_informations(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_import2)
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":mag:  <b>Importer une annonce depuis LouerAgile</b>\n \n LouerAgile.fr n'est en aucun cas affilié au chatbot appart-ai. L'utilisation des services de LouerAgile ne fait pas l'objet d'une opération commerciale. LouerAgile ne partage aucune information confidentielle liée à votre compte. \n \n Je suis toujours à l'écoute, envoie moi le lien de l'annonce.", use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return ANALYSE_IMPORT

keyboard_end_import = [[InlineKeyboardButton(emojize(":compass:  Comparer dans Paris ", use_aliases=True), callback_data=str(COMPARE_PARIS))],
                [InlineKeyboardButton(emojize(":compass:  Comparer avec les villes limitrophes", use_aliases=True), callback_data=str(COMPARE_COURONNE))],
                [InlineKeyboardButton(emojize(":mag:  Importer une autre annonce", use_aliases=True), callback_data=str(IMPORT))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_import_result(update, context):
    link = update.message.text
    token = re.search("token=(.*?)&ad=", link).group(1)
    ad = re.search("ad=(.*?)&from=", link).group(1)
    link_get = 'https://api.loueragile.fr/apiv2/alert/' + token + '/ad/' + ad
    r = requests.get(link_get).json()

    users[update.effective_chat.id]['area'] = r['ad']['area']
    users[update.effective_chat.id]['room'] = r['ad']['room']
    users[update.effective_chat.id]['furnished'] = r['ad']['furnished']
    users[update.effective_chat.id]['real_price'] = r['ad']['rent']
    if users[update.effective_chat.id]['furnished'] == 1:
        users[update.effective_chat.id]['furnished'] = "Meublé"
    else :
        users[update.effective_chat.id]['furnished'] = "Non meublé"
    users[update.effective_chat.id]['key_ville'] = dico_key_values_city2[str('ville_' + r['ad']['postal_code'])]

    personal_dictionary = {**dictionary}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])
    if users[update.effective_chat.id]['key_ville'] != "1er - Louvre":
        personal_dictionary[dico_key_values_city[users[update.effective_chat.id]['key_ville']]] = float(1)
    users[update.effective_chat.id]['result'] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))
    users[update.effective_chat.id]['difference'] = users[update.effective_chat.id]['real_price'] - users[update.effective_chat.id]['result']
    users[update.effective_chat.id]['difference_percent'] = users[update.effective_chat.id]['difference']/users[update.effective_chat.id]['real_price']

    reply_markup = InlineKeyboardMarkup(keyboard_end_import)
    if users[update.effective_chat.id]['difference_percent'] > 0.1:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize(
                ":robot: Bip bop ! J'ai fini de calculer l'estimation rapide du loyer estimé de l'appartement dont tu m'as donné le lien ! \n \n <b> Localisation </b>: {}  \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \n :zap: L'appartement que tu as décris a un loyer estimé de <b>{}€</b> par mois (charges comprises). \n \n :red_circle: Attention, Le loyer réel de cet appartement est <b>{}€</b>, soit <b>{}€</b> de plus que le loyer estimé. Cet appartement a un loyer très au dessus de la moyenne du marché.".format(
                    users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'],
                    users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'],
                    users[update.effective_chat.id]['result'], users[update.effective_chat.id]['real_price'],
                users[update.effective_chat.id]['difference']), use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=reply_markup)
    elif users[update.effective_chat.id]['difference_percent'] > 0.05 and users[update.effective_chat.id]['difference_percent'] < 0.1:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize(
                ":robot: Bip bop ! J'ai fini de calculer l'estimation rapide du loyer estimé de l'appartement dont tu m'as donné le lien ! \n \n <b> Localisation </b>: {}  \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \n :zap: L'appartement que tu as décris a un loyer estimé de <b>{}€</b> par mois (charges comprises). \n \n :orange_circle: Le loyer réel de cet appartement est <b>{}€</b>, soit <b>{}€</b> de plus que le loyer estimé. Cet appartement a un loyer au dessus de la moyenne du marché.".format(
                    users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'],
                    users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'],
                    users[update.effective_chat.id]['result'], users[update.effective_chat.id]['real_price'],
                users[update.effective_chat.id]['difference']), use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=reply_markup)
    elif users[update.effective_chat.id]['difference_percent'] > -0.05 and users[update.effective_chat.id]['difference_percent'] < 0.05:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize(
                ":robot: Bip bop ! J'ai fini de calculer l'estimation rapide du loyer estimé de l'appartement dont tu m'as donné le lien ! \n \n <b> Localisation </b>: {}  \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \n :zap: L'appartement que tu as décris a un loyer estimé de <b>{}€</b> par mois (charges comprises). \n \n :blue_circle: Le loyer réel de cet appartement est <b>{}€</b>. Cet appartement a un loyer dans la moyenne du marché.".format(
                    users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'],
                    users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'],
                    users[update.effective_chat.id]['result'], users[update.effective_chat.id]['real_price'],
                users[update.effective_chat.id]['result']), use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=reply_markup)
    elif users[update.effective_chat.id]['difference_percent'] > -0.1 and users[update.effective_chat.id]['difference_percent'] < -0.05 :
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize(
                ":robot: Bip bop ! J'ai fini de calculer l'estimation rapide du loyer estimé de l'appartement dont tu m'as donné le lien ! \n \n <b> Localisation </b>: {}  \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \n :zap: L'appartement que tu as décris a un loyer estimé de <b>{}€</b> par mois (charges comprises). \n \n :green_circle: Le loyer réel de cet appartement est <b>{}€</b>, soit <b>{}€</b> de moins que le loyer estimé. Cet appartement semble un bon plan.".format(
                    users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'],
                    users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'],
                    users[update.effective_chat.id]['result'], users[update.effective_chat.id]['real_price'],
                    users[update.effective_chat.id]['difference']), use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=reply_markup)
    elif users[update.effective_chat.id]['difference_percent'] < -0.1 :
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize(
                ":robot: Bip bop ! J'ai fini de calculer l'estimation rapide du loyer estimé de l'appartement dont tu m'as donné le lien ! \n \n <b> Localisation </b>: {}  \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \n :zap: L'appartement que tu as décris a un loyer estimé de <b>{}€</b> par mois (charges comprises). \n \n :yellow_circle: Le loyer réel de cet appartement est <b>{}€</b>, soit <b>{}€</b> de moins que le loyer estimé. Cet appartement a un prix anormalement bas, renseigne toi bien et sois méfiant, ça pourrait être une arnaque.".format(
                    users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'],
                    users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'],
                    users[update.effective_chat.id]['result'], users[update.effective_chat.id]['real_price'],
                    users[update.effective_chat.id]['difference']), use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=reply_markup)
    else :
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=emojize(
                ":robot: L'algorithme a buggé", use_aliases=True),
            parse_mode=telegram.ParseMode.HTML,
            reply_markup=reply_markup)

    try:
        open('analyse_import_log.csv', 'r')
        with open('analyse_import_log.csv', 'a') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].values())
    except FileNotFoundError:
        with open('analyse_import_log.csv', 'a+') as f:
            w = csv.writer(f)
            w.writerow(users[update.effective_chat.id].keys())
            w.writerow(users[update.effective_chat.id].values())
    return ANALYSE_IMPORT


keyboard_compare_paris_import = [
                [InlineKeyboardButton(emojize(":compass:  Comparer avec les villes limitrophes", use_aliases=True), callback_data=str(COMPARE_COURONNE))],
                [InlineKeyboardButton(emojize(":mag:  Importer une autre annonce", use_aliases=True), callback_data=str(IMPORT))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_rapide_compare_paris_import(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_paris_import)

    personal_dictionary = {**dictionary}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])

    predictions = dict()
    for i in list_of_city[0:20]:
        i_id = dico_key_values_city[i]
        if i == "1er - Louvre":
            predictions[i] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))
        else:
            personal_dictionary[i_id] = float(1)
            predictions[i] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))
            personal_dictionary[i_id] = float(0)

    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i, y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":compass: <b>Comparaison avec les arrondissements de Paris </b> \n \nSi l'appartement que tu m'as décris n'étais pas à {} mais ailleurs dans Paris, quel serait son prix ? C'est ce que je te présente ici. \n \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \nLes loyers estimés sont classées du plus élevé au plus faible. \n \n 	:small_red_triangle:{} : {}€ \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€ ".format(
                users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'],
                users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], *ordre_list),
            use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return ANALYSE_IMPORT

keyboard_compare_couronne_import = [
                [InlineKeyboardButton(emojize(":compass:  Comparer dans Paris ", use_aliases=True), callback_data=str(COMPARE_PARIS))],
                [InlineKeyboardButton(emojize(":mag:  Importer une autre annonce", use_aliases=True), callback_data=str(IMPORT))],
                [InlineKeyboardButton(emojize(":house:  Retour au menu", use_aliases=True), callback_data=str(START))]]

def analyse_rapide_compare_couronne_import(update, context):
    reply_markup = InlineKeyboardMarkup(keyboard_compare_couronne_import)

    personal_dictionary = {**dictionary}
    personal_dictionary['furnished'] = float(users[update.effective_chat.id]['furnished'] == "Meublé")
    personal_dictionary['area'] = float(users[update.effective_chat.id]['area'])
    personal_dictionary['room'] = float(users[update.effective_chat.id]['room'])

    predictions = dict()
    for i in list_of_city[20:]:
        i_id = dico_key_values_city[i]
        personal_dictionary[i_id] = float(1)
        predictions[i] = int(fast_model.predict(pd.DataFrame([personal_dictionary])))
        personal_dictionary[i_id] = float(0)

    ordre = {k: v for k, v in sorted(predictions.items(), reverse=True, key=lambda item: item[1])}
    ordre_list = []
    for i, y in ordre.items():
        ordre_list.append(i)
        ordre_list.append(y)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=emojize(":compass: <b>Comparaison avec les villes limitrophes de Paris </b> \n \nSi l'appartement que tu m'as décris n'étais pas à {} mais ailleurs dans Paris, quel serait son prix ? C'est ce que je te présente ici. \n \n <b> Superficie </b>: {}  \n <b> Nombre de pièces </b>: {}  \n <b> Meublé </b>: {} \n \nLes loyers estimés sont classées du plus élevé au plus faible. \n \n 	:small_red_triangle:{} : {}€ \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n 	:small_red_triangle:{} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n       {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€  \n :small_red_triangle_down: {} : {}€ ".format(
                users[update.effective_chat.id]['key_ville'], users[update.effective_chat.id]['area'],
                users[update.effective_chat.id]['room'], users[update.effective_chat.id]['furnished'], *ordre_list),
            use_aliases=True),
        parse_mode=telegram.ParseMode.HTML,
        reply_markup=reply_markup)
    return ANALYSE_IMPORT

# ERRORS ################################################################################

def unknown1(update, context):
    update.message.reply_text(emojize("Salut ! Prêt à analyser le marché locatif parisien ? :rocket:", use_aliases=True))
    update.message.reply_text("Utilise /start pour commencer.")
    return TYPING_REPLY_TYPE

def unknown(update, context):
    update.message.reply_text(emojize("Bip bop ! Désolé je n'ai pas compris :robot:", use_aliases=True))
    update.message.reply_text("Utilise /start pour que je puisse t'aider.")
    return TYPING_REPLY_TYPE

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

# MAIN ################################################################################

def main():
    # Create the Updater and pass it your bot's token.
    updater = Updater("", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            FIRST: [CallbackQueryHandler(start_predict_area, pattern='^' + str(PREDICT_AREA) + '$'),
                    CallbackQueryHandler(predict_area_budget, pattern='^' + str(PREDICT_AREA_ONE) + '$'),
                    CallbackQueryHandler(verify, pattern='^' + str(VERIFY) + '$'),
                    CallbackQueryHandler(bop, pattern='^' + str(BOP) + '$'),
                    CallbackQueryHandler(credits, pattern='^' + str(CREDITS) + '$'),
                    CallbackQueryHandler(start_over, pattern='^' + str(START) + '$'),
                    CallbackQueryHandler(comment_choisir, pattern='^' + str(COMMENT_CHOISIR) + '$'),
                    CallbackQueryHandler(analyse_rapide_ville_1, pattern='^' + str(ANALYSE_RAPIDE) + '$'),
                    CallbackQueryHandler(analyse_complete_ville_1, pattern='^' + str(ANALYSE_COMPLETE) + '$'),
                    CallbackQueryHandler(analyse_import, pattern='^' + str(IMPORT) + '$'),
                    MessageHandler(Filters.text('/start'), start),
                    MessageHandler(Filters.text, unknown)],

            TYPING_REPLY: [CommandHandler("start", start),
                           MessageHandler(Filters.text("Paris"), analyse_rapide_ville_1_paris),
                           MessageHandler(Filters.text("Communes limitrophes"), analyse_rapide_ville_1_petite_couronne),
                           MessageHandler(Filters.text(emojize(":house: Retour")), analyse_rapide_ville_1),
                           MessageHandler(Filters.text(list_of_city), analyse_rapide_area),
                           MessageHandler(Filters.text, unknown)
                           ],
            TYPING_REPLY_AREA: [MessageHandler(Filters.text(['{:.0f}'.format(x) for x in list(range(9,200))]), analyse_rapide_room),
                                CommandHandler("start", start),
                                MessageHandler(Filters.text, analyse_rapide_area_error)],
            TYPING_REPLY_ROOM: [MessageHandler(Filters.text(['{:.0f}'.format(x) for x in list(range(5))]), analyse_rapide_furnished),
                                CommandHandler("start", start),
                                MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_FURNISHED: [MessageHandler(Filters.text(['Meublé','Non meublé']), resultat),
                                     CallbackQueryHandler(analyse_rapide_compare_paris, pattern='^' + str(COMPARE_PARIS) + '$'),
                                     CallbackQueryHandler(analyse_rapide_compare_couronne, pattern='^' + str(COMPARE_COURONNE) + '$'),
                                     CallbackQueryHandler(verify, pattern='^' + str(VERIFY) + '$'),
                                     CallbackQueryHandler(start_over, pattern='^' + str(START) + '$'),
                                     CommandHandler("start", start),
                                     MessageHandler(Filters.text, unknown)],

            TYPING_REPLY_COMPLETE: [CommandHandler("start", start),
                           MessageHandler(Filters.text("Paris"), analyse_complete_ville_1_paris),
                           MessageHandler(Filters.text("Communes limitrophes"), analyse_complete_ville_1_petite_couronne),
                           MessageHandler(Filters.text(emojize(":house: Retour")), analyse_complete_ville_1),
                           MessageHandler(Filters.text(list_of_city), analyse_complete_area),
                           MessageHandler(Filters.text, unknown)
                           ],
            TYPING_REPLY_AREA_COMPLETE: [
                            MessageHandler(Filters.text(['{:.0f}'.format(x) for x in list(range(9, 200))]), analyse_complete_room),
                            CommandHandler("start", start),
                            MessageHandler(Filters.text, analyse_complete_area_error)],
            TYPING_REPLY_ROOM_COMPLETE: [
                            MessageHandler(Filters.text(['{:.0f}'.format(x) for x in list(range(5))]), analyse_complete_furnished),
                            CommandHandler("start", start),
                            MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_FURNISHED_COMPLETE: [MessageHandler(Filters.text(['Meublé', 'Non meublé']), analyse_complete_floor),
                             CommandHandler("start", start),
                             MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_BALCON_COMPLETE: [MessageHandler(Filters.text(["RDC", "1", '2', '3', '4', '5', '6', '7', '8 ou plus']), analyse_complete_balcon),
                                            MessageHandler(Filters.text(['Oui', 'Non']), analyse_complete_gardien),
                                            CommandHandler("start", start),
                                            MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_GARDIEN_COMPLETE: [MessageHandler(Filters.text(['Oui', 'Non']), analyse_complete_parking),
                                           CommandHandler("start", start),
                                           MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_PARKING_COMPLETE: [MessageHandler(Filters.text(['Oui', 'Non']), analyse_complete_machinealaver),
                                            CommandHandler("start", start),
                                            MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_MACHINEALAVER_COMPLETE: [MessageHandler(Filters.text(['Oui', 'Non']), analyse_complete_americaine),
                                            CommandHandler("start", start),
                                            MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_AMERICAINE_COMPLETE: [MessageHandler(Filters.text(['Oui','Non']), analyse_complete_qualite),
                                            CommandHandler("start", start),
                                            MessageHandler(Filters.text, unknown)],
            TYPING_REPLY_QUALITE_COMPLETE: [MessageHandler(Filters.text(['Oui','Non']), resultat_complete),
                                            CallbackQueryHandler(analyse_complete_compare_paris, pattern='^' + str(COMPARE_PARIS) + '$'),
                                            CallbackQueryHandler(analyse_complete_compare_couronne, pattern='^' + str(COMPARE_COURONNE) + '$'),
                                            CallbackQueryHandler(verify, pattern='^' + str(VERIFY) + '$'),
                                            CallbackQueryHandler(start_over, pattern='^' + str(START) + '$'),
                                            CommandHandler("start", start),
                                            MessageHandler(Filters.text, unknown)],
            ANALYSE_IMPORT:[MessageHandler(Filters.regex('/?token='), analyse_import_result),
                            CallbackQueryHandler(verify, pattern='^' + str(VERIFY) + '$'),
                            CallbackQueryHandler(analyse_import, pattern='^' + str(IMPORT) + '$'),
                            CallbackQueryHandler(analyse_rapide_compare_paris_import, pattern='^' + str(COMPARE_PARIS) + '$'),
                            CallbackQueryHandler(analyse_rapide_compare_couronne_import, pattern='^' + str(COMPARE_COURONNE) + '$'),
                            CallbackQueryHandler(analyse_import_informations,pattern='^' + str(INFORMATIONS) + '$'),
                            CallbackQueryHandler(start_over,pattern='^' + str(START) + '$'),
                            CommandHandler("start", start),
                            MessageHandler(Filters.text, unknown)],

            PREDICT_AREA_CATEGORY: [
                                    MessageHandler(Filters.text(['{:.0f}'.format(x) for x in list(range(400, 4000))]), predict_area_furnished),
                                    MessageHandler(Filters.text(['Meublé', 'Non meublé']), predict_area_paris_couronne),
                                    MessageHandler(Filters.text(['Uniquement Paris']), predict_area_paris),
                                    MessageHandler(Filters.text(['Paris et les communes limitrophes']), predict_area_couronne),
                                    CallbackQueryHandler(predict_area_paris, pattern='^' + str(COMPARE_PARIS) + '$'),
                                    CallbackQueryHandler(predict_area_couronne, pattern='^' + str(COMPARE_COURONNE) + '$'),
                                    CallbackQueryHandler(predict_area_budget, pattern='^' + str(PREDICT_AREA_ONE) + '$'),
                                    CallbackQueryHandler(start_over, pattern='^' + str(START) + '$'),
                                    CommandHandler("start", start),
                                    MessageHandler(Filters.text, predict_area_budget_2)]

        },
        fallbacks=[CommandHandler('start', start)]
    )


    dp.add_handler(conv_handler)
    dp.add_error_handler(error)

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text, unknown1))


    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
