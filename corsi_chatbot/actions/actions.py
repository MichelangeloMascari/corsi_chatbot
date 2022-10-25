# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk import Action
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet
import torch
import json
import time
from sentence_transformers import util, SentenceTransformer
from deep_translator import GoogleTranslator
from autocorrect import Speller


class ValidateRipetizioneEserciziForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_ripetizione_esercizi_form"


    def validate_vuole_ripetizione(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
   
        
        if tracker.get_intent_of_latest_message() == "conferma":
            
            return {"vuole_ripetizione": True}
        else:
            SlotSet("vuole_ripetizione",False)
            
            return {"requested_slot": None}

        return {"vuole_ripetizione": None}
        

    def validate_tempo(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
     
     
        valore = tracker.get_slot("vuole_ripetizione")
        print(valore)
        
        if valore :
            dispatcher.utter_message(text="Perfetto!")
            return {"tempo": slot_value}
        else:
            return {"tempo": None}

class ActionParseAll(Action):

    def name(self) -> Text:
        return "action_parse_all"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #Step 1: for all slots get the value, spell it and translate it, save to a dict nomemetadato:metadato
        #Step 2: iterate over the dict keys and parse
        #Step 3: add every parsed result to a Json called request.json and send it to the recsys, save it locally
        


        #instantiate speller, translator and embedder
        spell = Speller('it', fast=True)
        translator = GoogleTranslator(source = "auto", target='en')
        embedder = SentenceTransformer('all-MiniLM-L6-v2')


        #define the json that's going to be sent to RecSys
        request = {
            "request_type": "recommend",
            "user_id": tracker.sender_id,
        }

        #Get all slot values
        slotdict = {
            "duration": "",
            "discipline": "",
            "language": "",
            "keywords": "",
        }

        #Map slots to a dictionary
        slotdict["duration"] = translator.translate(spell(str(tracker.get_slot("durata_lezioni"))))
        slotdict["discipline"] = translator.translate(spell(str(tracker.get_slot("disciplina"))))
        slotdict["language"] = translator.translate(spell(str(tracker.get_slot("lingua"))))
        slotdict["keywords"] = translator.translate(spell(str(tracker.get_slot("argomenti"))))
        
        #iterate over slots and parse
        for key in slotdict:
            #save the queries in a list
            total_kw_list = []
            queries = slotdict[key].split(", ")

            #create the embeddings of all the possible values in MERLOT
            file = open(f'merlotclean/{key}.txt')
            list = file.read().splitlines()
            corpus = list

            #load a pre made tensor for the embeddings
            corpus_embeddings = torch.load(f'tensors/{key}tensor.pt')

            #define the number of responses(needs to be tweaked accordingly)
            top_k = min(5, len(corpus))

            #calculate semantic similarities
            for query in queries:
                query_embedding = embedder.encode(query, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
                hits = hits[0]
                print("Query: "+ query)
                #insert the hits into the Json
                if key != "keywords":
                    hit = hits[0]
                    item_to_append = {key: corpus[hit["corpus_id"]]}
                    request.update(item_to_append)
                #keyword is a multi field and dictionaries cant have multiple objects with the same key
                #so the top hit for every keyword is saved to a list and later put into the json
                else:
                    total_kw_list.append(corpus[hits[0]["corpus_id"]])
                for hit in hits:
                    print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
                print("==============================")
            if key == "keywords":
                keyword_item = {key: total_kw_list}
                request.update(keyword_item)

            #placeholder print, send it to the RecSys(needs to be written)
            print(json.dumps(request))

            #save the request as a json file named: sender_id_currenttime.json
            currtime = time.localtime()
            year = currtime.tm_year
            mon= currtime.tm_mon
            day=currtime.tm_mday
            hour=currtime.tm_hour
            minute = currtime.tm_min
            timestr = f'{year}_{mon}_{day}_{hour}_{minute}'
            f = open(f'C:\\Users\\Mixy\\Documents\\{tracker.sender_id}_{timestr}.json', "w")
            json.dump(request, f, indent=4)
        return []

class ActionRecommend(Action):
    
    def name(self) -> Text:
        return "action_recommend"
    
    def run(self, dispatcher: "CollectingDispatcher", tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Aspetta qualche secondo, sto interrogando il mio sistema per proporti delle risorse")
        ## Creare gli embedding di Disciplina, Keyword, language e duration, questi sono le relazioni del grafo usao come model
        # Capire bene il model e scrivere una funzione di interrogazione per ottenere la raccomandazione

        return []
    
      
        

   

