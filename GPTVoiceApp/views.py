from flask import request,render_template,redirect,url_for,flash, jsonify, Flask, session
from GPTVoiceApp import app
from werkzeug.utils import secure_filename
from GPTVoiceApp.utils.back import transcription
from IPython.display import display, Audio
from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader
from pymongo import MongoClient
from collections import Counter
from pymongo import UpdateOne


import plotly.express as px
import subprocess
import soundfile
import librosa.display
import matplotlib.pyplot as plt
import os
import string 
import requests
import time

import os
import openai


app.secret_key = 'votre_clé_secrète'

def start_mongodb():
    mongodb_process = subprocess.Popen(['mongod'])
    # Attendez un court instant pour laisser le serveur MongoDB s'initialiser
    time.sleep(2)
    return mongodb_process

def stop_mongodb(mongodb_process):
    mongodb_process.terminate()
    mongodb_process.wait()

def open_mongodb_connection():
    client = MongoClient('localhost', 27017)
    nom_base_de_donnees = 'GptVoice'
    db = client[nom_base_de_donnees]
    return client, db

def close_mongodb_connection(client):
    client.close()
        
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# Simple Conversation

@app.route('/', methods=['GET', 'POST'])
def index():
    
    mongodb_process = start_mongodb()
    client, db = open_mongodb_connection()
    
    collection_conversation = db['Llama2ChatBot']
    collection_conversation_all_voices = db['Llama2ChatBotAllVoices']
    collection_all_speak = db['TextToSpeakMultiple']
    
    if request.method == 'POST':
        if 'delete_conversation' in request.form :
            title_to_delete = request.form['delete_conversation']
            collection_conversation_all_voices.delete_one({"title": title_to_delete})
    
    #Discutions
    tous_les_documents = collection_conversation_all_voices.find({}, {"title": 1, "_id": 0})
    conversation = list(tous_les_documents)
    
    close_mongodb_connection(client)
    stop_mongodb(mongodb_process)
    
    return render_template('index.html', conversation=conversation)

    
    
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# Simple Conversation


@app.route('/conversation', methods=['GET', 'POST'])

def conversation():
    
    mongodb_process = start_mongodb()
    client, db = open_mongodb_connection()
    
    collection_conversation = db['Llama2ChatBot']
    collection_conversation_all_voices = db['Llama2ChatBotAllVoices']
    collection_all_speak = db['TextToSpeakMultiple']

    # Configuration du dossier où les fichiers audio seront stockés temporairement
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}
    


    #tests = [2,3,8,20,30,5,6,9,10,40]
    tests = [8,20,9,40]

    if request.method == 'POST':
        if 'show_conversation' in request.form :
            session['conv_name'] = request.form['show_conversation']
            
        if 'message' in request.form and 'title' in request.form:
            collection_conversation_all_voices.insert_one({"title" : request.form['title'], "messages" : [{"role" : "system", "content" : [request.form['message']], "list_fichier" : [[]], 'Voice' : 'system' }]})
            session['conv_name'] = request.form['title']
            
        if 'export' in request.form:
            # Supprimer tous les documents de la collection
            export_json()
            
                    
        if 'reinitialiser' in request.form:
            # Supprimer tous les documents de la collection
            collection_conversation_all_voices.delete_many({})
            audio_directory = 'GPTVoiceApp/static/audio'

            # Parcourez tous les fichiers dans le répertoire
            for filename in os.listdir(audio_directory):
                # Vérifiez si le fichier est un fichier ".wav" et commence par "Llama2ChatBot_"
                if filename.endswith(".wav") and filename.startswith("Llama2ChatBot_"):
                    # Construisez le chemin complet du fichier audio
                    audio_path = os.path.join(audio_directory, filename)

                    # Supprimez le fichier audio
                    os.remove(audio_path)
                    print(f"Fichier audio supprimé : {audio_path}")
                    
        #----------------------------------------------------------------------------------------------------------------
        # Speak To Text
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                normalized_text = SpeakToTextEspnet(file_path)
                return render_template('conversation.html', normalized_text=normalized_text, tests=tests,chart_div=chart_div)

       
        #----------------------------------------------------------------------------------------------------------------
        # TRY DISCUSSION WITH ALL AUDIO AND LLAMA2
        elif 'gpt_question_all_voices' in request.form:
            if request.method == 'POST':
                messages_gpt = []
                all_messages_voices = collection_conversation_all_voices.find_one({"title":  session['conv_name']})
                messages = all_messages_voices.get("messages", [])
                
                # Convertissez le curseur en une liste Python
                all_messages_list_voices = list(messages)

                # Affichez les messages
                for test in all_messages_list_voices:
                    messages_gpt.append({"role" : test['role'], "content" : test['content'][-1]})
                    
                #TRAITEMENT UTILISATEUR 
                gpt_question = request.form.get('gpt_question_all_voices')
                messages_gpt.append({"role": "user", "content": gpt_question},)
                
                wav_filename =  session['conv_name'] + "_" + str(len(messages_gpt)) + "_user.wav"
                
                wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                if os.path.exists(wav_filename2):
                    os.remove(wav_filename2)
                #TextToSpeakEspenet(gpt_question, 55, wav_filename2)
                
                new_message = {"role" : "user", "content" : [gpt_question], "list_fichier" : [[wav_filename]], 'Voice' : 'User' }
                all_messages_list_voices.append(new_message)
                
                result = collection_conversation_all_voices.update_one(
                    {"title": session['conv_name']},
                    {"$set": {"messages": all_messages_list_voices}},
                    upsert=True
                )
                
                print(messages_gpt)
                #REPONSE LLAMA2
                openai.api_base = "http://172.20.10.5:1234/v1" # point to the local server
                openai.api_key = "" # no need for an API key
                completion = openai.ChatCompletion.create(model="local-model", messages=messages_gpt)

                #TRAITEMENT SYSTEME
                
                fichier = []
                for voice, test in enumerate(tests):
                    if voice < 2:
                        wav_filename =  session['conv_name'] + "_" + str(len(messages_gpt) + 1) + "_" + "H" + "_system_voice_" + str(voice + 1) + ".wav"
                        wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                    else :
                        wav_filename =  session['conv_name'] + "_" + str(len(messages_gpt) + 1)  +  "_" + "F" + "_system_voice_" + str(voice - 1) + ".wav"
                        wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                    
                    fichier.append(wav_filename)
                    
                    if os.path.exists(wav_filename2):
                        os.remove(wav_filename2)
                    #TextToSpeakEspenet(completion.choices[0].message['content'], test, wav_filename2)
                    
                
                messages.append({"role": "assistant", "content": completion.choices[0].message['content']},)
                new_message = {"role" : "assistant", "content" : [completion.choices[0].message['content']], "list_fichier" : [fichier], "Voice" : "Assistant" }
                all_messages_list_voices.append(new_message)
                
                collection_conversation_all_voices.update_one(
                    {"title": session['conv_name']},
                    {"$set": {"messages": all_messages_list_voices}}
                )
                
        elif 'reload' in request.form:
            if request.method == 'POST':
                
                messages_gpt = []
                all_messages_voices = collection_conversation_all_voices.find_one({"title":  session['conv_name']})
                messages = all_messages_voices.get("messages", [])
                # Convertissez le curseur en une liste Python
                all_messages_list_voices = list(messages)

                # Affichez les messages
                print(all_messages_list_voices)
                for test in all_messages_list_voices:
                    messages_gpt.append({"role" : test['role'], "content" : test['content'][-1]})
                
                nb_element = len(messages[-1]["content"])
                
                messages_before_changes = all_messages_list_voices
                #supprime le dernier message
                messages_gpt.pop()

                print(messages_gpt)
                
                #REPONSE LLAMA2
                openai.api_base = "http://172.20.10.5:1234/v1" # point to the local server
                openai.api_key = "" # no need for an API key
                completion = openai.ChatCompletion.create(model="local-model", messages=messages_gpt)

                #TRAITEMENT SYSTEME
                
                fichier = []
                for voice, test in enumerate(tests):
                    if voice < 2:
                        wav_filename =  session['conv_name'] + "_v" + str(nb_element + 1) + "_" + str(len(messages_gpt) + 1) + "_" + "H" + "_system_voice_" + str(voice + 1) + ".wav"
                        wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                    else :
                        wav_filename =  session['conv_name'] + "_v" + str(nb_element + 1) + "_" + str(len(messages_gpt) + 1)  +  "_" + "F" + "_system_voice_" + str(voice - 1) + ".wav"
                        wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                    
                    fichier.append(wav_filename)
                    
                    if os.path.exists(wav_filename2):
                        os.remove(wav_filename2)
                    #TextToSpeakEspenet(completion.choices[0].message['content'], test, wav_filename2)
                    
                messages_gpt.append({"role": "assistant", "content": completion.choices[0].message['content']},)
                
                messages_before_changes[-1]['content'].append(completion.choices[0].message['content'])
                messages_before_changes[-1]['list_fichier'].append(fichier)
                
                result = collection_conversation_all_voices.update_one(
                    {"title": session['conv_name']},
                    {
                        "$set": {
                            "messages": messages_before_changes,
                        }
                    }
                )
        elif 'generate_voice' in request.form:
            all_messages_voices = collection_conversation_all_voices.find_one({"title":  session['conv_name']})
            messages = all_messages_voices.get("messages", [])
            # Convertissez le curseur en une liste Python
            all_messages_list_voices = list(messages)
            
            for num_base, i in enumerate(all_messages_list_voices):
                if(num_base != 0):
                    for num, text in enumerate(i['content']):
                        if i['role'] == "system" or i['role'] == "assistant" : 
                            for num_audio, voice in enumerate(tests):
                                file_path =  "GPTVoiceApp/static/audio/" + i['list_fichier'][num][num_audio]
                                print(file_path)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                TextToSpeakEspenet(text, voice, file_path)
                        else:
                            file_path =  "GPTVoiceApp/static/audio/" + i['list_fichier'][num][0]
                            print(file_path)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            TextToSpeakEspenet(text, 55, file_path)
        
        elif 'Delete' in request.form:
            conv_name = session['conv_name']
            id_message = request.form.get('id_message')
            
            all_messages_voices = collection_conversation_all_voices.find_one({"title":  session['conv_name']})
            messages = all_messages_voices.get("messages", [])
            # Convertissez le curseur en une liste Python
            all_messages_list_voices = list(messages)
            all_messages_list_voices.pop(int(id_message))
            
            
            result = collection_conversation_all_voices.update_one(
                {"title": session['conv_name']},
                {
                    "$set": {
                        "messages": all_messages_list_voices,
                    }
                }
            )

    #Discutions
    tous_les_documents = collection_conversation_all_voices.find_one({"title": session['conv_name']})
    messages2 = tous_les_documents.get("messages", [])
    documents_liste = list(messages2)
    
    close_mongodb_connection(client)
    stop_mongodb(mongodb_process)
    
    return render_template('conversation.html', response_gpt_V4 = documents_liste,)




#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# OTHER

@app.route('/get_tests', methods=['GET'])
def get_tests():
    
    mongodb_process = start_mongodb()
    client, db = open_mongodb_connection()

    collection_conversation_all_voices = db['Llama2ChatBotAllVoices']
    
    # Retrieve all users from the 'users' collection
    tests = collection_conversation_all_voices.find({"title" : "Nurse"}, {'_id': 0})
    test_list = list(tests)
    
    close_mongodb_connection(client)
    stop_mongodb(mongodb_process)
    
    return jsonify(test_list)

@app.route('/test22', methods=['GET'])
def tests22():
    # Retrieve all users from the 'users' collection
    tests = collection_conversation_all_voices.find({}, {'_id': 0})
    test_list = list(tests)
    
    return jsonify(test_list)


def get_test_shart():
    # Retrieve all users from the 'users' collection
    tests = collection_test.find({}, {'_id': 0})
    test_list = list(tests)    
    return (test_list)


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# TEXT TO SPEAK ESPENET

def TextToSpeakEspenet(text_to_speak, vocoder, wav_filename):
    from espnet2.bin.tts_inference import Text2Speech
    from espnet2.utils.types import str_or_none
    from espnet_model_zoo.downloader import ModelDownloader
    from IPython.display import display, Audio
    from scipy.io import wavfile

    import glob
    import os
    import numpy as np
    import kaldiio
    import time
    import torch
    import base64

    import tempfile
    from flask import send_file
    import io

    print(vocoder)
    #@title English multi-speaker pretrained model { run: "auto" }
    lang = 'English'
    tag = "kan-bayashi/vctk_full_band_multi_spk_vits"
    # "kan-bayashi/vctk_gst_tacotron2", 
    #  "kan-bayashi/vctk_gst_transformer", 
    #  "kan-bayashi/vctk_xvector_tacotron2", 
    #  "kan-bayashi/vctk_xvector_transformer", 
    #  "kan-bayashi/vctk_xvector_conformer_fastspeech2", 
    #  "kan-bayashi/vctk_gst+xvector_tacotron2", 
    #  "kan-bayashi/vctk_gst+xvector_transformer", 
    #  "kan-bayashi/vctk_gst+xvector_conformer_fastspeech2", 
    #  "kan-bayashi/vctk_multi_spk_vits", 
    #  "kan-bayashi/vctk_full_band_multi_spk_vits", 
    #  "kan-bayashi/libritts_xvector_transformer", 
    #  "kan-bayashi/libritts_xvector_conformer_fastspeech2", 
    #  "kan-bayashi/libritts_gst+xvector_transformer", 
    #  "kan-bayashi/libritts_gst+xvector_conformer_fastspeech2", 
    #  "kan-bayashi/libritts_xvector_vits"
    vocoder_tag = "parallel_wavegan/vctk_parallel_wavegan.v1.long"
    
    if not os.path.isfile(wav_filename):
        text2speech = Text2Speech.from_pretrained(
            model_tag=str_or_none(tag),
            vocoder_tag=str_or_none(vocoder_tag),
            device="cuda",
            # Only for Tacotron 2 & Transformer
            threshold=0.5,
            # Only for Tacotron 2
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3,
            # Only for FastSpeech & FastSpeech2 & VITS
            speed_control_alpha=1.0,
            # Only for VITS
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )

        d = ModelDownloader()
        model_dir = os.path.dirname(d.download_and_unpack(tag)["train_config"])

        # X-vector selection
        spembs = None
        if text2speech.use_spembs:
            xvector_ark = [p for p in glob.glob(f"{model_dir}/../../dump/**/spk_xvector.ark", recursive=True) if "tr" in p][0]
            xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
            spks = list(xvectors.keys())

            # randomly select speaker
            random_spk_idx = np.random.randint(0, len(spks))
            spk = spks[random_spk_idx]
            spembs = xvectors[spk]
            print(f"selected spk: {spk}")

        # Speaker ID selection
        sids = None
        if text2speech.use_sids:
            spk2sid = glob.glob(f"{model_dir}/../../dump/**/spk2sid", recursive=True)[0]
            with open(spk2sid) as f:
                lines = [line.strip() for line in f.readlines()]
            sid2spk = {int(line.split()[1]): line.split()[0] for line in lines}
            
            # randomly select speaker
            sids = np.array(vocoder)
            spk = sid2spk[int(sids)]
            print(f"selected spk: {spk}")

        # Reference speech selection for GST
        speech = None
        if text2speech.use_speech:
            # you can change here to load your own reference speech
            # e.g.
            # import soundfile as sf
            # speech, fs = sf.read("/path/to/reference.wav")
            # speech = torch.from_numpy(speech).float()
            speech = torch.randn(50000,) * 0.01

        # synthesis
        with torch.no_grad():
            start = time.time()
            wav = text2speech(text_to_speak, speech=speech, spembs=spembs, sids=sids)["wav"]
        rtf = (time.time() - start) / (len(wav) / text2speech.fs)
        print(f"RTF = {rtf:5f}")

        
        display(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs))
        wavfile.write(wav_filename, text2speech.fs, wav.view(-1).cpu().numpy())

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# ESPENET SPEAK TO TEXT

def SpeakToTextEspnet(file_path):
    #@title Choose English ASR model { run: "auto" }
    lang = 'en'
    fs = 16000 #@param {type:"integer"}
    tag = 'Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave' #@param ["Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave", "kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave"] {type:"string"}

    d = ModelDownloader()
    # It may takes a while to download and build models
    speech2text = Speech2Text(
        **d.download_and_unpack(tag),
        device="cuda",
        minlenratio=0.0,
        maxlenratio=0.0,
        ctc_weight=0.3,
        beam_size=10,
        batch_size=0,
        nbest=1
    )

    speech, rate = soundfile.read(file_path)
        # Ajoutez le reste de votre code de traitement audio ici
    nbests = speech2text(speech)
    text, *_ = nbests[0]

    display(Audio(speech, rate=rate))
    librosa.display.waveshow(speech, sr=rate)
    plt.show()
    print(f"ASR hypothesis: {text_normalizer(text)}")
    print("*" * 50)
    normalized_text = text_normalizer(text)

    return render_template('conversation.html' , normalized_text=normalized_text, tests= tests)

def export_json():
    cursor = collection_conversation.find({})
    documents = [doc for doc in cursor]

    # Exporter vers JSON
    with open('export.json', 'w') as json_file:
        json_file.write(jsonify(documents))
        
@app.route('/del_data', methods=['GET', 'POST'])
def del_data():
    mongodb_process = start_mongodb()
    client, db = open_mongodb_connection()
    
    collection_conversation = db['Llama2ChatBot']
    collection_conversation_all_voices = db['Llama2ChatBotAllVoices']
    collection_all_speak = db['TextToSpeakMultiple']
    
    collection_conversation_all_voices.delete_many({})
    audio_directory = 'GPTVoiceApp/static/audio'

    # Parcourez tous les fichiers dans le répertoire
    for filename in os.listdir(audio_directory):
        # Vérifiez si le fichier est un fichier ".wav" et commence par "Llama2ChatBot_"
        if filename.endswith(".wav") and filename.startswith("Llama2ChatBot_"):
            # Construisez le chemin complet du fichier audio
            audio_path = os.path.join(audio_directory, filename)

            # Supprimez le fichier audio
            os.remove(audio_path)
            print(f"Fichier audio supprimé : {audio_path}")
            
    close_mongodb_connection(client)
    stop_mongodb(mongodb_process)
    return 'lol'

@app.route('/json', methods=['GET', 'POST'])
def get_json():
    mongodb_process = start_mongodb()
    client, db = open_mongodb_connection()

    collection_conversation_all_voices = db['Llama2ChatBotAllVoices']
    
    cursor = collection_conversation_all_voices.find({})
    mongo_data = list(cursor)

    # Convertir ObjectId en str pour la sérialisation JSON
    for entry in mongo_data:
        entry['_id'] = str(entry['_id'])

    # Conversion en JSON avec jsonify
    json_data = jsonify(mongo_data)

    close_mongodb_connection(client)
    stop_mongodb(mongodb_process)
    
    return json_data
    


