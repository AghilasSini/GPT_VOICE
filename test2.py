#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# Simple Index

@app.route('/simple', methods=['GET', 'POST'])
def simple():
    
    mongodb_process = start_mongodb()
    client, db = open_mongodb_connection()

    collection_conversation_all_voices = db['Llama2ChatBotAllVoices2']

    # Configuration du dossier où les fichiers audio seront stockés temporairement
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

    #tests = [2,3,8,20,30,5,6,9,10,40]
    tests = [8,20,9,40]

    if request.method == 'POST':
        if 'export' in request.form:
            # Supprimer tous les documents de la collection
            export_json()
            
                    
        if 'reinitialiser' in request.form:
            # Supprimer tous les documents de la collection
            collection_conversation_all_voices.delete_many({})
            audio_directory = 'GPTVoiceApp/static/audio'

            # Parcourez tous les fichiers dans le répertoire
            # for filename in os.listdir(audio_directory):
            #     # Vérifiez si le fichier est un fichier ".wav" et commence par "Llama2ChatBot_"
            #     if filename.endswith(".wav") and filename.startswith("Llama2ChatBot_"):
            #         # Construisez le chemin complet du fichier audio
            #         audio_path = os.path.join(audio_directory, filename)

            #         # Supprimez le fichier audio
            #         os.remove(audio_path)
            #         print(f"Fichier audio supprimé : {audio_path}")
                    
        #----------------------------------------------------------------------------------------------------------------
        # Speak To Text
        # if 'file' in request.files:
        #     file = request.files['file']
        #     if file and allowed_file(file.filename):
        #         filename = secure_filename(file.filename)
        #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #         normalized_text = SpeakToTextEspnet(file_path)
        #         return render_template('conversation.html', normalized_text=normalized_text, tests=tests,chart_div=chart_div)

        
        #----------------------------------------------------------------------------------------------------------------
        # TRY DISCUSSION WITH ALL AUDIO AND LLAMA2
        elif 'gpt_question_all_voices' in request.form:
            if request.method == 'POST':
                
                print("yoyoyoyooyioyo")
                messages = []
                all_messages_voices = collection_conversation_all_voices.find()

                # Convertissez le curseur en une liste Python
                all_messages_list_voices = list(all_messages_voices)

                messages.append({"role" : "system", "content" : "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Your response need to be short."})
                # Affichez les messages
                for test in all_messages_list_voices:
                    messages.append({"role" : test['role'], "content" : test['content'][-1]})
                    
                #TRAITEMENT UTILISATEUR 
                gpt_question = request.form.get('gpt_question_all_voices')
                messages.append({"role": "user", "content": gpt_question},)
                
                wav_filename = []
                # wav_filename = "Llama2ChatBot_" + str(len(messages)) + "_user.wav"
                
                # wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                # if os.path.exists(wav_filename2):
                #     os.remove(wav_filename2)
                # TextToSpeakEspenet(gpt_question, 55, len(messages), wav_filename2)
                
                
                collection_conversation_all_voices.insert_one({"role" : "user", "content" : [gpt_question], "list_fichier" : [wav_filename]})

                
                #REPONSE LLAMA2
                openai.api_base = "http://172.20.10.5:1234/v1" # point to the local server
                openai.api_key = "" # no need for an API key
                completion = openai.ChatCompletion.create(model="local-model", messages=messages)

                #TRAITEMENT SYSTEME
                
                fichier = []
                # for voice, test in enumerate(tests):
                #     if voice < 5:
                #         wav_filename = "Llama2ChatBot_" + str(len(messages) + 1) + "_" + "H" + "_system_voice_" + str(voice + 1) + ".wav"
                #         wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                #     else :
                #         wav_filename = "Llama2ChatBot_" + str(len(messages) + 1)  +  "_" + "F" + "_system_voice_" + str(voice - 4) + ".wav"
                #         wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                    
                #     fichier.append(wav_filename)
                    
                #     if os.path.exists(wav_filename2):
                #         os.remove(wav_filename2)
                #     TextToSpeakEspenet(completion.choices[0].message['content'], test, len(messages), wav_filename2)
                    
                
                messages.append({"role": "system", "content": completion.choices[0].message['content']},)
                collection_conversation_all_voices.insert_one({"role" : "system", "content" : [completion.choices[0].message['content']], "list_fichier" : [fichier]})
    
        elif 'reload' in request.form:
            if request.method == 'POST':
                
                messages = []
                all_messages_voices = collection_conversation_all_voices.find()

                # Convertissez le curseur en une liste Python
                all_messages_list_voices = list(all_messages_voices)

                # Affichez les messages
                for test in all_messages_list_voices:
                    messages.append({"role" : test['role'], "content" : test['content'][-1]})
                
                print(messages)
                
                #supprime le dernier message
                messages.pop()

                #REPONSE LLAMA2
                openai.api_base = "http://172.20.10.5:1234/v1" # point to the local server
                openai.api_key = "" # no need for an API key
                completion = openai.ChatCompletion.create(model="local-model", messages=messages)

                #TRAITEMENT SYSTEME
                
                fichier = []
                # for voice, test in enumerate(tests):
                #     if voice < 5:
                #         wav_filename = "Llama2ChatBot_" + str(len(messages) + 1) + "_" + "H" + "_system_voice_" + str(voice + 1) + ".wav"
                #         wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                #     else :
                #         wav_filename = "Llama2ChatBot_" + str(len(messages) + 1)  +  "_" + "F" + "_system_voice_" + str(voice - 4) + ".wav"
                #         wav_filename2 = "GPTVoiceApp/static/audio/" + wav_filename
                    
                #     fichier.append(wav_filename)
                    
                #     if os.path.exists(wav_filename2):
                #         os.remove(wav_filename2)
                #     TextToSpeakEspenet(completion.choices[0].message['content'], test, len(messages), wav_filename2)
                    
                messages.append({"role": "system", "content": completion.choices[0].message['content']},)
                dernier_document = collection_conversation_all_voices.find_one(sort=[('_id', -1)])
                
                
                
                result = collection_conversation_all_voices.update_one(
                    {"_id": dernier_document["_id"]},
                    {
                        "$push": {
                            "content": completion.choices[0].message['content'],
                            "list_fichier": fichier
                        }
                    }
                )
    
    #Discutions
    tous_les_documents = collection_conversation_all_voices.find()
    documents_liste = list(tous_les_documents)
    
    close_mongodb_connection(client)
    stop_mongodb(mongodb_process)
    
    return render_template('simple.html', response_gpt_V4 = documents_liste,)
    #Discutions
    tous_les_documents = collection_conversation_all_voices.find()
    documents_liste = list(tous_les_documents)
    
    close_mongodb_connection(client)
    stop_mongodb(mongodb_process)
    
    return render_template('simple.html', response_gpt_V4 = documents_liste,)

    