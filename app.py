from brain import encoded_image, analyze_image_with_query
from voice_patients import record_audio, trascribe_with_groq
from voice_doctors import text_to_speech_with_gtts
import os
import gradio as gr
import langdetect


# Language detection function
def detect_language(text):
    """Attempt to detect language from text"""
    try:
        detected = langdetect.detect(text)
        # Map detected language codes to our supported languages
        language_map = {
            "en": "en",  # English
            "hi": "hi",  # Hindi
            "pa": "pa",  # Punjabi
            "ml": "ml",  # Malayalam
        }
        return language_map.get(detected, "en")  # Default to English if not in our supported list
    except:
        return "en"  # Default to English if detection fails


# System prompts for different languages
def get_system_prompt(language="en"):
    """Get appropriate system prompt for the selected language"""
    prompts = {
        "en": """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
                What's in this image?. Do you find anything wrong with it medically? 
                If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
                your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
                Donot say 'In the image I see' but say 'With what I see, I think you have ....'
                Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
                Keep your answer concise (max 2 sentences). No preamble, start your answer right away please""",

        "hi": """आपको एक पेशेवर डॉक्टर के रूप में कार्य करना है, मुझे पता है कि आप नहीं हैं लेकिन यह सीखने के उद्देश्य के लिए है।
                इस छवि में क्या है? क्या आपको इसमें चिकित्सकीय रूप से कुछ गलत लगता है?
                यदि आप एक निदान करते हैं, तो उनके लिए कुछ उपचार सुझाएं। अपनी प्रतिक्रिया में कोई संख्या या विशेष वर्ण न जोड़ें।
                आपकी प्रतिक्रिया एक लंबे पैराग्राफ में होनी चाहिए। साथ ही हमेशा ऐसे जवाब दें जैसे आप किसी वास्तविक व्यक्ति को जवाब दे रहे हों।
                'छवि में मैं देखता हूं' न कहें बल्कि कहें 'जो मैं देखता हूं, मुझे लगता है कि आपको....'
                एक AI मॉडल के रूप में जवाब न दें, आपका जवाब एक AI बॉट के नहीं बल्कि वास्तविक डॉक्टर जैसा होना चाहिए,
                अपना जवाब संक्षिप्त रखें (अधिकतम 2 वाक्य)। कोई प्रस्तावना नहीं, कृपया अपना जवाब तुरंत शुरू करें""",

        "pa": """ਤੁਹਾਨੂੰ ਇੱਕ ਪੇਸ਼ੇਵਰ ਡਾਕਟਰ ਵਜੋਂ ਕੰਮ ਕਰਨਾ ਹੈ, ਮੈਨੂੰ ਪਤਾ ਹੈ ਕਿ ਤੁਸੀਂ ਨਹੀਂ ਹੋ ਪਰ ਇਹ ਸਿੱਖਣ ਦੇ ਉਦੇਸ਼ ਲਈ ਹੈ।
                ਇਸ ਤਸਵੀਰ ਵਿੱਚ ਕੀ ਹੈ? ਕੀ ਤੁਹਾਨੂੰ ਇਸ ਵਿੱਚ ਕੋਈ ਚਿਕਿਤਸਕ ਰੂਪ ਵਿੱਚ ਗਲਤ ਲੱਗਦਾ ਹੈ?
                ਜੇ ਤੁਸੀਂ ਇੱਕ ਨਿਦਾਨ ਕਰਦੇ ਹੋ, ਤਾਂ ਉਨ੍ਹਾਂ ਲਈ ਕੁਝ ਇਲਾਜ ਸੁਝਾਓ। ਆਪਣੇ ਜਵਾਬ ਵਿੱਚ ਕੋਈ ਨੰਬਰ ਜਾਂ ਵਿਸ਼ੇਸ਼ ਅੱਖਰ ਨਾ ਜੋੜੋ।
                ਤੁਹਾਡਾ ਜਵਾਬ ਇੱਕ ਲੰਬੇ ਪੈਰਾਗ੍ਰਾਫ ਵਿੱਚ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ। ਹਮੇਸ਼ਾ ਇਸ ਤਰ੍ਹਾਂ ਜਵਾਬ ਦਿਓ ਜਿਵੇਂ ਤੁਸੀਂ ਅਸਲ ਵਿੱਚ ਕਿਸੇ ਵਿਅਕਤੀ ਨੂੰ ਜਵਾਬ ਦੇ ਰਹੇ ਹੋ।
                'ਤਸਵੀਰ ਵਿੱਚ ਮੈਂ ਦੇਖਦਾ ਹਾਂ' ਨਾ ਕਹੋ ਪਰ ਕਹੋ 'ਜੋ ਮੈਂ ਦੇਖਦਾ ਹਾਂ, ਮੈਨੂੰ ਲਗਦਾ ਹੈ ਕਿ ਤੁਹਾਡੇ ਕੋਲ....'
                ਇੱਕ AI ਮਾਡਲ ਵਜੋਂ ਜਵਾਬ ਨਾ ਦਿਓ, ਤੁਹਾਡਾ ਜਵਾਬ AI ਬੋਟ ਨਹੀਂ ਸਗੋਂ ਇੱਕ ਅਸਲ ਡਾਕਟਰ ਵਰਗਾ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ,
                ਆਪਣਾ ਜਵਾਬ ਸੰਖੇਪ ਰੱਖੋ (ਵੱਧ ਤੋਂ ਵੱਧ 2 ਵਾਕ)। ਕੋਈ ਭੂਮਿਕਾ ਨਹੀਂ, ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਜਵਾਬ ਤੁਰੰਤ ਸ਼ੁਰੂ ਕਰੋ""",

        "ml": """നിങ്ങൾ ഒരു പ്രൊഫഷണൽ ഡോക്ടറായി പ്രവർത്തിക്കണം, നിങ്ങൾ അല്ലെന്ന് എനിക്കറിയാം, പക്ഷേ ഇത് പഠന ആവശ്യത്തിനാണ്.
                ഈ ചിത്രത്തിൽ എന്താണുള്ളത്? മെഡിക്കലി അതിൽ എന്തെങ്കിലും തെറ്റ് കാണുന്നുണ്ടോ?
                നിങ്ങൾ ഒരു രോഗനിർണയം നടത്തുകയാണെങ്കിൽ, അവയ്ക്കായി ചില പരിഹാരങ്ങൾ നിർദ്ദേശിക്കുക. നിങ്ങളുടെ മറുപടിയിൽ അക്കങ്ങളോ പ്രത്യേക പ്രതീകങ്ങളോ ചേർക്കരുത്.
                നിങ്ങളുടെ മറുപടി ഒരു നീണ്ട ഖണ്ഡികയിലായിരിക്കണം. യഥാർത്ഥ വ്യക്തിയോട് സംസാരിക്കുന്നതുപോലെ എപ്പോഴും മറുപടി നൽകുക.
                'ചിത്രത്തിൽ ഞാൻ കാണുന്നു' എന്ന് പറയാതെ 'ഞാൻ കാണുന്നത് വച്ച്, നിങ്ങൾക്ക് ഉണ്ടെന്ന് തോന്നുന്നു....'
                ഒരു AI മോഡൽ എന്ന നിലയിൽ മറുപടി നൽകരുത്, നിങ്ങളുടെ മറുപടി ഒരു AI ബോട്ടിന്റേതല്ല മറിച്ച് യഥാർത്ഥ ഡോക്ടറിന്റെ രീതിയിലായിരിക്കണം,
                നിങ്ങളുടെ മറുപടി ചുരുക്കി പറയുക (പരമാവധി 2 വാക്യങ്ങൾ). ആമുഖമില്ല, ദയവായി നിങ്ങളുടെ മറുപടി ഉടൻ തുടങ്ങുക"""
    }

    return prompts.get(language, prompts["en"])


def process_inputs(audio_filepath, image_filepath, language="auto"):
    # Default to English if no language specified
    if language == "auto" or not language:
        # First transcribe the audio to detect language
        speech_to_text_output = trascribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )

        # Detect language from the transcription
        detected_language = detect_language(speech_to_text_output)
    else:
        # Use the selected language
        detected_language = language
        # Now transcribe with known language for better accuracy
        speech_to_text_output = trascribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3",
            language=detected_language
        )

    if image_filepath:
        # Get the system prompt in the detected language
        system_prompt = get_system_prompt(detected_language)

        # Analyze image with the query in the detected language
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encoded_image(image_filepath),
            model="llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = "No image provided for me to analyze"

    # Generate doctor's voice response in the detected language
    output_audio_filepath = "final.mp3"
    voice_of_doctor = text_to_speech_with_gtts(
        input_text=doctor_response,
        output_filepath=output_audio_filepath,
        language=detected_language
    )

    return speech_to_text_output, doctor_response, voice_of_doctor


# Create Gradio interface - keeping your original structure
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath"),
        gr.Dropdown(choices=["auto", "en", "hi", "pa", "ml"],
                    value="auto",
                    label="Language (auto, English, Hindi, Punjabi, Malayalam)")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice", autoplay=True)
    ],
    title="Multilingual AI Doctor with Vision and Voice"
)

iface.launch(server_name="0.0.0.0", server_port=8080)