import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ipc import find_ipc_hybrid

# Load fastText language detector
lang_model = fasttext.load_model("lid.176.bin")

# Load NLLB-200 translator
nllb_model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
translator = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

def detect_lang(text):
    label, prob = lang_model.predict(text)
    lang_code = label[0].replace("__label__", "")
    return lang_code, prob

def translate(text, target_lang):
    """Translate text to target language using NLLB"""
    try:
        # Encode input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get target language token ID (direct approach that works)
        target_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        
        # Generate translation
        translated_tokens = translator.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_length=512,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True
        )
        
        # Decode and return
        result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return result
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def process_input(user_text):
    # 1. Detect language
    src_lang, prob = detect_lang(user_text)

    # Map fastText → NLLB codes
    lang_map = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "te": "tel_Telu",
        "ta": "tam_Taml",
        "ml": "mal_Mlym",
        "kn": "kan_Knda",
        "mr": "mar_Deva",
        "bn": "ben_Beng",
        "gu": "guj_Gujr",
        "pa": "pan_Guru",
        "or": "ori_Orya",
        "ur": "urd_Arab"
    }

    # If already English, skip translation
    if src_lang == "en":
        eng_text = user_text
    else:
        eng_text = translate(user_text, "eng_Latn")

    # 2. Run IPC classifier (your existing function)
    ipc_results = find_ipc_hybrid(eng_text, top_k=3)

    # 3. Translate back results to user’s original language
    final_results = []
    for r in ipc_results:
        result_text = f"Section {r['IPC']}: {r['offense']} | Punishment: {r['punishment']}"
        if src_lang != "en":
            result_text = translate(result_text, lang_map[src_lang])
        final_results.append(result_text)

    return final_results

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    print("   Multilingual IPC Legal Assistant")
    print("   Enter your legal query in any supported language")
    print("   Supported: English, Hindi, Telugu, Tamil, Malayalam, Kannada, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu")
    print("   Type 'quit' to exit\n")
    
    while True:
        user_query = input("Enter your legal query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_query:
            print("Please enter a query.")
            continue
            
        print("Processing (detecting language, translating, analyzing law)...")
        
        try:
            # Detect language first
            detected_lang, confidence = detect_lang(user_query)
            print(f"Detected language: {detected_lang} (confidence: {confidence[0]:.3f})")
            
            # Process the query
            results = process_input(user_query)
            
            if results:
                print(f"\nLegal Analysis Results:\n" + "="*50)
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result}")
                print("="*50)
            else:
                print("No relevant legal sections found. Try rephrasing your query.")
                
        except KeyError as e:
            print(f"Language not supported: {e}")
            print("Try using English or one of the supported Indian languages.")
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try again with a different query.")
        
        print("\n" + "-"*60)
