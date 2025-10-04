import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load the model and tokenizer (use your fine-tuned model path)
model_path = "./mbart-en-te-model"  # Change to your model directory
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)

# Set source and target languages
tokenizer.src_lang = "en_XX"
target_lang = "te_IN"

# Translation function
def translate_to_telugu(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=100
    )
    output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return output

# Streamlit UI
st.set_page_config(page_title="English to Telugu Translator", page_icon="🌐")
st.title("🌐 English → Telugu Translation using mBART-50")
st.markdown("Type your English sentence below:")

input_text = st.text_area("📝 English Text", height=150)

if st.button("🔄 Translate"):
    if input_text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        telugu_output = translate_to_telugu(input_text)
        st.success("✅ Telugu Translation:")
        st.write(f"🔤 {telugu_output}")
