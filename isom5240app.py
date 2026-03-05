# Program title: Storytelling App

# import part
import streamlit as st
from transformers import pipeline

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    pipe = pipeline(
        "text-generation",
        model="pranavpsv/gpt2-genre-story-generator"
    )

    prompt = (
        "You are a friendly storyteller for young children.\n"
        "Write a simple, warm and positive story in English for a child aged 3 to 10.\n"
        "The story should be between 50 and 100 words.\n"
        f"The story is about: {text}\n\n"
        "Story:\n"
    )

    outputs = pipe(
        prompt,
        max_new_tokens=120,    # 稍微放宽一点
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

    full_text = outputs[0]["generated_text"]
    story_text = full_text[len(prompt):]  # 去掉 prompt

    # 严格控制在 50–100 词之间
    words = story_text.split()
    # 先截断到 100 词
    if len(words) > 100:
        words = words[:100]
    story_text = " ".join(words)

    return story_text

# text2audio
def text2audio(story_text):
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = pipe(story_text)
    return audio_data


def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


        #Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(scenario)

        #Stage 2: Text to Story
        st.text('Generating a story...')
        story = text2story(scenario)
        st.write(story)

        #Stage 3: Story to Audio data
        st.text('Generating audio data...')
        audio_data =text2audio(story)

        # Play button
        if st.button("Play Audio"):
            # Get the audio array and sample rate
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]

            # Play audio directly using Streamlit
            st.audio(audio_array,
                     sample_rate=sample_rate)


if __name__ == "__main__":
    main()
