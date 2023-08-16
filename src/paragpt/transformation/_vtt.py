from typing import List
import re


def brush_up_vtt(text: str) -> str:
    text = text.replace("WEBVTT", "")
    text = re.sub("^\d+\n", "", text, flags=re.M)   # block numbers
    text = re.sub(".+-.+-.+", "", text, flags=re.M)
    text = re.sub("\d+:\d+:\d.+?\n", "", text, flags=re.M)  # timestamps
    text = text.strip("\n")
    text = re.sub("\n+", "\n", text, flags=re.M)
    return text


def split_speaker_text(text: str) -> List[str]:
    if ": " in text:
        return text.split(": ")
    return ["", text]

def clean_teams_vtt(text: str) -> List[str]:
    text = brush_up_vtt(text)
    lines = text.split("\n")
    speaker_text = list(map(split_speaker_text, lines))
    conversation = []
    previous_speaker = None

    for line in speaker_text:
        speaker, text = line
        if speaker == "" or previous_speaker == speaker:
            conversation[-1] += " %s" % text
        else:
            conversation.append(f"{speaker}: {text}")

        previous_speaker = speaker
    print("Cleaned Conversation:\n", conversation)
    return conversation
