from llama_cpp import Llama
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from tavily_agentv2 import verify_article

llm = Llama(
    model_path="../MSc_Project/emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1, 
    verbose=False
)


def prompt_vreg(text: str) -> str:
    """Sentiment strength — Vreg — best retrieval signal per paper"""
    return (
        f"Human:\nTask: Evaluate the valence intensity of the writer's mental state "
        f"based on the text, assigning it a real-valued score from 0 (most negative) "
        f"to 1 (most positive).\nText: {text}\nIntensity Score:\n\nA:\n>>"
    )

def prompt_voc(text: str) -> str:
    """Sentiment classification — Voc"""
    return (
        f"Human:\nTask: Categorize the text into an ordinal class that best characterizes "
        f"the writer's mental state, considering various degrees of positive and negative "
        f"sentiment intensity. 3: very positive mental state can be inferred. "
        f"2: moderately positive mental state can be inferred. "
        f"1: slightly positive mental state can be inferred. "
        f"0: neutral or mixed mental state can be inferred. "
        f"-1: slightly negative mental state can be inferred. "
        f"-2: moderately negative mental state can be inferred. "
        f"-3: very negative mental state can be inferred\n"
        f"Text: {text}\nIntensity Class:\n\nA:\n>>"
    )

def prompt_eireg(text: str, emotion: str) -> str:
    """Emotion intensity — EIreg — one call per emotion (anger/fear/joy/sadness)"""
    return (
        f"Human:\nTask: Assign a numerical value between 0 (least E) and 1 (most E) "
        f"to represent the intensity of emotion E expressed in the text.\n"
        f"Text: {text}\nEmotion: {emotion}\nIntensity Score:\n\nA:\n>>"
    )

def prompt_ec(text: str) -> str:
    """Emotion detection — Ec"""
    return (
        f"Human:\nTask: Categorize the text's emotional tone as either "
        f"'neutral or no emotion' or identify the presence of one or more of the given "
        f"emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, "
        f"sadness, surprise, trust).\nText: {text}\nThis text contains emotions:\n\nA:\n>>"
    )


def generate(prompt: str, max_tokens: int = 20) -> str:
    out = llm(prompt, max_tokens=max_tokens, temperature=0, echo=False)
    return out["choices"][0]["text"].strip()

def parse_float(raw: str, default: float = 0.5) -> float:
    """Extract first float found in output"""
    import re
    matches = re.findall(r"[-+]?\d*\.?\d+", raw)
    if matches:
        return max(0.0, min(1.0, float(matches[0])))
    return default

def parse_voc(raw: str) -> int:
    """Extract integer from -3 to 3"""
    import re
    matches = re.findall(r"-?\d+", raw)
    if matches:
        return max(-3, min(3, int(matches[0])))
    return 0

def parse_ec(raw: str) -> list:
    """Extract emotion labels from output"""
    emotions = ["anger","anticipation","disgust","fear","joy",
                "love","optimism","pessimism","sadness","surprise","trust"]
    return [e for e in emotions if e in raw.lower()]


def get_affective_profile(text: str) -> dict:
    """
    Returns all 5 affective dimensions for one article.
    This is what gets stored in your retrieval database.
    """
    text_short = text[:800]  
    vreg_raw = generate(prompt_vreg(text_short))
    vreg     = parse_float(vreg_raw)
    voc_raw  = generate(prompt_voc(text_short))
    voc      = parse_voc(voc_raw)
    eireg = {}
    for emotion in ["anger", "fear", "joy", "sadness"]:
        raw = generate(prompt_eireg(text_short, emotion))
        eireg[emotion] = parse_float(raw)
    ec_raw = generate(prompt_ec(text_short), max_tokens=30)
    ec     = parse_ec(ec_raw)

    return {
        "vreg":         vreg,           
        "voc":          voc,            
        "eireg_anger":  eireg["anger"], 
        "eireg_fear":   eireg["fear"],
        "eireg_joy":    eireg["joy"],
        "eireg_sadness":eireg["sadness"],
        "ec":           ec,             
    }


inpt = """

Trump Just Made A Campaign Promise So Ridiculous It Makes ‘Read My Lips’ Look Good ‹ Opposition Report. 

Posted by Frank Wilkenmeyer on 19 Sep 2016

We’ve all seen campaign promises go horribly wrong. One of the most famous is probably “read my lips…no new taxes,” a moment in time that won an election and lost the next after George HW Bush made the promise but signed the largest tax increase in the history of tax increases. What Donald Trump just promised as a reaction to the attacks over the weekend in Chelsea and Minnesota may have that beat:

.@JasonMillerinDC: Statement on Last Weekend’s Terror Attacks https://t.co/DZlIvkcHmB pic.twitter.com/l3EVTvLcfO — Official Team Trump (@TeamTrump) September 19, 2016

Take a look at that and consider what it says. The personal attacks on President Obama and Secretary Clinton aside, Trump just promised to destroy the existential threat of terrorism in America. He didn’t promise to increase funding to counter-terrorism units or deliver a plan to fight internet terrorism recruitment, he promised to destroy all terrorists and end terrorism on our soil.

Let’s imagine for a moment that Trump actually tried to implement this plan of his. “Destroy” doesn’t leave much to the imagination. While the mouth-breathers who support him are probably giving this statement a hearty “yeeeeehaw,” the reality is that he’s talking about something far more terrifying than the low rate of terrorism we actually witness in America. He’s talking about profiling Muslims. He’s talking about Donald Trump’s version of the Patriot Act, where nobody with brown skin is safe.The man being hunted by the FBI, for example, isn’t just a Muslim, he’s an

The man being hunted by the FBI, for example, isn’t just a Muslim, he’s an American citizen with the right to due process. Just because he’s a suspect in a bombing doesn’t mean he’s guilty of setting off bombs. With this blanket Trump statement, however, this man would be in a deck of cards marked “kill” and not on the most wanted list. There’s nothing that would lead one to believe that citizenship, constitutional rights or due process would come into play.

What that means is that under Trump’s dream regime, we could expect Muslim heavy neighborhoods like Dearborn and Queens to become heavily patrolled and segregated and their residents routinely grabbed off the streets never to return. Without a way to differentiate Christian Arabs from Muslims, would they need to wear some kind of identifying mark to make them easier to spot? If you think I’m being overly critical and you doubt that fascism is where an American presidential candidate is leaning, read that statement one more time. It isn’t about the narrative. It isn’t about prevention or education.

It’s about destroying the existential threat of terrorism at all costs. It may sound good to a bunch of knuckle-draggers who love to hate based on skin color and religion, but to those of us who have studied history and understand exactly what that terminology means, Donald Trump just became even more frightening.

The only solace we have is that what he’s promising is nothing but another “read my lips.” Terrorism isn’t something you can simply “destroy” just because you said so. The thought of turning our streets into a Denzel Washington movie might get you a few more votes from the daft but it isn’t going to provide you with the laws you need to actually implement your fascism. Like the rest of Donald Trump’s campaign, in the end this is nothing more than a bloviating blowhard peddling fear to weak-minded, gullible morons.

Featured image from file

"""

print(get_affective_profile(inpt))
print("\n ######################################## \n")
print(verify_article(inpt))

