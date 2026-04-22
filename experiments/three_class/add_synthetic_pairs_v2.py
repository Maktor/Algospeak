"""
experiments/three_class/add_synthetic_pairs_v2.py

Hand-crafted algospeak pairs targeting real-world coded language that was
systematically missing from the LLM-generated training data.

Covers:
  - unalive / sewerslide  (suicide / self-harm)
  - corn                  (porn)
  - seggs                 (sex)
  - SA                    (rape / sexual assault)
  - gr@ss / w33d          (weed)

Run once from project root:
    uv run python experiments/three_class/add_synthetic_pairs_v2.py

After running, re-run the full 3-class pipeline:
    uv run python experiments/three_class/prepare.py
    uv run python poc/src/encoder_prep.py --config experiments/three_class/config.yaml
    uv run python poc/src/train.py --config experiments/three_class/config.yaml --fresh
    uv run python poc/src/inference.py --config experiments/three_class/config.yaml --notes "..."
"""

from pathlib import Path
from collections import Counter
import pandas as pd

PAIRS_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "splits" / "class012_pairs_reclassified.csv"

# Format: (original, algospeak, technique, deny_terms_detected, deny_terms_transformed)
pairs = [

    # ── unalive — first person ────────────────────────────────────────
    ("i want to kill myself so bad right now",
     "i want to unalive so bad right now",
     "substitution", "kill myself", "kill myself"),
    ("genuinely thinking about suicide tonight ngl",
     "genuinely thinking about unaliving tonight ngl",
     "substitution", "suicide", "suicide"),
    ("suicidal thoughts hitting different tonight ngl",
     "unalive thoughts hitting different tonight ngl",
     "substitution", "suicidal thoughts", "suicidal thoughts"),
    ("i keep googling how to kill myself",
     "i keep googling how to unalive myself",
     "substitution", "kill myself", "kill myself"),
    ("been having thoughts about ending my life every single night",
     "been having unalive thoughts every single night",
     "substitution", "ending my life", "ending my life"),
    ("i told my therapist i wanted to kill myself",
     "i told my therapist i wanted to unalive",
     "substitution", "kill myself", "kill myself"),
    ("the antidepressants are making my suicidal thoughts worse",
     "the antidepressants are making my unalive thoughts worse",
     "substitution", "suicidal thoughts", "suicidal thoughts"),
    ("i have a plan to kill myself i just havent done it yet",
     "i have an unalive plan i just havent done it yet",
     "substitution", "kill myself", "kill myself"),
    ("i researched methods to kill myself",
     "i researched unalive methods",
     "substitution", "kill myself", "kill myself"),
    ("suicide is on my mind 24 7 and i cant make it stop",
     "unaliving is on my mind 24 7 and i cant make it stop",
     "substitution", "suicide", "suicide"),
    ("does anyone else get urges to kill yourself when stressed",
     "does anyone else get unalive urges when stressed",
     "substitution", "kill yourself", "kill yourself"),
    ("im exhausted i just keep thinking about dying",
     "im exhausted unalive thoughts are constant",
     "substitution", "dying|suicidal", "dying|suicidal"),
    ("i attempted suicide last year and nobody knew",
     "i had an unalive attempt last year and nobody knew",
     "substitution", "suicide", "suicide"),
    ("tw suicide i almost did it last night",
     "tw unalive i almost unalived last night",
     "substitution", "suicide", "suicide"),
    ("cw self harm i have been cutting again and thinking about ending it",
     "cw unalive mention i have been s/h again and thinking about unaliving",
     "substitution", "self harm|suicide", "self harm|suicide"),
    ("how do people stop suicidal urges i cant do this anymore",
     "how do people stop unalive urges i cant do this anymore",
     "substitution", "suicidal", "suicidal"),
    ("killing myself seems like the only option at this point",
     "unaliving seems like the only option at this point",
     "substitution", "killing myself", "killing myself"),
    ("i wrote a suicide note already",
     "i wrote my unalive note already",
     "substitution", "suicide note", "suicide note"),
    ("i want to kill myself please someone talk to me",
     "i want to unalive please someone talk to me",
     "substitution", "kill myself", "kill myself"),
    ("i dont want to die i just want the pain to stop but killing myself feels like the only way",
     "i dont want to die i just want the pain to stop but unaliving feels like the only way",
     "substitution", "killing myself|die", "killing myself|die"),

    # ── unalive — third person ────────────────────────────────────────
    ("my best friend told me she wants to kill herself",
     "my best friend told me she wants to unalive herself",
     "substitution", "kill herself", "kill herself"),
    ("my sister attempted suicide twice this year",
     "my sister attempted to unalive twice this year",
     "substitution", "suicide", "suicide"),
    ("he posted about wanting to die right before it happened",
     "he posted about unaliving right before it happened",
     "substitution", "die", "die"),
    ("she left a note saying she was going to kill herself",
     "she left a note saying she was going to unalive herself",
     "substitution", "kill herself", "kill herself"),
    ("three of my friends tried to kill themselves this semester",
     "three of my friends tried to unalive this semester",
     "substitution", "kill themselves", "kill themselves"),
    ("she sent me a text saying she wanted to die then went offline",
     "she sent me a text saying she wanted to unalive then went offline",
     "substitution", "die", "die"),
    ("nobody believed him when he said he was suicidal",
     "nobody believed him when he had unalive thoughts",
     "substitution", "suicidal", "suicidal"),
    ("he threatened to kill himself if she broke up with him",
     "he threatened to unalive if she broke up with him",
     "substitution", "kill himself", "kill himself"),
    ("she talked about suicide for months before anyone took her seriously",
     "she talked about unaliving for months before anyone took her seriously",
     "substitution", "suicide", "suicide"),
    ("my friend is in the hospital after a suicide attempt",
     "my friend is in the hospital after an unalive attempt",
     "substitution", "suicide", "suicide"),
    ("checking in on everyone thinking about killing themselves today",
     "checking in on everyone with unalive thoughts today",
     "substitution", "killing themselves", "killing themselves"),
    ("the suicide rate for trans teens is heartbreaking",
     "the unalive rate for trans teens is heartbreaking",
     "substitution", "suicide", "suicide"),
    ("depression makes everything feel like dying is the answer",
     "depression makes everything feel like unaliving is the answer",
     "substitution", "dying|suicide", "dying|suicide"),
    ("please reach out if you are thinking about killing yourself",
     "please reach out if you are having unalive thoughts",
     "substitution", "killing yourself", "killing yourself"),

    # ── sewerslide (phonetic variant of suicide) ──────────────────────
    ("thinking about suicide tonight",
     "thinking about sewersliding tonight",
     "phonetic", "suicide", "suicide"),
    ("tw suicide please be careful scrolling",
     "sewerslide tw please be careful scrolling",
     "phonetic", "suicide", "suicide"),
    ("i almost attempted suicide last month",
     "i almost sewerslided last month",
     "phonetic", "suicide attempt", "suicide attempt"),
    ("suicidal thoughts wont leave me alone",
     "sewerslide thoughts wont leave me alone",
     "phonetic", "suicidal thoughts", "suicidal thoughts"),
    ("she killed herself last spring and i miss her every day",
     "she sewerslided last spring and i miss her every day",
     "phonetic", "killed herself", "killed herself"),
    ("i keep thinking about suicide when things get really bad",
     "i keep thinking about sewersliding when things get really bad",
     "phonetic", "suicide", "suicide"),
    ("suicide is always in the back of my head",
     "sewerslide is always in the back of my head",
     "phonetic", "suicide", "suicide"),
    ("i researched suicide methods last night and im scared of myself",
     "i researched sewerslide methods last night and im scared of myself",
     "phonetic", "suicide methods", "suicide methods"),

    # ── corn (porn) ───────────────────────────────────────────────────
    ("my boyfriend watches porn all day instead of spending time with me",
     "my boyfriend watches corn all day instead of spending time with me",
     "substitution", "porn", "porn"),
    ("teens are being exposed to porn at younger and younger ages",
     "teens are being exposed to corn at younger and younger ages",
     "substitution", "porn", "porn"),
    ("i am addicted to porn and i dont know how to stop",
     "i am addicted to corn and i dont know how to stop",
     "substitution", "porn", "porn"),
    ("she makes porn for a living and makes six figures",
     "she makes corn for a living and makes six figures",
     "substitution", "porn", "porn"),
    ("the porn industry exploits people who have no other options",
     "the corn industry exploits people who have no other options",
     "substitution", "porn", "porn"),
    ("he sent me porn without asking and i reported him",
     "he sent me corn without asking and i reported him",
     "substitution", "porn", "porn"),
    ("porn addiction is a real thing and it ruins relationships",
     "corn addiction is a real thing and it ruins relationships",
     "substitution", "porn", "porn"),
    ("found porn on his phone and it was pretty extreme stuff",
     "found corn on his phone and it was pretty extreme stuff",
     "substitution", "porn", "porn"),
    ("she did porn back in college and people still use it against her",
     "she did corn back in college and people still use it against her",
     "substitution", "porn", "porn"),
    ("he makes porn with his girlfriend and posts it online",
     "he makes corn with his girlfriend and posts it online",
     "substitution", "porn", "porn"),
    ("what free porn sites are still up after the crackdowns",
     "what free corn sites are still up after the crackdowns",
     "substitution", "porn", "porn"),
    ("porn is literally everywhere online even when you dont search for it",
     "corn is literally everywhere online even when you dont search for it",
     "substitution", "porn", "porn"),
    ("she was tricked into making porn and cant get it taken down",
     "she was tricked into making corn and cant get it taken down",
     "substitution", "porn", "porn"),
    ("revenge porn is illegal in most states now",
     "revenge corn is illegal in most states now",
     "substitution", "revenge porn", "revenge porn"),
    ("my ex posted porn of me online without my consent",
     "my ex posted corn of me online without my consent",
     "substitution", "porn", "porn"),

    # ── seggs (sex) ───────────────────────────────────────────────────
    ("we finally had sex after dating for a year",
     "we finally had seggs after dating for a year",
     "substitution", "sex", "sex"),
    ("she had sex for the first time last weekend",
     "she had seggs for the first time last weekend",
     "substitution", "sex", "sex"),
    ("having sex outside of marriage is a sin according to my parents",
     "having seggs outside of marriage is a sin according to my parents",
     "substitution", "sex", "sex"),
    ("he pressured her into having sex when she wasnt ready",
     "he pressured her into having seggs when she wasnt ready",
     "substitution", "sex", "sex"),
    ("we had sex on the first date and it was amazing",
     "we had seggs on the first date and it was amazing",
     "substitution", "sex", "sex"),
    ("she wants to have sex but i dont feel comfortable yet",
     "she wants to have seggs but i dont feel comfortable yet",
     "substitution", "sex", "sex"),
    ("they were having sex in the parking lot and someone called the police",
     "they were having seggs in the parking lot and someone called the police",
     "substitution", "sex", "sex"),
    ("i had sex with someone i met at a party and now i regret it",
     "i had seggs with someone i met at a party and now i regret it",
     "substitution", "sex", "sex"),
    ("talking about sex education in schools should be mandatory",
     "talking about seggs education in schools should be mandatory",
     "substitution", "sex", "sex"),
    ("she is having sex with her professor which is so wrong",
     "she is having seggs with her professor which is so wrong",
     "substitution", "sex", "sex"),
    ("he asked for sex in exchange for the job and she reported him",
     "he asked for seggs in exchange for the job and she reported him",
     "substitution", "sex", "sex"),
    ("they had sex in the school bathroom and got expelled",
     "they had seggs in the school bathroom and got expelled",
     "substitution", "sex", "sex"),

    # ── SA (rape / sexual assault) ────────────────────────────────────
    ("she was raped at a party and nobody believed her",
     "she was SAd at a party and nobody believed her",
     "abbreviation", "raped", "raped"),
    ("he raped his coworker and got promoted instead of fired",
     "he SAd his coworker and got promoted instead of fired",
     "abbreviation", "raped", "raped"),
    ("i was raped when i was sixteen and never told anyone",
     "i was SAd when i was sixteen and never told anyone",
     "abbreviation", "raped", "raped"),
    ("rape culture on college campuses needs to be taken seriously",
     "SA culture on college campuses needs to be taken seriously",
     "abbreviation", "rape", "rape"),
    ("she pressed charges after being raped by her ex",
     "she pressed charges after being SAd by her ex",
     "abbreviation", "raped", "raped"),
    ("he was accused of rape but the case got dismissed",
     "he was accused of SA but the case got dismissed",
     "abbreviation", "rape", "rape"),
    ("i was raped by someone i trusted and it destroyed me",
     "i was SAd by someone i trusted and it destroyed me",
     "abbreviation", "raped", "raped"),
    ("most rape cases never get reported because victims are blamed",
     "most SA cases never get reported because victims are blamed",
     "abbreviation", "rape", "rape"),
    ("she was raped and got pregnant and they want to force her to keep it",
     "she was SAd and got pregnant and they want to force her to keep it",
     "abbreviation", "raped", "raped"),
    ("tw rape he assaulted me at a concert and walked away free",
     "tw SA he SAd me at a concert and walked away free",
     "abbreviation", "rape", "rape"),

    # ── gr@ss / w33d (weed) ───────────────────────────────────────────
    ("where can i find a good weed dealer around here",
     "where can i find a good gr@ss plug around here",
     "unknown_spelling", "weed", "weed"),
    ("i smoke weed every day to deal with my anxiety",
     "i smoke gr@ss every day to deal with my anxiety",
     "unknown_spelling", "weed", "weed"),
    ("weed should be legal everywhere it is literally medicine",
     "w33d should be legal everywhere it is literally medicine",
     "unknown_spelling", "weed", "weed"),
    ("he got expelled for selling weed at school",
     "he got expelled for selling gr@ss at school",
     "unknown_spelling", "weed", "weed"),
    ("i bought weed from someone i didnt know and it was laced",
     "i bought w33d from someone i didnt know and it was laced",
     "unknown_spelling", "weed", "weed"),
    ("she smokes weed before every class and still gets straight As",
     "she smokes gr@ss before every class and still gets straight As",
     "unknown_spelling", "weed", "weed"),
    ("my parents found my weed stash and kicked me out",
     "my parents found my w33d stash and kicked me out",
     "unknown_spelling", "weed", "weed"),
    ("weed dealers in this city are all selling garbage lately",
     "gr@ss dealers in this city are all selling garbage lately",
     "unknown_spelling", "weed", "weed"),
    ("i quit weed after smoking every day for five years",
     "i quit w33d after smoking every day for five years",
     "unknown_spelling", "weed", "weed"),
    ("he sells weed out of his dorm room like its nothing",
     "he sells gr@ss out of his dorm room like its nothing",
     "unknown_spelling", "weed", "weed"),
    ("he is addicted to cocaine and lost everything",
     "he is addicted to nose candy and lost everything",
     "substitution", "cocaine", "cocaine"),
    ("selling drugs online is way too easy now",
     "selling gr@ss and other stuff online is way too easy now",
     "unknown_spelling", "drugs", "drugs"),
]


def main():
    existing = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")
    print(f"Before: {len(existing)} rows")

    existing_texts = set(existing["text"].astype(str).str.strip())
    new_rows = []
    skipped = 0
    for orig, algo, tech, detected, transformed in pairs:
        if orig.strip() in existing_texts:
            skipped += 1
            continue
        new_rows.append({
            "text":                    orig,
            "classification":          2,
            "original_classification": 2,
            "algospeak_text":          algo,
            "techniques":              tech,
            "deny_terms_detected":     detected,
            "deny_terms_transformed":  transformed,
        })

    if not new_rows:
        print("Nothing to add — all pairs already present.")
        return

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(PAIRS_CSV, index=False, encoding="utf-8-sig")
    print(f"After:  {len(combined)} rows (+{len(new_rows)}, skipped {skipped} duplicates)")

    term_counts = Counter()
    for r in new_rows:
        term_counts.update(str(r["deny_terms_transformed"]).split("|"))
    print("\nNew pairs by deny term:")
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        print(f"  {term:<30} {count}")


if __name__ == "__main__":
    main()
