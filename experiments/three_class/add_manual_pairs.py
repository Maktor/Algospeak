"""
experiments/three_class/add_manual_pairs.py

Appends hand-crafted algospeak pairs for real-world terms that were missing
from the synthetic generation (unalive, corn, seggs, SA).

Run once from project root:
    uv run python experiments/three_class/add_manual_pairs.py

After running, re-run the full 3-class pipeline:
    uv run python experiments/three_class/prepare.py
    uv run python poc/src/encoder_prep.py --config experiments/three_class/config.yaml
    uv run python poc/src/train.py --config experiments/three_class/config.yaml --fresh
    uv run python poc/src/inference.py --config experiments/three_class/config.yaml --notes "..."
"""

from pathlib import Path
import pandas as pd

PAIRS_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "splits" / "class012_pairs_reclassified.csv"

pairs = [
    # unalive (suicide / kill myself)
    ('im going to kill myself',                             'im going to unalive myself',                             'substitution', 'kill myself',    'kill myself'),
    ('gonna kill myself lol jk',                            'gonna unalive myself lol jk',                            'substitution', 'kill myself',    'kill myself'),
    ('i think about suicide every single day',              'i think about unaliving every single day',               'substitution', 'suicide',        'suicide'),
    ('i have been having suicidal thoughts lately',         'i have been having unalive thoughts lately',             'substitution', 'suicide',        'suicide'),
    ('she killed herself after years of abuse',             'she unalived herself after years of abuse',              'substitution', 'killed herself', 'killed herself'),
    ('my friend tried to kill himself last week',           'my friend tried to unalive himself last week',           'substitution', 'kill himself',   'kill himself'),
    ('depression makes you want to kill yourself',          'depression makes you want to unalive yourself',          'substitution', 'kill yourself',  'kill yourself'),
    ('i want to kill myself sometimes not gonna lie',       'i want to unalive myself sometimes not gonna lie',       'substitution', 'kill myself',    'kill myself'),
    ('why do so many people kill themselves',               'why do so many people unalive themselves',               'substitution', 'kill themselves','kill themselves'),
    ('the suicide rate among teens is insane',              'the unalive rate among teens is insane',                 'substitution', 'suicide',        'suicide'),
    ('he was suicidal for months before anyone noticed',    'he was unalive-coded for months before anyone noticed',  'substitution', 'suicidal',       'suicidal'),
    ('she talked about suicide before she died',            'she talked about unaliving before she died',             'substitution', 'suicide',        'suicide'),
    ('if you feel like killing yourself please reach out',  'if you feel like unaliving yourself please reach out',   'substitution', 'killing yourself','killing yourself'),
    ('kms i cant deal with this anymore',                   'gonna unalive fr i cant deal with this anymore',         'substitution', 'kms',            'kms'),
    ('honestly just want to end it all',                    'honestly just want to unalive at this point',            'substitution', 'end it all',     'end it all'),
    ('thoughts of suicide wont leave me alone',             'unalive thoughts wont leave me alone',                   'substitution', 'suicide',        'suicide'),
    ('i keep thinking about killing myself',                'i keep thinking about unaliving myself',                 'substitution', 'killing myself', 'killing myself'),
    ('she attempted suicide three times',                   'she attempted to unalive three times',                   'substitution', 'suicide',        'suicide'),
    ('my brother killed himself two years ago',             'my brother unalived himself two years ago',              'substitution', 'killed himself', 'killed himself'),
    ('this is making me suicidal',                          'this is making me want to unalive',                      'substitution', 'suicidal',       'suicidal'),
    # corn (porn)
    ('where do you watch porn',                             'where do you watch corn',                                'substitution', 'porn',           'porn'),
    ('i watch too much porn honestly',                      'i watch too much corn honestly',                         'substitution', 'porn',           'porn'),
    ('she did porn videos back in the day',                 'she did corn videos back in the day',                    'substitution', 'porn',           'porn'),
    ('stop watching porn it rots your brain',               'stop watching corn it rots your brain',                  'substitution', 'porn',           'porn'),
    ('free porn sites are everywhere now',                  'free corn sites are everywhere now',                     'substitution', 'porn',           'porn'),
    # seggs (sex)
    ('we had sex last night',                               'we had seggs last night',                                'substitution', 'sex',            'sex'),
    ('she wants to have sex with me',                       'she wants to have seggs with me',                        'substitution', 'sex',            'sex'),
    ('having sex with multiple partners is risky',          'having seggs with multiple partners is risky',           'substitution', 'sex',            'sex'),
    ('they had sex in the car',                             'they had seggs in the car',                              'substitution', 'sex',            'sex'),
    # SA (rape / sexual assault)
    ('she was raped by her teacher',                        'she was SAd by her teacher',                             'abbreviation', 'raped',          'raped'),
    ('he raped her while she was drunk',                    'he SAd her while she was drunk',                         'abbreviation', 'raped',          'raped'),
    ('rape is not taken seriously enough',                  'SA is not taken seriously enough',                       'abbreviation', 'rape',           'rape'),
    ('she reported being raped on campus',                  'she reported being SAd on campus',                       'abbreviation', 'raped',          'raped'),
]

def main():
    existing = pd.read_csv(PAIRS_CSV, encoding='utf-8-sig')
    print(f'Before: {len(existing)} rows')

    # Skip any pairs already present (idempotent)
    existing_texts = set(existing['text'].astype(str).str.strip())
    new_rows = []
    for orig, algo, tech, detected, transformed in pairs:
        if orig.strip() in existing_texts:
            print(f'  Skipping (already exists): {orig[:60]}')
            continue
        new_rows.append({
            'text':                    orig,
            'classification':          2,
            'original_classification': 2,
            'algospeak_text':          algo,
            'techniques':              tech,
            'deny_terms_detected':     detected,
            'deny_terms_transformed':  transformed,
        })

    if not new_rows:
        print('Nothing to add — all pairs already present.')
        return

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(PAIRS_CSV, index=False, encoding='utf-8-sig')
    print(f'After:  {len(combined)} rows (+{len(new_rows)})')


if __name__ == '__main__':
    main()
