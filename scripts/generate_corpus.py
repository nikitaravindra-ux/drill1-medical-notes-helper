"""Generate corpus/generated.txt: TEMPLATE-BASED synthetic administrative text (seeded, deterministic).

This part of the corpus is deliberately labelled `generated` because slot-filled templates are far more predictable
than real prose; a language model trained on them reaches a misleadingly low perplexity. The evaluation therefore
reports perplexity on hand-written and generated validation text separately.

All names, clinics and dates are invented. No real person or record is involved.
Usage:  python scripts/generate_corpus.py
"""
import random
from pathlib import Path

SEED = 2024
N_DOCS = 520

HEADER = """# SYNTHETIC, TEMPLATE-GENERATED TEXT (scripts/generate_corpus.py, seed {seed}).
# Every line was produced by filling slots in sentence templates with invented names, clinics and dates.
# It contains no real patient information and was not derived from any real record or public dataset.
# Administrative wording only. One line is one document.
"""

SURNAMES = ["Ashdown", "Brightwater", "Calloway", "Draycott", "Ellsworth", "Fenwick", "Gallagher-Moore", "Hargreave",
            "Isherwood", "Jessop", "Kettleby", "Lindqvist", "Marchetti", "Nightingale-Reid", "Okafor-Lund", "Pemberton",
            "Quillfeather", "Rosenthal", "Sandoval", "Thornbury", "Underhill", "Vickery", "Whitlock", "Yarrow",
            "Abernathy", "Tanaka-Holm", "Valdecasas", "Penhallow"]
CLINICS = ["Harbourview Family Practice", "Eastgate Surgery", "Riverside Community Clinic", "Northfield Health Centre",
           "Lakeshore Medical Centre", "Westbank Centre", "Millbrook Group Practice", "Oakridge Outpatient Unit"]
DEPTS = ["physiotherapy", "radiology", "outpatient", "audiology", "dietetics", "occupational therapy", "podiatry",
         "community nursing", "imaging", "speech therapy"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
TIMES = ["8:30 a.m.", "9:00 a.m.", "9:45 a.m.", "10:15 a.m.", "11:30 a.m.", "1:00 p.m.", "2:15 p.m.", "3:30 p.m.",
         "4:45 p.m."]
SPANS = ["two weeks", "three weeks", "four weeks", "six weeks", "eight weeks", "three months", "six months"]
WORKDAYS = ["two", "three", "five", "seven", "ten", "fourteen"]
CHANNELS = ["by telephone", "by text message", "by email", "by letter", "through the online portal"]
TITLES = ["Mr.", "Ms.", "Mrs.", "Dr."]


def pick(r, xs):
    return r.choice(xs)


def person(r, titles=None):
    return f"{pick(r, titles or TITLES)} {pick(r, SURNAMES)}"


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def when(r):
    return pick(r, [
        lambda: f"{pick(r, DAYS)} {r.randint(1, 28)} {pick(r, MONTHS)} at {pick(r, TIMES)}",
        lambda: f"{pick(r, DAYS)} at {pick(r, TIMES)}",
        lambda: f"{pick(r, ['next', 'this'])} {pick(r, DAYS)} morning",
        lambda: f"the {ordinal(r.randint(2, 28))} of {pick(r, MONTHS)}",
    ])()


# Each generator returns one sentence. Several shapes per topic, with optional clauses.
def appointment(r):
    return pick(r, [
        lambda: f"Your appointment with {person(r)} is on {when(r)}.",
        lambda: f"We can offer you {when(r)}, or {when(r)} if that suits you better.",
        lambda: f"{person(r)} will see you at {pick(r, CLINICS)} on {when(r)}{pick(r, ['', ', so please arrive early', ', and the visit should last about half an hour'])}.",
        lambda: f"The {pick(r, DEPTS)} team has a free slot on {when(r)}; would you like us to hold it for you?",
        lambda: f"Please arrive {pick(r, ['ten', 'fifteen', 'twenty'])} minutes before your time so that reception can check your details.",
        lambda: f"If you cannot make {when(r)}, please let us know {pick(r, ['a day', 'two days', 'a week'])} in advance.",
        lambda: f"Your booking has been moved from {when(r)} to {when(r)} because {pick(r, ['the clinician is away', 'the room is being used for training', 'the building is closed that day'])}.",
    ])()


def reminder(r):
    return pick(r, [
        lambda: f"This is a reminder from {pick(r, CLINICS)} about your visit on {when(r)}.",
        lambda: f"We will send you a reminder {pick(r, CHANNELS)} {pick(r, ['the day before', 'two days before', 'a week before'])}.",
        lambda: f"Reply {pick(r, ['YES', 'CONFIRM'])} to keep this appointment or {pick(r, ['NO', 'CANCEL'])} to release it.",
        lambda: f"We tried to reach you {pick(r, CHANNELS)} today but could not get through; please contact reception.",
    ])()


def referral(r):
    return pick(r, [
        lambda: f"Dear {person(r, ['Dr.', 'Mr.', 'Ms.'])}, I am writing to refer the above patient to the {pick(r, DEPTS)} service.",
        lambda: f"Thank you for agreeing to see the patient; the referral form and a list of previous visits are enclosed.",
        lambda: f"This referral is {pick(r, ['routine', 'priority', 'urgent'])}, and we would appreciate a reply within {pick(r, WORKDAYS)} working days.",
        lambda: f"Please copy your correspondence to the patient and to {person(r)} at {pick(r, CLINICS)}.",
        lambda: f"The patient has asked for {pick(r, ['an afternoon', 'a morning', 'a ground floor', 'an interpreter-assisted'])} appointment where possible.",
        lambda: f"We have received your referral and added it to the {pick(r, DEPTS)} waiting list.",
        lambda: f"Kind regards, {person(r)}, {pick(r, ['Practice Manager', 'Clinic Coordinator', 'Administrator'])}, {pick(r, CLINICS)}.",
    ])()


def followup(r):
    return pick(r, [
        lambda: f"A follow-up visit has been arranged for {pick(r, SPANS)} from today.",
        lambda: f"There is no need to come back unless we write to you; a questionnaire will arrive {pick(r, CHANNELS)} in {pick(r, SPANS)}.",
        lambda: f"A member of the {pick(r, DEPTS)} team will call you within {pick(r, WORKDAYS)} working days to arrange the next step.",
        lambda: f"Please book your next visit at the front desk before you leave, or ring us during the week.",
        lambda: f"Your next review is due in {pick(r, SPANS)}, and we will remind you {pick(r, CHANNELS)}.",
    ])()


def insurance(r):
    return pick(r, [
        lambda: f"We have asked your insurer to authorise the visit planned for {when(r)}.",
        lambda: f"The insurer needs {pick(r, WORKDAYS)} working days to reply, and we will inform you as soon as we hear.",
        lambda: f"Authorisation has been granted for {pick(r, ['one visit', 'two visits', 'three visits'])} within {pick(r, SPANS)}.",
        lambda: f"Please bring your insurance card to the appointment so that the front desk can copy it.",
        lambda: f"The billing office can explain what your policy covers and what you may be asked to pay.",
        lambda: f"The claim form must be signed and returned within {pick(r, ['fourteen', 'thirty', 'sixty'])} days.",
    ])()


def intake(r):
    return pick(r, [
        lambda: f"Please complete the registration form and hand it to reception at {pick(r, CLINICS)}.",
        lambda: f"Tell us if your address, telephone number or preferred language has changed since your last visit.",
        lambda: f"You will need photo identification and proof of address to register as a new patient.",
        lambda: f"Please name someone we can contact in an emergency, and tell us how they are related to you.",
        lambda: f"Children under sixteen must come with a parent or guardian, who will sign the form on their behalf.",
    ])()


def discharge(r):
    return pick(r, [
        lambda: f"Transport home has been booked for {pick(r, TIMES)}, and the driver will call when they arrive.",
        lambda: f"Please collect your belongings and your discharge letter from the ward desk before you leave.",
        lambda: f"The ward will telephone you {pick(r, ['the day after', 'two days after', 'a few days after'])} you go home.",
        lambda: f"A copy of your discharge letter will be sent to your general practitioner within {pick(r, WORKDAYS)} working days.",
        lambda: f"A district nurse will visit your home on {pick(r, DAYS)} and will phone first to agree a time.",
    ])()


def consent(r):
    return pick(r, [
        lambda: f"To request a copy of your records, complete the form and return it to the records office with proof of identity.",
        lambda: f"We cannot share your information with a third party without your signed consent.",
        lambda: f"Your request has been logged, and we will reply within {pick(r, ['ten', 'twenty', 'thirty'])} days.",
        lambda: f"You may change your consent choices at any time by telling reception in writing.",
        lambda: f"A parent or guardian must sign the consent section for patients under sixteen.",
    ])()


def office(r):
    return pick(r, [
        lambda: f"Reception note: caller asked to move {pick(r, ['an appointment', 'a visit', 'a booking'])} from {pick(r, DAYS)} to {pick(r, DAYS)}; diary updated.",
        lambda: f"Staff reminder: please check the patient's name and the date of the appointment before handing out any paperwork.",
        lambda: f"Please file signed forms in the {pick(r, ['blue', 'red', 'green'])} folder at the end of each day.",
        lambda: f"Handover: {pick(r, ['two', 'three', 'four'])} patients are waiting for a call back, and the {pick(r, ['printer', 'scanner', 'check-in screen'])} needs attention.",
        lambda: f"{person(r)} will update the booking template, and a progress report is due on {pick(r, DAYS)}.",
    ])()


TOPICS = [appointment, reminder, referral, followup, insurance, intake, discharge, consent, office]
OPENERS = ["", "", "", "Good morning. ", "Hello. ", "Thank you for calling. ", "Dear patient, ", "To whom it may concern: "]
CLOSERS = ["", "", " Thank you.", " We apologise for any inconvenience.", " Please call if you have any questions.",
           " If anything changes, tell reception."]


def make_doc(r):
    main = pick(r, TOPICS)
    n = r.randint(2, 5)
    topics = [main] + [pick(r, TOPICS) if r.random() < 0.25 else main for _ in range(n - 1)]
    sents = [t(r) for t in topics]
    text = " ".join(sents)
    return (pick(r, OPENERS) + text + pick(r, CLOSERS)).replace("..", ".").strip()


def main():
    r = random.Random(SEED)
    docs = [make_doc(r) for _ in range(N_DOCS)]
    out = Path(__file__).resolve().parent.parent / "corpus" / "generated.txt"
    out.write_text(HEADER.format(seed=SEED) + "\n" + "\n".join(docs) + "\n", encoding="utf-8")
    print(f"wrote {len(docs)} documents to {out}")


if __name__ == "__main__":
    main()
