"""
Rider-Waite-Smith tarot deck definition.

Each card has:
  - slug: stable filename id (matches assets/tarot/{slug}.jpg)
  - name: display name
  - arcana: "major" | "minor"
  - suit: None for major, else one of wands/cups/swords/pentacles
  - number: 0..21 for major, 1..14 for minor (11=Page, 12=Knight, 13=Queen, 14=King)
  - keywords_up / keywords_rev: short comma-list for static fallback readings
  - meaning: one-sentence essence (LLM context, not user-facing on its own)

Image filenames follow the Wikimedia Commons RWS naming convention so the
download script can fetch them without a per-card URL table.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Card:
    slug: str
    name: str
    arcana: str
    suit: Optional[str]
    number: int
    keywords_up: str
    keywords_rev: str
    meaning: str
    wiki_filename: str  # e.g. "RWS_Tarot_00_Fool.jpg" or "Wands01.jpg"


# Major Arcana (0-21)
_MAJOR = [
    ("the-fool", "The Fool", 0, "RWS_Tarot_00_Fool.jpg",
     "new beginnings, leap of faith, innocence, spontaneity",
     "recklessness, hesitation, naivety",
     "A leap into the unknown — beginnings unburdened by what came before."),
    ("the-magician", "The Magician", 1, "RWS_Tarot_01_Magician.jpg",
     "manifestation, willpower, skill, focus",
     "manipulation, untapped talent, illusion",
     "The will directed — having the tools and the focus to make a thing real."),
    ("the-high-priestess", "The High Priestess", 2, "RWS_Tarot_02_High_Priestess.jpg",
     "intuition, mystery, the subconscious, hidden knowledge",
     "secrets, repression, disconnection from intuition",
     "What is known beneath words — the inner voice you should not override."),
    ("the-empress", "The Empress", 3, "RWS_Tarot_03_Empress.jpg",
     "abundance, fertility, nurture, sensuality, creation",
     "stagnation, dependence, smothering",
     "Generative life force — bodies, gardens, art that wants to be made."),
    ("the-emperor", "The Emperor", 4, "RWS_Tarot_04_Emperor.jpg",
     "structure, authority, stability, discipline",
     "domination, rigidity, loss of control",
     "Order and the building of lasting structures; the fatherly hand."),
    ("the-hierophant", "The Hierophant", 5, "RWS_Tarot_05_Hierophant.jpg",
     "tradition, institutions, mentorship, conformity",
     "rebellion, unconventionality, breaking tradition",
     "Inherited wisdom — the path others have walked and named."),
    ("the-lovers", "The Lovers", 6, "RWS_Tarot_06_Lovers.jpg",
     "love, union, choice, alignment of values",
     "disharmony, misalignment, broken commitments",
     "Choice with the whole self — what you choose to bind your life to."),
    ("the-chariot", "The Chariot", 7, "RWS_Tarot_07_Chariot.jpg",
     "willpower, victory, control, momentum",
     "loss of direction, scattered effort, being driven by impulse",
     "Forward motion through harnessed opposites — drive shaped by control."),
    ("strength", "Strength", 8, "RWS_Tarot_08_Strength.jpg",
     "courage, gentle power, patience, inner force",
     "self-doubt, weakness, raw passion uncontained",
     "The lion calmed not by force but by presence — quiet, unshakeable courage."),
    ("the-hermit", "The Hermit", 9, "RWS_Tarot_09_Hermit.jpg",
     "introspection, solitude, inner guidance, withdrawal",
     "isolation, loneliness, refusal to seek",
     "The lantern carried inward — wisdom found by stepping away."),
    ("wheel-of-fortune", "Wheel of Fortune", 10, "RWS_Tarot_10_Wheel_of_Fortune.jpg",
     "cycles, fate, turning points, luck",
     "bad luck, resistance to change, breaking cycles",
     "What turns turns — the wheel's motion, beyond personal will."),
    ("justice", "Justice", 11, "RWS_Tarot_11_Justice.jpg",
     "fairness, truth, cause and effect, accountability",
     "injustice, dishonesty, evading consequences",
     "The scales — clear-eyed reckoning of what is owed."),
    ("the-hanged-man", "The Hanged Man", 12, "RWS_Tarot_12_Hanged_Man.jpg",
     "surrender, new perspective, suspension, sacrifice",
     "stalling, martyrdom, indecision",
     "Voluntary stillness — seeing the world rightly by hanging upside down."),
    ("death", "Death", 13, "RWS_Tarot_13_Death.jpg",
     "endings, transformation, transition, letting go",
     "resistance to change, stagnation, fear of endings",
     "The end that makes a beginning possible — what dies is what was finished."),
    ("temperance", "Temperance", 14, "RWS_Tarot_14_Temperance.jpg",
     "balance, blending, patience, moderation",
     "imbalance, excess, impatience",
     "The middle path; mixing opposites until something new pours out."),
    ("the-devil", "The Devil", 15, "RWS_Tarot_15_Devil.jpg",
     "attachment, shadow, addiction, materialism, the chain you can step out of",
     "release, awareness, breaking free, reclaiming power",
     "The chains you wear loosely — what binds you because you let it."),
    ("the-tower", "The Tower", 16, "RWS_Tarot_16_Tower.jpg",
     "sudden upheaval, revelation, collapse of false structures",
     "averted disaster, fear of change, narrow escape",
     "The lightning strike — what was built on lies cannot stand."),
    ("the-star", "The Star", 17, "RWS_Tarot_17_Star.jpg",
     "hope, renewal, faith, calm after storm",
     "despair, disconnection from hope, lost faith",
     "After the Tower, the still water and the open sky — quiet hope returning."),
    ("the-moon", "The Moon", 18, "RWS_Tarot_18_Moon.jpg",
     "illusion, intuition, the unconscious, hidden anxieties",
     "confusion lifting, secrets revealed, clarity",
     "The dream-light path — what feels true at night and may not be."),
    ("the-sun", "The Sun", 19, "RWS_Tarot_19_Sun.jpg",
     "joy, vitality, success, clarity",
     "diminished joy, blocked optimism, temporary cloud",
     "Unambiguous warmth — the world clearly seen and clearly enjoyed."),
    ("judgement", "Judgement", 20, "RWS_Tarot_20_Judgement.jpg",
     "awakening, reckoning, calling, rebirth",
     "self-doubt, refusing the call, harsh self-criticism",
     "The trumpet — being summoned to a larger version of yourself."),
    ("the-world", "The World", 21, "RWS_Tarot_21_World.jpg",
     "completion, integration, fulfillment, wholeness",
     "incompletion, delay, loose ends",
     "The cycle closed — everything in its place, nothing missing."),
]


_SUITS = [
    # (suit_slug, suit_display, wiki_prefix, element, themes)
    ("wands", "Wands", "Wands", "fire", "drive, creativity, passion, action"),
    ("cups", "Cups", "Cups", "water", "emotions, relationships, intuition, art"),
    ("swords", "Swords", "Swords", "air", "thought, conflict, truth, intellect"),
    ("pentacles", "Pentacles", "Pents", "earth", "money, body, work, the material world"),
]

# Per-suit minor arcana keyword tables. Indexed 1..14.
# 1=Ace, 2-10 numerical, 11=Page, 12=Knight, 13=Queen, 14=King.
_MINOR_KEYWORDS: dict[str, dict[int, tuple[str, str, str]]] = {
    "wands": {
        1:  ("inspiration, ignition, creative spark", "delays, missed opportunity, lack of direction",
             "A flame newly lit — the first impulse to act."),
        2:  ("planning, choices, future vision", "fear of unknown, playing it safe",
             "Standing on the edge of a known world, looking out."),
        3:  ("expansion, foresight, ships coming in", "delays, frustration, narrow vision",
             "What you've sent out begins to return."),
        4:  ("celebration, harmony, homecoming", "transition, instability, disconnection",
             "The completed dwelling — joy with people you belong to."),
        5:  ("petty conflict, friction, sparring", "avoiding conflict, finding agreement",
             "Disorganized friction — clashing without true enemies."),
        6:  ("victory, recognition, public success", "fall from grace, private win, ego inflation",
             "The parade — earned acclaim, witnessed."),
        7:  ("defending position, perseverance, holding the line", "overwhelm, giving up, exposed",
             "From the high ground — defending what you've built."),
        8:  ("swift movement, momentum, news arriving", "delays, things scattering, wait extended",
             "Arrows in flight — fast forward motion."),
        9:  ("resilience, last stand, weary readiness", "exhaustion, paranoia, refusing to rest",
             "Wounded but standing — the strength of the nearly-finished."),
        10: ("burden, responsibility taken too far", "release, delegation, putting it down",
             "Carrying more than is yours to carry."),
        11: ("curiosity, free spirit, discovery", "scattered energy, immaturity, false starts",
             "The young messenger of fire — alive with possibility."),
        12: ("adventure, action, charge ahead", "recklessness, impatience, burnout",
             "Riding hard toward what calls — heat and motion."),
        13: ("warm authority, charisma, vibrant focus", "demanding, jealous, overextended",
             "The warmed throne — leadership through passion."),
        14: ("visionary leadership, mastery of action", "tyranny, impulsiveness, hot-headed power",
             "Fire ruled and ruling — a sovereign of action."),
    },
    "cups": {
        1:  ("emotional opening, new love, creativity flowing", "blocked emotion, emptiness, repressed feelings",
             "The cup overflowing — the heart willing to receive."),
        2:  ("partnership, mutual attraction, harmonious bond", "imbalance, broken connection, distrust",
             "Two meeting as equals — the mirror of being chosen back."),
        3:  ("celebration, friendship, community joy", "overindulgence, scattered group, gossip",
             "Toasts among the people who know you."),
        4:  ("apathy, contemplation, missing what's offered", "renewed interest, awareness, acceptance",
             "The cup unseen — gifts ignored while looking inward."),
        5:  ("grief, loss, dwelling on what spilled", "acceptance, recovery, looking at what remains",
             "Three cups spilled, two still standing — grief that overlooks the unbroken."),
        6:  ("nostalgia, innocence, gifts from the past", "stuck in the past, idealizing memory",
             "Childhood scent — the simple sweetness that came before."),
        7:  ("possibilities, illusion, wishful thinking", "clarity, choosing, cutting through fantasy",
             "Visions in the smoke — too many shapes to choose."),
        8:  ("walking away, seeking deeper meaning", "fear of change, staying when you should go",
             "Leaving what was good but no longer enough."),
        9:  ("contentment, wishes granted, satisfaction", "smugness, dissatisfaction beneath comfort",
             "The wish-fulfilled cup — what you asked for, present."),
        10: ("emotional fulfillment, family harmony, blessed life", "broken home, misalignment, false harmony",
             "The rainbow over the home — full-hearted belonging."),
        11: ("artistic openness, gentle messages, dreaminess", "moodiness, immaturity, fragile feelings",
             "The young dreamer — soft eyes, soft heart."),
        12: ("romantic gesture, idealism, following the heart", "moodiness, unreliable, deception",
             "The questing knight of feeling — bearing offerings."),
        13: ("emotional depth, intuition, compassionate strength", "moodiness, codependency, drowning",
             "The deep well — knowing the heart without losing the self."),
        14: ("emotional mastery, calm wisdom, diplomacy", "manipulation, repression, volatile under calm",
             "The ocean steadied — feeling great depth without flooding."),
    },
    "swords": {
        1:  ("breakthrough, mental clarity, truth seen", "confusion, mental block, misuse of force",
             "The sword raised — a clear new thought cutting through."),
        2:  ("stalemate, difficult choice, blindness", "indecision lifting, seeing clearly again",
             "Blindfolded balance — unable or unwilling to look."),
        3:  ("heartbreak, painful truth, grief", "healing, releasing pain, forgiveness",
             "The pierced heart — the ache that means something mattered."),
        4:  ("rest, recovery, withdrawal", "burnout, restlessness, refusing rest",
             "The knight at rest — repair before returning."),
        5:  ("conflict, hollow victory, defeat", "reconciliation, walking away, regret",
             "Winning at a cost not worth paying."),
        6:  ("transition, leaving troubled waters", "unfinished business, stuck, unable to move on",
             "The boat across — leaving for calmer shores."),
        7:  ("deception, strategy, getting away with it", "honesty caught up to, returning what was taken",
             "The thief in the night — clever moves that may not last."),
        8:  ("self-imposed limits, feeling trapped", "freedom, seeing the way out, releasing restriction",
             "Bound but never truly tied — the cage that opens from inside."),
        9:  ("anxiety, nightmares, dread", "anxiety lifting, nightmares ending, hope",
             "The 3am mind — fears worse than what's actually there."),
        10: ("collapse, defeat, the end of a cycle", "recovery, the worst is past, rebirth ahead",
             "Rock bottom — the only direction left is up."),
        11: ("curious mind, sharp speech, vigilance", "gossip, immaturity, mean wit",
             "The watchful young one — words too quick for their own good."),
        12: ("decisive action, ideas in motion", "recklessness, harsh words, impulsive choices",
             "The charging mind — clarity becomes a weapon at speed."),
        13: ("unbiased clarity, honest counsel, truth-teller", "coldness, cruelty, bitterness",
             "Air made law — clarity without sentimentality."),
        14: ("authority through clarity, ethical mind", "tyrannical thought, cold judgment, abuse of intellect",
             "Reasoned rule — the mind sovereign over passion."),
    },
    "pentacles": {
        1:  ("new opportunity, prosperity beginning, manifestation", "missed opportunity, scarcity mindset",
             "The seed coin — material possibility offered."),
        2:  ("juggling, adaptability, balance under flux", "overwhelm, dropped balls, financial strain",
             "Two coins spinning — keeping it all in motion."),
        3:  ("teamwork, craft, collaborative building", "lack of teamwork, mediocrity, disjointed effort",
             "The cathedral built by many hands."),
        4:  ("holding tightly, security, possessiveness", "letting go, generosity, releasing grip",
             "The clenched fist — what you protect by refusing to share."),
        5:  ("hardship, exclusion, material lack", "recovery, finding shelter, end of hardship",
             "Out in the cold — but the lit window is just there."),
        6:  ("generosity, giving and receiving fairly", "strings attached, debt, uneven exchange",
             "The scales of giving — fair flow of resources."),
        7:  ("assessment, patience with growth, harvest pending", "impatience, poor returns, wasted effort",
             "Looking at the vine — has the work paid yet?"),
        8:  ("craftsmanship, dedication to the craft, mastery in progress", "perfectionism, drudgery, careless work",
             "The apprentice's bench — repeated practice deepening."),
        9:  ("self-reliance, abundance enjoyed, refined success", "loneliness in success, materialism, dependence",
             "The walled garden — earned beauty enjoyed alone."),
        10: ("legacy, lasting wealth, generational stability", "family disputes, fleeting wealth, inheritance issues",
             "The completed estate — wealth that becomes inheritance."),
        11: ("practical learning, new venture, grounded curiosity", "lack of focus, procrastination, missed practical chance",
             "The student of the earth — patient with slow gains."),
        12: ("steady effort, reliability, methodical progress", "stubbornness, stagnation, dull labor",
             "The plowman's pace — slow, certain, productive."),
        13: ("nurturing abundance, embodied wisdom, practical care", "smothering, materialism, neglected self-care",
             "The fertile court — wealth as care made tangible."),
        14: ("material mastery, generosity from strength, secure leadership", "greed, corruption, hoarding",
             "Earth made law — provider, builder, steady hand."),
    },
}

# Pip name on the Wikimedia Commons file is "Pents" for pentacles (e.g. Pents01.jpg).
# The display/data slug is "pentacles" though.


def _build_deck() -> list[Card]:
    deck: list[Card] = []
    for slug, name, num, wiki, up, rev, meaning in _MAJOR:
        deck.append(Card(
            slug=slug, name=name, arcana="major", suit=None, number=num,
            keywords_up=up, keywords_rev=rev, meaning=meaning, wiki_filename=wiki,
        ))

    rank_names = {1: "Ace", 11: "Page", 12: "Knight", 13: "Queen", 14: "King"}
    for suit_slug, suit_disp, wiki_prefix, _element, _themes in _SUITS:
        for n in range(1, 15):
            rank = rank_names.get(n, str(n))
            up, rev, meaning = _MINOR_KEYWORDS[suit_slug][n]
            display = f"{rank} of {suit_disp}"
            slug = f"{suit_slug}-{n:02d}"
            deck.append(Card(
                slug=slug,
                name=display,
                arcana="minor",
                suit=suit_slug,
                number=n,
                keywords_up=up,
                keywords_rev=rev,
                meaning=meaning,
                wiki_filename=f"{wiki_prefix}{n:02d}.jpg",
            ))
    return deck


DECK: list[Card] = _build_deck()
DECK_BY_SLUG: dict[str, Card] = {c.slug: c for c in DECK}

assert len(DECK) == 78, f"Deck must be 78 cards, got {len(DECK)}"
