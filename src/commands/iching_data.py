"""
I Ching hexagram dataset — King Wen ordering (1..64).

Each hexagram is identified by its 6-bit `lines` string read BOTTOM-UP:
  - position 0 = bottom line (line 1 in I Ching parlance)
  - position 5 = top line    (line 6)
  - "1" = yang (solid)
  - "0" = yin  (broken)

Trigrams (also bottom-up):
  Heaven   ☰  111
  Earth    ☷  000
  Thunder  ☳  100
  Water    ☵  010
  Mountain ☶  001
  Wind     ☴  011
  Fire     ☲  101
  Lake     ☱  110

Hexagram = upper trigram (lines 4-6) over lower trigram (lines 1-3).
Cosmetically the upper sits on top of the lower in the rendered image.

Texts here are short — name, pinyin, English title, keywords, one-sentence
essence. The LLM does the heavy lifting on top of those primitives plus the
cast result and changing-line numbers, mirroring the tarot data shape.
"""

from __future__ import annotations

from dataclasses import dataclass


# Trigram bit pattern (bottom-up) → (chinese, pinyin, english, glyph)
TRIGRAMS: dict[str, tuple[str, str, str, str]] = {
    "111": ("乾", "Qián",  "Heaven",   "☰"),
    "000": ("坤", "Kūn",   "Earth",    "☷"),
    "100": ("震", "Zhèn",  "Thunder",  "☳"),
    "010": ("坎", "Kǎn",   "Water",    "☵"),
    "001": ("艮", "Gèn",   "Mountain", "☶"),
    "011": ("巽", "Xùn",   "Wind",     "☴"),
    "101": ("離", "Lí",    "Fire",     "☲"),
    "110": ("兌", "Duì",   "Lake",     "☱"),
}


@dataclass(frozen=True)
class Hexagram:
    number: int          # 1..64 (King Wen)
    lines: str           # 6-char bit string, bottom-up ("1"=yang, "0"=yin)
    chinese: str         # 1-2 char Chinese name
    pinyin: str          # romanization with tone marks
    name: str            # English title (Wilhelm-style)
    keywords: str        # short comma list, e.g. "creative, initiative, force"
    meaning: str         # one-sentence essence

    @property
    def lower_bits(self) -> str:
        """Bottom trigram — lines 1-3, bottom-up."""
        return self.lines[:3]

    @property
    def upper_bits(self) -> str:
        """Top trigram — lines 4-6, bottom-up."""
        return self.lines[3:]

    @property
    def lower_trigram(self) -> tuple[str, str, str, str]:
        return TRIGRAMS[self.lower_bits]

    @property
    def upper_trigram(self) -> tuple[str, str, str, str]:
        return TRIGRAMS[self.upper_bits]

    def trigram_pair_label(self) -> str:
        """Human-readable 'X over Y' (upper element first, as in Wilhelm)."""
        return f"{self.upper_trigram[2]} over {self.lower_trigram[2]}"


# (number, lines, chinese, pinyin, name, keywords, meaning)
# Lines are written BOTTOM-UP. Verified against King Wen ordering.
_RAW: list[tuple[int, str, str, str, str, str, str]] = [
    (1,  "111111", "乾",   "Qián",     "The Creative",
     "creative force, heaven, leadership, initiative",
     "Pure generative force — the right moment to lead, build, and act with sustained intent."),
    (2,  "000000", "坤",   "Kūn",      "The Receptive",
     "yielding, receptive, devotion, support",
     "Pure receptive ground — strength through patience, response, and devoted following."),
    (3,  "100010", "屯",   "Zhūn",     "Difficulty at the Beginning",
     "birth pangs, struggle, perseverance, gathering helpers",
     "The first sprout pushing through hard ground — chaotic beginnings that reward steady help and patience."),
    (4,  "010001", "蒙",   "Méng",     "Youthful Folly",
     "inexperience, learning, the teacher, asking",
     "The student approaches the teacher — wisdom comes from acknowledging what you don't know and asking once."),
    (5,  "111010", "需",   "Xū",       "Waiting",
     "nourishment, patience, timing, faith",
     "Waiting with confidence — the situation is not yet ripe; conserve strength and trust the unfolding."),
    (6,  "010111", "訟",   "Sòng",     "Conflict",
     "dispute, caution, mediation, withdrawal",
     "An argument that cannot be won by pushing — meet halfway, or step back before it escalates."),
    (7,  "010000", "師",   "Shī",      "The Army",
     "discipline, leadership, organized effort, command",
     "Disciplined collective action under a steady commander — moral authority is what makes force righteous."),
    (8,  "000010", "比",   "Bǐ",       "Holding Together",
     "union, alliance, mutual support, belonging",
     "People joining around a clear center — find your circle and commit, or be left outside the bond."),
    (9,  "111011", "小畜", "Xiǎo Chù", "The Taming Power of the Small",
     "small restraint, gentle influence, patience, subtle preparation",
     "A small force restraining a larger one — refine details and wait for the storm to clear before acting."),
    (10, "110111", "履",   "Lǚ",       "Treading",
     "conduct, careful step, treading on the tiger, decorum",
     "Walking on the tiger's tail — proceed with measured courtesy and the danger does not bite."),
    (11, "111000", "泰",   "Tài",      "Peace",
     "harmony, prosperity, communication, flourishing",
     "Heaven and earth in exchange — a fertile, productive period; tend it without complacency."),
    (12, "000111", "否",   "Pǐ",       "Standstill",
     "stagnation, separation, withdrawal, blockage",
     "Heaven and earth no longer meeting — communication breaks down; the noble retreats and waits for spring."),
    (13, "101111", "同人", "Tóng Rén", "Fellowship with Men",
     "companionship, shared cause, public alliance, openness",
     "Brotherhood in the open — strength through shared purpose with people of like mind."),
    (14, "111101", "大有", "Dà Yǒu",   "Possession in Great Measure",
     "great possession, abundance, generous wealth, responsibility",
     "Great holding — riches and influence carry the duty to share and to remain modest."),
    (15, "001000", "謙",   "Qiān",     "Modesty",
     "humility, modesty, leveling, quiet excellence",
     "The mountain inside the earth — true depth that doesn't need to advertise; lasting respect comes from this."),
    (16, "000100", "豫",   "Yù",       "Enthusiasm",
     "anticipation, inspiring leadership, music, mobilization",
     "Movement that stirs others — the rallying drum; lead with warmth and people will follow gladly."),
    (17, "100110", "隨",   "Suí",      "Following",
     "following, adaptability, willingness to be led, responsiveness",
     "Knowing when to follow — adapt to what the moment offers and to people who deserve your trust."),
    (18, "011001", "蠱",   "Gǔ",       "Work on What Has Been Spoiled",
     "decay, repair, ancestral inheritance, reform",
     "Inherited rot that must be cleaned up — the slow, unglamorous work of restoring what has gone bad."),
    (19, "110000", "臨",   "Lín",      "Approach",
     "approaching opportunity, ascent, the right moment near",
     "The good is coming closer — prepare to meet it and act before the season turns again."),
    (20, "000011", "觀",   "Guān",     "Contemplation",
     "viewing, contemplation, observation, the ritual gaze",
     "Stand back and observe — what you see clearly from a height becomes the basis for right action."),
    (21, "100101", "噬嗑", "Shì Hé",   "Biting Through",
     "decisive action, clearing obstruction, justice, biting through",
     "An obstruction that must be bitten through — decisive, clean enforcement; do not flinch."),
    (22, "101001", "賁",   "Bì",       "Grace",
     "form, beauty, ornament, presentation",
     "Grace and adornment — make the form beautiful, but never let the surface outweigh the substance."),
    (23, "000001", "剝",   "Bō",       "Splitting Apart",
     "stripping, decay from below, retreat, conserving",
     "The supporting wall is being undermined — withdraw, preserve, and wait for the cycle to turn."),
    (24, "100000", "復",   "Fù",       "Return",
     "return, the turn of the year, renewed light, revival",
     "Light returning after dark — a single yang line at the bottom; small beginnings of recovery."),
    (25, "100111", "无妄", "Wú Wàng",  "Innocence",
     "the unexpected, spontaneity, integrity, truthful action",
     "Acting from the original nature — keep your motives pure and unforced, and what comes is the right response."),
    (26, "111001", "大畜", "Dà Chù",   "The Taming Power of the Great",
     "great accumulation, daily renewal, restraint, building reserves",
     "Storing and refining strength — the daily discipline that lets you act with weight when the time comes."),
    (27, "100001", "頤",   "Yí",       "The Corners of the Mouth",
     "nourishment, what you take in, speech, sustenance",
     "What feeds you matters — words, food, company; tend the inputs and the output takes care of itself."),
    (28, "011110", "大過", "Dà Guò",   "Preponderance of the Great",
     "excess, ridgepole bending, critical pressure, urgent action",
     "The beam is sagging — extraordinary stress that demands extraordinary response, but only briefly."),
    (29, "010010", "坎",   "Kǎn",      "The Abysmal (Water)",
     "danger, repeated trial, sincerity in peril, flowing through",
     "Danger upon danger — like water, do not cling; flow through, keep your shape, do not stop."),
    (30, "101101", "離",   "Lí",       "The Clinging (Fire)",
     "clarity, dependence, brilliance, attachment",
     "Fire clings to fuel — clarity comes from depending on the right thing; cling to what is real."),
    (31, "001110", "咸",   "Xián",     "Influence (Wooing)",
     "attraction, mutual feeling, courtship, sincere influence",
     "Two attracted to each other — the influence of feeling, the start of a real bond."),
    (32, "011100", "恆",   "Héng",     "Duration",
     "endurance, constancy, lasting union, steady course",
     "What lasts — the steady marriage, the long discipline; the form changes but the direction holds."),
    (33, "001111", "遯",   "Dùn",      "Retreat",
     "strategic withdrawal, dignified retreat, choosing your moment",
     "Step back with dignity — the wave is rising; you don't fight it, you leave the beach."),
    (34, "111100", "大壯", "Dà Zhuàng","The Power of the Great",
     "great vigor, surplus strength, rightful force, restraint",
     "Real strength — but precisely because it's real, you don't have to throw it around."),
    (35, "000101", "晉",   "Jìn",      "Progress",
     "advancement, sunrise, recognition, rising influence",
     "Sun rising over the earth — clear, recognized progress; let your light be plainly seen."),
    (36, "101000", "明夷", "Míng Yí",  "Darkening of the Light",
     "hidden brightness, persecution, inner light, endurance through dark times",
     "The light is wounded — you keep your fire inside, hold to your principles quietly until dawn."),
    (37, "101011", "家人", "Jiā Rén",  "The Family",
     "household, roles, inner order, kindness within structure",
     "A well-run household — clear roles, warm authority; outer order grows from inner order."),
    (38, "110101", "睽",   "Kuí",      "Opposition",
     "estrangement, polarity, misunderstanding, paradoxical alliance",
     "Two facing different directions — small things can still be done; insist on agreement and you lose both."),
    (39, "001010", "蹇",   "Jiǎn",     "Obstruction",
     "obstacle, impasse, finding helpers, turning back",
     "Stopped by something real — turn back, find allies, approach from another angle."),
    (40, "010100", "解",   "Xiè",      "Deliverance",
     "release, untying, forgiveness, springtime release",
     "The knot loosens — old burdens fall away; act quickly to clear the rest before they re-tangle."),
    (41, "110001", "損",   "Sǔn",      "Decrease",
     "deliberate decrease, sacrifice, simplifying, giving below",
     "Reducing the lower to enrich the upper — voluntary loss that produces real strength."),
    (42, "100011", "益",   "Yì",       "Increase",
     "increase, gain through generosity, beneficial growth",
     "Reducing the upper to enrich the lower — generosity from above; a season of mutual flourishing."),
    (43, "111110", "夬",   "Guài",     "Break-through (Resolution)",
     "decisive resolution, exposing wrongdoing, public resolve",
     "One yin clinging at the top — the moment to push through, openly and with allies, not by force alone."),
    (44, "011111", "姤",   "Gòu",      "Coming to Meet",
     "unexpected encounter, seductive newcomer, watchful caution",
     "A small dark force entering — meet it carefully; do not let the unfamiliar take root unchallenged."),
    (45, "000110", "萃",   "Cuì",      "Gathering Together",
     "assembly, congregation, sacrifice at the temple, leadership of the gathered",
     "People drawn together around something sacred — a leader holds the center, and the gathering does the work."),
    (46, "011000", "升",   "Shēng",    "Pushing Upward",
     "ascent, gradual growth, working upward, sustained climb",
     "Growth like a sapling — patient, daily upward effort; the right small step today, again tomorrow."),
    (47, "010110", "困",   "Kùn",      "Oppression (Exhaustion)",
     "depletion, being trapped, words distrusted, inner integrity",
     "Drained, trapped, unheard — outwardly nothing works; inwardly you keep your character intact."),
    (48, "011010", "井",   "Jǐng",     "The Well",
     "the village well, source, shared sustenance, depth",
     "The well at the center of the village — the true source feeds everyone; tend it and don't muddy the water."),
    (49, "101110", "革",   "Gé",       "Revolution (Molting)",
     "decisive change, shedding the skin, the right moment to overturn",
     "Time to overturn the old order — but only when belief, timing, and necessity align; not before."),
    (50, "011101", "鼎",   "Dǐng",     "The Cauldron",
     "the sacred vessel, transformation, cultivation, nourishment of the wise",
     "The ritual cauldron — what you cook in it nourishes the worthy; institutions and practices that transform."),
    (51, "100100", "震",   "Zhèn",     "The Arousing (Thunder)",
     "shock, sudden movement, awakening fear, reverence",
     "Thunder shakes everyone — initial fear, then laughter; respond with composure and the shock blesses."),
    (52, "001001", "艮",   "Gèn",      "Keeping Still (Mountain)",
     "stillness, resting in the back, meditation, silenced ego",
     "The mountain — stop the inner chatter; act only from a place that is fully still."),
    (53, "001011", "漸",   "Jiàn",     "Development (Gradual Progress)",
     "stages, gradualness, courtship, the right sequence",
     "Slow, ordered advance — each stage in its proper time; rushing forfeits the whole."),
    (54, "110100", "歸妹", "Guī Mèi",  "The Marrying Maiden",
     "subordinate position, irregular union, accepting limits, integrity within constraint",
     "An imperfect arrangement — accept the limited role gracefully; what matters is how you carry yourself within it."),
    (55, "101100", "豐",   "Fēng",     "Abundance (Fullness)",
     "peak, abundance at noon, awareness of decline, full sun",
     "High noon — the peak; enjoy it without forgetting that the sun has begun to lean west."),
    (56, "001101", "旅",   "Lǚ",       "The Wanderer",
     "traveler, transient lodging, modesty among strangers, careful conduct",
     "Travelling through unfamiliar country — keep small, keep careful; no one owes the stranger his welcome."),
    (57, "011011", "巽",   "Xùn",      "The Gentle (Wind)",
     "penetrating influence, gentle persistence, repeated guidance",
     "Wind, again and again — small persistent influence works where one big push would fail."),
    (58, "110110", "兌",   "Duì",      "The Joyous (Lake)",
     "joy, openness, conversation, sincere pleasure",
     "Two lakes feeding each other — joy that compounds in honest exchange; the right kind of cheer."),
    (59, "010011", "渙",   "Huàn",     "Dispersion (Dissolution)",
     "dissolving rigidities, melting hardness, scattering the egotism",
     "Wind moving over water — what was frozen melts; dissolve the hardness so the larger movement can resume."),
    (60, "110010", "節",   "Jié",      "Limitation",
     "appropriate limits, measure, joyful restraint, banks of the stream",
     "Banks make the river — accept the right limits and the energy is shaped, not stopped."),
    (61, "110011", "中孚", "Zhōng Fú", "Inner Truth",
     "sincerity, truth that reaches the depths, trust beneath words",
     "Truth that goes all the way through — sincerity that even pigs and fishes can feel; this is what moves people."),
    (62, "001100", "小過", "Xiǎo Guò", "Preponderance of the Small",
     "small excess, attention to detail, modest action, the bird returning to its nest",
     "A small surplus — favor the modest gesture and the careful step; this isn't the moment for grand."),
    (63, "101010", "既濟", "Jì Jì",    "After Completion",
     "completion, fragility of order, vigilance after success",
     "The crossing is finished — every line in its right place; the danger is now complacency, not failure."),
    (64, "010101", "未濟", "Wèi Jì",   "Before Completion",
     "the work not yet done, near miss, careful approach to the last step",
     "Almost across the river — every line is in the wrong place; the work is real and the finish requires care."),
]


HEXAGRAMS: tuple[Hexagram, ...] = tuple(
    Hexagram(*row) for row in _RAW
)

# Index by 6-bit pattern → hexagram, for fast lookup from cast results.
HEXAGRAM_BY_LINES: dict[str, Hexagram] = {h.lines: h for h in HEXAGRAMS}

# Sanity: King Wen has exactly 64 distinct patterns.
assert len(HEXAGRAMS) == 64, f"expected 64 hexagrams, got {len(HEXAGRAMS)}"
assert len(HEXAGRAM_BY_LINES) == 64, "duplicate line patterns in dataset"
