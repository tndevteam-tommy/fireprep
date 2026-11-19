"""Everything anybody says, with variety.

Two kinds of talk come out of here:

* **Lines** - situational remarks (greeting the player, a guard's warning, a merchant
  refusing stolen goods...). ``line(kind, rng, **fmt)`` picks one of several phrasings
  so the same situation is not worded the same way twice in a row.
* **Conversations** - short exchanges between two NPCs who meet: a greeting suited to
  the time of day, a reply, one or two topical exchanges coloured by who they are
  (a guard grumbles about the watch, the smith about iron, the innkeeper about
  thirsty customers) and a goodbye. ``build_conversation`` returns the scripted
  lines; the world plays them out a few seconds apart as speech bubbles.

Nothing here touches game state - it is all text and random choice - so it is
trivial to extend: add strings to the tables.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------- situational lines
LINES: Dict[str, Sequence[str]] = {
    # the player talks to someone
    "greet_guard": ("Keep your nose clean, traveller. Move along.", "Nothing to see here. On your way.",
                    "Mind yourself on my street.", "Quiet day. Let's keep it that way.", "You lost, friend? The inn's that way."),
    "greet_fence": ("Got stolen goods? I'll buy them, no questions asked.", "Anything warm in your pockets? I pay fair. Quietly.",
                    "Keep your voice down and show me what you've got.", "The guards don't come in here. Business?"),
    "greet_fence_woken": ("Mrf... it's you. Fine. Show me what you've got, then let me sleep.",
                          "At this hour? ...Coin's coin. What have you brought?",
                          "No rest for the wicked. Quick, then, before I change my mind."),
    "greet_merchant": ("Welcome! Take a look at my wares.", "Good day to you. Browse as long as you like.",
                       "Ah, a customer. What can I get you?", "Everything's fresh in. Have a look."),
    "greet_smith": ("Blades, helms, tools and good iron. Nothing else - I'm a smith, not a pedlar.",
                    "Mind the sparks. Looking for steel?", "If it's iron you want, you've come to the right forge.",
                    "I'll take ore off your hands, or sell you something sharp."),
    "greet_innkeeper": ("Welcome to the Drowsy Dragon! Ale, stew, or a bed for the night?",
                        "Sit anywhere, love. Rooms are ten a night, fifty the week.",
                        "You look like you could use a hot meal. Or a bed.", "Ale's fresh and the beds are clean. What'll it be?"),
    "greet_tailor": ("Looking for something finer to wear?", "Cloth, thread and quiet shoes - come in.",
                     "Mind the pins. What can I fit you for?", "Soft soles for soft steps, if you take my meaning."),
    "inn_welcome": ("Welcome to the Drowsy Dragon! A room for the night? Ten a night, fifty the week.",
                    "Come in, come in! Looking for a bed? I've rooms free - ten a night.",
                    "A traveller! Warm yourself. A room's ten a night if you're stopping."),
    "inn_welcome_woken": ("Mm? A guest, at this hour... well, come in. A room is ten a night, if you're stopping.",
                          "Who's... oh, a customer. No rest for an innkeeper. A bed's ten for the night.",
                          "(yawning) You'll be wanting a room, I suppose. Ten a night. Say the word."),
    "greet_closed": ("The shop's closed right now. Come back during opening hours.", "Not open yet, friend. Later.",
                     "We're shut. Try again when the sign's out."),
    "greet_asleep": ("Zzz... (fast asleep)", "Zzz... mm... not the candlesticks... zzz",
                     "(a snore, then a mutter) ...five more minutes..."),
    "greet_townsfolk": ("Morning to you.", "Fine day for it.", "Don't I know you? No? Well, good day.",
                        "Mind the well, the step's loose.", "If you're after Barnaby, he's at the store.",
                        "Lovely evening. Off to the inn?", "Haven't seen you about before."),
    "greet_townsfolk_night": ("Late to be out, isn't it?", "Can't sleep either?", "Keep your torch lit, the watch are twitchy."),
    # warnings and trouble
    "trespass": ("Hey! You've no business in here. Out!", "What are you doing in here? Get out!",
                 "This is private. Leave, now.", "Out! Before I call the watch."),
    "torch_law": ("It's the law to carry a lit torch after dark. Light one, now!", "No torch after dark? Light one or I'll assume the worst.",
                  "Torch. Lit. Now. Town law."),
    "skulking": ("You there! What are you skulking about for?", "Stand up straight. What are you up to?",
                 "Creeping about, are we? I'm watching you.", "I see you sneaking. Don't."),
    "skulking_citizen": ("Hey! What are you skulking about for?", "What's all this creeping? Stand up!", "I see you there. Stop that."),
    "halt": ("Halt right there, criminal scum!", "Stop! In the name of the watch!", "That's it - you're coming with me!"),
    "cry_for_help": ("Guards! Help! A criminal is committing crimes!", "Help! Thief! Somebody fetch the watch!",
                     "Guards! Over here, quickly!"),
    "caught_red_handed": ("Caught you red-handed!", "Again?! Stop right there, thief!",
                          "That's enough! One more and you're coming with me!", "You're under arrest the moment I get my hands on you!"),
    "thief_guards": ("Thief! Guards!", "Stop, thief! Guards!", "Someone's robbing me! Guards!"),
    "stolen_goods": ("That {item} is stolen! Guards!", "I know that {item} - it's stolen! Guards!",
                     "Get that hot {item} away from me! Guards!"),
    "cant_afford": ("I can't afford that right now - I've only {gold}g in the till.", "Not at that price, not today. The till has {gold}g.",
                    "Come back when I've sold something - {gold}g is all I have."),
    "wrong_trade": ("I don't deal in that sort of thing.", "Take that elsewhere - I'm a {role}.",
                    "Not my trade. Try Barnaby."),
    "cry_response": ("A cry for help! On my way!", "Trouble! Hold on!", "I heard that - coming!"),
    "investigate": ("What was that? I'd better take a look.", "Hm? Who's there?", "Did you hear that? I'll check."),
    "gone": ("Gone... but I'll know that face when I see it.", "Slipped away. Not for long.", "Nothing here. Eyes open, then."),
    "woke_up": ("Who's there?! Making all that noise!", "What in blazes... who's there?", "Wha-? Who's making that racket?"),
    "racket_guard": ("Keep that clanking down! You'll have the whole street up.", "By the saints - must you rattle like a tinker's cart?",
                     "Quiet, you. Some of us are listening for thieves.", "Half the town can hear you coming. Pipe down."),
    "racket_citizen": ("What a racket! Can't you walk quietly?", "Clank, clank, clank - have a thought for folk abed!",
                       "Must you make that noise? The whole street can hear you.", "Is that a man or a cart of pans?"),
    "cell_noise": ("Keep it down in there!", "Quiet, you!", "Rattle those bars all you like."),
    "leave_late": ("It's late. Time for you to go - I want to turn in.", "Off home with you, it's past my bedtime.",
                   "I'll be locking up now. Out you go, there's a good fellow."),
    "leave_again": ("I said it's late. Off with you, I'm not asking again.", "Still here? Out. I mean it.",
                    "Don't make me say it a third time."),
    "leave_now": ("Right, that's it. Out!", "Enough! Out of my house!", "That does it. Out!"),
    # shopkeepers and the player on their premises
    "shop_skulk": ("Stand up straight in my shop, or get out of it.", "None of that creeping about in here. Up, or out.",
                   "What are you crouching for? Stand up or leave."),
    "shop_skulk_again": ("I said stand up. Out, or I'll have the watch on you.", "Still skulking? Right. Out you go.",
                         "Last warning: on your feet, or out of the door."),
    "shop_out": ("That's it. Out of my shop!", "Enough! Out, and don't come creeping back.", "Out! Go on, out!"),
    "shop_closing": ("We're closing. Out you go.", "Closing time - off with you, I'm locking up.",
                     "Still here? I'm shutting the shop. Out."),
    "locked_out": ("Locked. I'll wait out here.", "Not open yet? Give it a minute.", "Hm, shut. They'll be along.",
                   "Where is everyone? I'll wait."),
    "no_food": ("There's nothing to eat! Has someone been at the food?", "Empty! Who's had my supper?",
                "Not a crumb left. Thieves, I'll wager."),
    "shopping": ("I'll take two of those.", "How much for the honey?", "That's dearer than last week!", "Just looking, thank you.",
                 "Put it on my account, would you?", "Fresh in, you say? I'll have one."),
    "working_smith": ("*clang* *clang*", "Hotter. It needs to be hotter.", "There's a good edge coming on this one.",
                      "*the bellows wheeze*"),
    "working_kitchen": ("Pots on, Hilde!", "Stew's nearly ready.", "Mind the bread, it's catching!"),
    "working_porter": ("Where d'you want these, Barnaby?", "*grunts under a crate*", "That's the last of the wine."),
    "cleaning": ("Who left this here?", "Ale rings everywhere. Honestly.", "Mind your feet, I'm sweeping."),
    "drinking": ("Another, Hilde!", "Best ale in the county, this.", "*sips* Ahh.", "Same again, when you've a moment."),
    "rent_night": ("{Room} is yours for the night. Sleep well - checkout's at noon.",
                   "Ten gold and the night is yours. {Room}, up the hall."),
    "rent_week": ("A week in {room}, then. I'll see you're not disturbed.", "Fifty for the week - {room} is yours till then."),
    "rent_more": ("Staying on? {Room} is yours till noon in {nights} days, then.",
                  "Another night in {room}. I'll keep the bed made up."),
    "rent_none": ("I'm full up tonight, sorry. Try again tomorrow.",),
    "rent_over": ("Your rent's run out, love. Time to settle up or move along.", "Checkout was noon. Out you come."),
    # the rich: two old families on the hill, and everybody else beneath them
    "greet_rich": ("Do I know you? No. I thought not.", "You are standing in my light. Move.",
                   "Good heavens, it speaks. What do you want?", "If you're begging, the chapel is down the hill.",
                   "Mind the hem - this silk cost more than your whole family.", "Servants use the back lanes, you know."),
    "greet_rich_night": ("At this hour? Whatever it is, the answer is no.", "The watch will hear of this. Off with you.",
                         "Go home, wherever that is. A ditch, I imagine."),
    "rich_ambient": ("Ugh. The smell of this street.", "Does nobody in this town own a broom?",
                     "Cheap, cheap, cheap. Is there anything here that isn't cheap?", "One does try to be gracious to the poor.",
                     "I'll have that guard's badge for slouching.", "In my grandfather's day this rabble knew its place."),
    # out in the country
    "greet_villager": ("Mornin'. Come from the town, have you?", "Mind the furrows.", "Don't see many townsfolk out here.",
                       "If it's work you want, the harvest needs hands.", "Lovely day for it, whatever it is."),
    "greet_hunter": ("Quiet. You'll scare the deer.", "The woods are no place to wander after dark.",
                     "Stay on the path, stranger. The Thornwood swallows people.",
                     "There's worse than wolves in those caves. Keep away from them."),
    "greet_bandit": ("Walk on, stranger. Quick, while you still can.", "You're a long way from the town watch.",
                     "Lost? You'd best find yourself again - elsewhere."),
    "greet_knight": ("Greyhold keeps its own peace, traveller. Keep it with us.", "State your business at the keep.",
                     "Mind where you tread - the old walls are coming down."),
    "greet_smuggler": ("Fishing. We're fishermen. Don't look in the boat.", "Lost? The road's back up the beach.",
                       "Tide's turning. So should you."),
    "greet_squire": ("Good day! Are you here about the tithe? I'm not supposed to talk about the tithe.",
                     "I'm to be a knight at midsummer! Probably.", "Ser Caddock says I polish like a wet cat. I'm improving."),
    "greet_boss": ("You've nerve, walking into my hall. I'll give you that.", "Another fool for the Deep. Turn around.",
                   "Everything in this hole is mine. Everything. Go."),
    # wardens: outlaws and knights who catch somebody where they should not be
    "warden_warn_bandit": ("You've no business in here. Turn round - now.", "Wrong cave, friend. Walk away while you can.",
                           "One more step and you'll wish you hadn't."),
    "warden_warn_boss": ("Back the way you came, worm. Last chance.", "You stand in the King's hall. Leave.",
                         "Turn round, or they'll carry you out."),
    "warden_warn_knight": ("Halt. Greyhold is closed to strangers - go back the way you came.",
                           "You trespass in the keep. Leave, in peace, now.", "No further. Turn about."),
    "warden_shout_bandit": ("Oi! Who let you in?", "Intruder! Get them!", "You picked the wrong cave, friend!"),
    "warden_shout_boss": ("A thief, in MY hall? Seize them!", "Nobody touches the King's hoard. Nobody!",
                          "You'll not leave this place with anything of mine."),
    "warden_shout_knight": ("Halt! You trespass in Greyhold!", "Stand where you are, thief!", "To arms! An intruder in the keep!"),
    "warden_seize_bandit": ("Out you go - and your purse stays here.", "Count yourself lucky we're only emptying your pockets.",
                            "Come back and we'll bury you in the bluff."),
    "warden_seize_boss": ("Throw this one out. And take the toll - everyone pays the toll.",
                          "You'll crawl out of the Deep lighter than you came in.", "Next time I feed you to the dark."),
    "warden_seize_knight": ("Out, and be grateful it's the road and not the gallows.", "Greyhold's goods stay in Greyhold. As does your fine.",
                            "Take them to the road. The fine is paid in coin, thief."),
    # the town gate, shut to the player until the tutorial is over
    "gate_refusal": ("Gate's barred and it stays barred. Nobody goes out without the captain's leave - and you haven't got it.",
                     "Not tonight, not tomorrow. Whatever business you've got, it's inside these walls.",
                     "Back off from the gate. You've debts in this town, by the look of you - settle them first."),
}

# ----------------------------------------------------------------------------- conversations
GREETINGS: Dict[str, Sequence[str]] = {
    "morning": ("Morning, {other}.", "Up early, {other}!", "Good morning to you, {other}.", "A fine morning, {other}."),
    "day": ("Afternoon, {other}.", "Well met, {other}.", "{other}! Good to see you.", "Busy day, {other}?"),
    "evening": ("Evening, {other}.", "There you are, {other}.", "Good evening, {other}.", "Long day, {other}?"),
    "night": ("Late for you, {other}.", "Can't sleep, {other}?", "Still up, {other}?"),
}
GREETING_REPLIES = ("And to you, {other}.", "Same to you.", "Aye, {other}. Yourself?", "Can't complain, {other}.",
                    "Well enough, thank you.", "Tolerable, tolerable.")

# (opener, reply) pairs; {other} is the listener's name. Role-specific topics are used when either speaker has the role.
TOPICS: Dict[str, Sequence[Tuple[str, str]]] = {
    "weather": (("Warm one today.", "Aye, the dust on the street says so."),
                ("Wind's turning - rain by morning, I'd say.", "My knees agree with you."),
                ("Lovely light on the street today.", "It is. Makes the place look almost respectable.")),
    "town": (("Have you seen the state of the well step?", "Someone will break an ankle on it."),
             ("Quiet lately, hasn't it been?", "Too quiet. Makes the watch nervous."),
             ("Did you hear about the candlesticks going missing?", "Barnaby's still fuming about it."),
             ("The inn was lively last night.", "Fenwick was, you mean.")),
    "trade": (("Prices at the store have crept up again.", "Everything has. Except my wages."),
              ("Mirelle has new cloth in from the south.", "Pretty, I'm sure. Not that I could afford it."),
              ("Gareth's hammer never stops.", "You can hear it from the far wall.")),
    "guard": (("Anything on the watch, {other}?", "Nothing but cats and drunks. The way I like it."),
              ("They say the captain's got you doubling the rounds.", "After the business at the store? Fair enough."),
              ("Quiet night ahead, {other}?", "If I've any say in it. Keep your door locked all the same.")),
    "smith": (("How's the iron coming, {other}?", "Stubborn. Like its owner, they tell me."),
              ("That forge could warm the whole street.", "It warms my bones, that's enough."),
              ("Any ore left in that seam?", "Enough for a few more swords, if the lad keeps digging.")),
    "innkeeper": (("Is the stew on, Hilde?", "Since dawn. Sit down before Fenwick has it all."),
                  ("Rooms full tonight?", "Two of three. There's always one for a friend."),
                  ("The ale's good this week.", "Don't tell Fenwick or he'll never leave.")),
    "shopkeeper": (("Got any honey in, Barnaby?", "Two jars, and the bees want a raise."),
                   ("You've let Dorran loose on the crates again.", "He's strong. Strong and careless."),
                   ("Stock counted this morning?", "Every bottle. Twice.")),
    "tailor": (("Is that a new cloak, Mirelle?", "Velvet. Don't touch it with those hands."),
               ("Could you take in a coat for me?", "Bring it by. And bring coin.")),
    "gossip": (("Tamsin says there was someone creeping about last night.", "Tamsin says a lot of things."),
               ("Pell's sweet on the tailor, you know.", "Pell's sweet on anyone who'll sell him a cloak on credit."),
               ("Jae keeps odd hours.", "Jae keeps odd everything. Best not to ask."),
               ("Lord Vane sent his soup back at the inn again.", "Hilde says she'll put something in it next time."),
               ("Lady Ashgrove called my boots 'a tragedy'.", "She called my face one. Twice."),
               ("Old Bertram's knee again - an arrow, he says.", "Last week it was a pike. Next week, a dragon."),
               ("Dunstan's written a song about the Reeve.", "Not one he'll sing in front of him, I'll bet."),
               ("Who's the lass Hobb keeps combing his hair for?", "Somebody up the hill. Poor lad.")),
    # the townsfolk's own trades (asked of the one who plies it)
    "priest": (("Will you say a word for my mother on Sunday, Sister?", "Every Sunday, {other}. She'd have liked the candles."),
               ("Is the bell cracked, Sister? It sounds sour.", "It sounds honest. That's rarer than you'd think.")),
    "crier": (("What's the news, {other}?", "Curfew at ten, tithe at midsummer, and my knee says rain."),
              ("Were you really in the watch, {other}?", "Twelve years. Then I took an arrow in the knee.")),
    "baker": (("Is there any of the seed loaf left, {other}?", "One, if you're quick. Lord Vane's man takes six a day."),
              ("You're up before the cocks, {other}.", "Ovens don't light themselves. More's the pity.")),
    "cooper": (("Is that rain barrel sound, {other}?", "Sound as me. Don't go looking inside it, mind."),
               ("Busy at the staves?", "The inn goes through barrels like Fenwick goes through ale.")),
    "stablehand": (("Still no horses, {other}?", "Not a one. I groom the stalls out of habit."),
                   ("You've combed your hair, {other}.", "Have I? Must be the wind.")),
    "herbwife": (("Have you anything for a bad back, {other}?", "Willow bark and a softer bed. Three pennies."),
                 ("They say you can read the leaves, {other}.", "I can read faces. Leaves are for the gullible.")),
    "locksmith": (("Can you fix a sticking lock, {other}?", "I can fix any lock. Opening them's another trade entirely."),
                  ("Who bought that big padlock off you?", "A locksmith never tells. That's the whole business.")),
    "reeve": (("When's the tithe collector due, Reeve?", "Midsummer, as every year. Have your share ready."),
              ("The well step's cracked again, Reeve.", "Then the well step can wait its turn like everybody else.")),
    "minstrel": (("Give us a song, {other}.", "For a penny, a verse. For a cup of ale, the whole ballad."),
                 ("Heard you played for the Ashgroves, {other}.", "Once. They paid me to stop.")),
    "tanner": (("Phew, {other}. The vats are ripe today.", "That's the smell of honest money."),
               ("Any hides for sale, {other}?", "Always. Just don't ask where the deer came from.")),
    "lamplighter": (("Late rounds again, {other}?", "Every night. The lamps don't light themselves either."),
                    ("See anything on your rounds, {other}?", "I see everything on my rounds. I say nothing. Healthier.")),
    "butler": (("How's his lordship, {other}?", "Displeased. He is always displeased. It keeps him young."),),
    "maid": (("Run off your feet, {other}?", "Her ladyship changed her gown four times before noon."),),
    "washerwoman": (("Lovely drying day, {other}.", "It is till the tanner lights his fires."),),
    "hunter": (("Any game in the Thornwood, {other}?", "Less every year. The bandits eat better than I do."),),
    "farmer": (("How are the fields, {other}?", "Thirsty. Like me."),),
    # the rich speak first, and down: to townsfolk ...
    "rich": (("Must you stand so close, {other}? One can smell the gutter on you.", "Beg pardon. I'll stand further off."),
             ("{other}, isn't it? Tell your people to sweep the street. It offends me.", "They're not my people, but... yes."),
             ("I suppose you've never worn silk, {other}. No. Of course you haven't.", "Can't say I have."),
             ("Is this what passes for a market? Dreadful. Simply dreadful.", "It's the only one we've got."),
             ("Curtsey when you address me, {other}. Were you raised in a barn?", "...The one behind the inn, as it happens.")),
    # out in the caves: the smugglers of the Saltcove ...
    "smuggler": (("Boat's late.", "Boat's always late. The tide doesn't read Kell's ledger."),
                 ("How much did we clear on the silk, {other}?", "Enough to keep Maggs in pepper till midsummer.",
                  "She'll only put it in the stew."),
                 ("Dame Ilsa's been walking the shore at night.", "Knights. Can't sleep, can't drink, can't take a bribe.",
                  "Caddock takes them. Just not where she can see."),
                 ("Finch thinks he's getting his own boat.", "Finch thinks a lot of things.",
                  "Let him. Somebody in this cave ought to think."),
                 ("Brannoc's staring at the crack in the wall again.", "Says Jory's in there. Listening.",
                  "Jory's in Saltmarsh with a wife and my share of the silk."),
                 ("If the Deep lot come calling again...", "Then we put to sea. Grimmaw hasn't got a boat, and he can't swim."),
                 ("Kell counts the take every evening, same as a priest at his beads.", "And sleeps till noon. Like a priest."),
                 ("Pepper, silk, brandy. What's in the long crates, {other}?", "Don't ask what's in the long crates.")),
    # ... and the Howling Deep's crew, the bandit king's men
    "bandit": (("Your throw, {other}.", "Sixes again, Ned? Those dice lean.", "All dice lean. Mine are honest about it."),
               ("Grimmaw's not slept again.", "Three hours. Always three. Like a clock with a grudge."),
               ("They've doubled the gate at Greyhold.", "Let them. At midsummer the collector comes. Wagons on the road. Soft wagons."),
               ("What's behind the iron door, anyway?", "Don't ask.", "Last one who asked is still in the vault. Bones and all."),
               ("Spring's running low.", "Crake'll have us drinking his stew.", "Better than eating it."),
               ("Anybody seen Gutter's lucky tooth?", "He hides it in the bunk room wall. Thinks nobody knows.",
                "Everybody knows. Nobody wants it."),
               ("Sable's put forty arrows in the straw man today.", "He owed her money."),
               ("Footprints by the crack behind the spring.", "The prospector's ghost.", "Ghosts don't leave footprints, {other}."),
               ("Who's got the mouth tonight?", "Vey, till the small hours. Then Sable, if she's sober.")),
    "boss": (("Who's on the mouth, {other}?", "Vey, my king. Till the fourth hour.", "And when he sleeps?",
              "...Sable walks the halls."),
             ("The collector's wagons come through the Thornwood at midsummer.", "We'll be waiting, my king."),
             ("Greyhold took my spurs. I'll take their tithe.", "And their heads, my king?", "One thing at a time, {other}.")),
    # ... the Order at Greyhold
    "knight": (("The tithe's counted?", "Twice. Ser Caddock counts it again at noon.", "He'd count it in his sleep, if he slept."),
               ("Lights on the Saltcove again last night.", "Smugglers. The excise will see to it.",
                "The excise drinks Kell's brandy and you know it."),
               ("Any word of Grimmaw?", "Hunters say he's dug in under the mountain. Deeper every year."),
               ("Did the boy polish the cuirass?", "He polished everything. Including the dog.")),
    "squire": (("When will I be knighted, {other}?", "When you stop asking.", "...Midsummer, then?"),
               ("Is it true Grimmaw was one of the Order?", "It is. And if you ever say his name in the hall again, you'll polish the privy.")),
    # ... and to the watch they pay for
    "rich_guard": (("You there, {other}. Is slouching part of your duties, or a personal talent?", "Very good. I'll stand straighter."),
                   ("I pay your wages, {other}. Try to look as if you earn them.", "The town pays me, m'lord. But aye."),
                   ("A thief could walk past you with my silver and you'd wave him through.", "Not on my watch."),
                   ("Keep the riff-raff off my street tonight, {other}. All of it.", "I'll do my rounds as always.")),
}
FAREWELLS = ("Well, I'd best get on.", "Mustn't keep you.", "Take care, {other}.", "See you at the inn.",
             "Good day to you, {other}.", "I'll let you get back to it.")
FAREWELL_REPLIES = ("Mind how you go.", "Aye, see you.", "Till later, {other}.", "Don't work too hard.", "Go well.")


OUTLAW_ROLES = ("smuggler", "bandit", "boss")


def time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "day"
    if 18 <= hour < 22:
        return "evening"
    return "night"


def line(kind: str, rng: random.Random, **fmt) -> str:
    """One of the phrasings for ``kind``, formatted."""
    options = LINES[kind]
    return rng.choice(list(options)).format(**fmt)


@dataclass
class Conversation:
    """A scripted exchange between two NPCs, played out by the world a few seconds per line.
    Casual chats break off as soon as either party has something better to do; an
    *official* exchange (a guard taking a theft report) is seen through unless somebody is
    asleep or alarmed, and may run ``on_finish`` afterwards. Lines are ``logged`` only when
    they are part of the player's story (a report), never idle gossip."""
    a: object                         # NPC
    b: object                         # NPC
    lines: List[Tuple[object, str]]   # (speaker, text) in order
    index: int = 0
    next_line_at: float = 0.0
    finished: bool = False
    official: bool = False
    logged: bool = False
    on_finish: Optional[Callable[[], None]] = None
    participants: Tuple = field(init=False)

    def __post_init__(self):
        self.participants = (self.a, self.b)


# --- a guard taking a shopkeeper's theft report -------------------------------------------
REPORT_LINES: Dict[str, Sequence[str]] = {
    "apology": ("Sorry about that, {merchant}. Duty called. Now, where were we?",
                "My apologies, {merchant} - that couldn't wait. Back to your trouble.",
                "Forgive the interruption, {merchant}. Let's start again."),
    "ask_what": ("All right, {merchant}. What's been taken?", "What's missing, {merchant}?", "Calm down. What's gone?"),
    "tell_what": ("{missing} - gone from {area}.", "{missing}. Cleared off {area} while my back was turned.",
                  "Look for yourself: {missing}, taken from {area}."),
    "guard_saw": ("I saw someone skulking about here myself. I'll have a word with them.",
                  "There was someone hanging about {area} who had no business there. I'll find them."),
    "merchant_glad": ("Do. And get my goods back.", "Good. I want every piece of it back."),
    "ask_seen": ("Did you see anyone suspicious?", "Anyone hanging about who shouldn't have been?"),
    "merchant_saw": ("Yes - someone poking about where they'd no business. I'd know the face again.",
                     "There was one, creeping about {area}. I'd know them if I saw them."),
    "guard_lead": ("Then I'll find them. Leave it with me.", "Good. I'll have a word with them."),
    "merchant_saw_nothing": ("Not a soul. It must have been done in the night.", "No. Whoever it was, they were quiet about it."),
    "guard_no_lead": ("Keep it locked up. I'll keep an eye out.", "Then there's not much to go on. I'll keep my eyes open."),
}


def build_report_exchange(guard, merchant, missing: str, area: str, guard_saw: bool, merchant_saw: bool,
                          rng: random.Random, apologise: bool = False) -> Conversation:
    """The guard who answered a theft report takes it from the shopkeeper face to face. The
    replies follow what each of them actually saw: a guard with a sighting of their own needs
    no description; otherwise he asks, and the merchant answers according to whether anyone was
    caught poking about the premises. A guard coming back to a report a crime cut short
    ``apologise``s first."""
    name = short_name(merchant)
    fmt = dict(merchant=name, missing=missing, area=area)
    pick = lambda kind: rng.choice(list(REPORT_LINES[kind])).format(**fmt)   # noqa: E731
    lines: List[Tuple[object, str]] = []
    if apologise:
        lines.append((guard, pick("apology")))
    lines += [(guard, pick("ask_what")), (merchant, pick("tell_what"))]
    if guard_saw:
        lines += [(guard, pick("guard_saw")), (merchant, pick("merchant_glad"))]
    else:
        lines.append((guard, pick("ask_seen")))
        if merchant_saw:
            lines += [(merchant, pick("merchant_saw")), (guard, pick("guard_lead"))]
        else:
            lines += [(merchant, pick("merchant_saw_nothing")), (guard, pick("guard_no_lead"))]
    return Conversation(guard, merchant, lines, official=True, logged=True)


def short_name(npc) -> str:
    """'Shopkeeper Barnaby' -> 'Barnaby', 'Guard Captain Vance' -> 'Captain', 'Jae the Fence' -> 'Jae'."""
    name = npc.name
    for prefix in ("Shopkeeper ", "Guard Captain ", "Guard ", "Jailer "):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            return "Captain" if prefix == "Guard Captain " else rest
    return name.split(" the ")[0]


def build_conversation(a, b, hour: int, rng: random.Random, exchanges: Optional[int] = None,
                       recent: Optional[Sequence[str]] = None) -> Conversation:
    """Greeting, reply, one or two topical exchanges coloured by the speakers' roles, goodbye, reply.
    Exchanges whose opener is in ``recent`` are avoided so the same gossip does not do the rounds
    twice in an afternoon."""
    na, nb = short_name(a), short_name(b)
    recent = set(recent or ())
    lines: List[Tuple[object, str]] = []
    lines.append((a, rng.choice(list(GREETINGS[time_of_day(hour)])).format(other=nb)))
    lines.append((b, rng.choice(list(GREETING_REPLIES)).format(other=na)))
    rich = [n for n in (a, b) if getattr(n, "role_tag", "") == "rich"]
    if rich and not all(getattr(n, "role_tag", "") == "rich" for n in (a, b)):
        # somebody from the hill: whatever else is said, they talk down to the other - guards included
        other = b if rich[0] is a else a
        pool = ["rich_guard" if other.is_guard else "rich"]
    elif any(getattr(n, "role_tag", "") in OUTLAW_ROLES for n in (a, b)):
        # down in a cave nobody talks about the weather or the price of honey at Barnaby's
        pool = [getattr(n, "role_tag", "") for n in (a, b) if getattr(n, "role_tag", "") in TOPICS]
    else:
        pool = ["weather", "town", "gossip"]
        for npc in (a, b):
            role = getattr(npc, "role_tag", "") or ("guard" if npc.is_guard else "")
            if role in TOPICS and role != "rich":
                pool.append(role)
            if npc.is_guard:
                pool.append("guard")
        if any(getattr(n, "can_trade", False) for n in (a, b)):
            pool.append("trade")
    count = exchanges if exchanges is not None else rng.choice((1, 1, 2))
    rng.shuffle(pool)
    # The pool is weighted - a role listed by both speakers (two guards) appears several times, so it
    # comes up first more often - but one conversation never goes over the same topic twice.
    topics = list(dict.fromkeys(pool))
    for topic in topics[:count]:
        options = [pair for pair in TOPICS[topic] if pair[0] not in recent] or list(TOPICS[topic])
        exchange = rng.choice(options)
        # the person the topic is *about* (a smith topic addressed to the smith) should be the replier;
        # the rich always have the first word
        speaker, listener = a, b
        role_of_a = getattr(a, "role_tag", "") or ("guard" if a.is_guard else "")

        def about(npc) -> bool:
            """The topic is this one's line of work (any of the watch - a captain, a jailer - for "guard")."""
            role = getattr(npc, "role_tag", "") or ("guard" if npc.is_guard else "")
            return topic == role or (topic == "guard" and npc.is_guard)
        if topic in ("rich", "rich_guard"):
            speaker, listener = (a, b) if role_of_a == "rich" else (b, a)
        elif topic in ("boss", "squire"):
            # the king has the first word, and the squire never stops asking
            speaker, listener = (a, b) if role_of_a == topic else (b, a)
        elif about(a) and not about(b):
            speaker, listener = b, a
        # an exchange runs to as many lines as it has, the two taking turns
        for k, text in enumerate(exchange):
            who, to = (speaker, listener) if k % 2 == 0 else (listener, speaker)
            lines.append((who, text.format(other=short_name(to))))
    lines.append((a, rng.choice(list(FAREWELLS)).format(other=nb)))
    lines.append((b, rng.choice(list(FAREWELL_REPLIES)).format(other=na)))
    return Conversation(a, b, lines)


def player_greeting(npc, hour: int, rng: random.Random) -> str:
    """What an NPC says when the player talks to them."""
    if npc.is_asleep:
        return line("greet_asleep", rng)
    if npc.is_guard:
        return line("greet_guard", rng)
    if npc.is_fence:
        return line("greet_fence_woken" if npc.awake_for_business else "greet_fence", rng)
    if npc.can_trade:
        if not npc.is_open_for_business:
            return line("greet_closed", rng)
        role = getattr(npc, "role_tag", "")
        kind = {"smith": "greet_smith", "innkeeper": "greet_innkeeper", "tailor": "greet_tailor"}.get(role, "greet_merchant")
        return line(kind, rng)
    role = getattr(npc, "role_tag", "")
    if role == "rich":
        return line("greet_rich_night" if time_of_day(hour) == "night" else "greet_rich", rng)
    outsider = {"farmer": "greet_villager", "villager": "greet_villager", "hunter": "greet_hunter", "bandit": "greet_bandit",
                "knight": "greet_knight", "boss": "greet_boss", "smuggler": "greet_smuggler", "squire": "greet_squire"}.get(role)
    if outsider is not None:
        return line(outsider, rng)
    if time_of_day(hour) == "night":
        return line("greet_townsfolk_night", rng)
    return line("greet_townsfolk", rng)


def warden_line(npc, moment: str, rng: random.Random) -> str:
    """A warden's shout on catching sight of an intruder (``moment`` "shout") or its words as it
    throws them out ("seize"), in its own voice: an outlaw's, the bandit king's, a knight's."""
    role = getattr(npc, "role_tag", "")
    voice = role if role in ("boss", "knight") else "bandit"
    return line(f"warden_{moment}_{voice}", rng)
