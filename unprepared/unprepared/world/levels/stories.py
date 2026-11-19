"""What the people of Emberhollow tell whoever stops to talk: their own stories, a line at a time
(``NPCDef.lines`` - see ``NPC.greeting``). Their trades and their days are in their schedules;
this is who they are, what they worry about, what they know about their neighbours - and now
and then a hint worth more than it sounds.

Characters defined with lines of their own (the newer townsfolk in ``quarters.py``, the
country in ``wilds.py``) keep them; ``with_stories`` gives the rest theirs.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Sequence, Tuple

STORIES: Dict[str, Tuple[str, ...]] = {
    "Shopkeeper Barnaby": (
        "Welcome, welcome! Candlesticks, honey, rope, pickaxes. If I haven't got it, you don't need it.",
        "Somebody had the candlesticks off my shelf last month. I've had the back door lock looked at. Ludo says it's 'an education'. What does that mean?",
        "My scales are perfectly honest. Corwin says they're short by a thumb. Corwin has enormous thumbs.",
        "I keep the day's takings under the counter till I go home. Safer than any strongbox, and nobody thinks to look.",
    ),
    "Hilde the Innkeeper": (
        "The Drowsy Dragon has stood on this street since before the Vanes came to the hill, and it'll stand after.",
        "Lord Vane sends his soup back. Every time. One day I'll send it back to him with something in it.",
        "Dunstan plays here of an evening. The watch sing along. I charge them double for the privilege.",
        "Bertram's told the same story of the arrow for twenty years. Gets longer every time. The arrow, I mean.",
    ),
    "Gareth the Smith": (
        "Iron, fire and sweat. Nothing else makes a blade.",
        "I had a brother, once, who made armour for the King. He lives in the Thornwood now and pretends I'm dead. I pretend back.",
        "Bring me ore and a mallet and use the anvil yourself, if you like - picks are easy work. Mind the sparks.",
        "The old mine north-east of the walls still has iron in it. And silver, they say. And Harl. Harl's still in there too.",
    ),
    "Mirelle the Tailor": (
        "Soft soles for soft steps, if you take my meaning.",
        "Lady Ashgrove calls my velvet 'serviceable'. She buys three bolts a month of it.",
        "I don't sell black cloth. Nobody honest needs black cloth. Jae asks me every week.",
        "Pell wants a cloak he can't afford. I've told him to earn it. He looked at the Vanes' windows when I said that. I wish he hadn't.",
    ),
    "Jae the Fence": (
        "Keep your voice down. The guards don't come in here, and I'd like it to stay that way.",
        "Everything has a price and somebody to pay it. I'm just the somebody in the middle.",
        "The Vanes and the Ashgroves sit up on that hill counting their silver. Somebody should count it for them.",
        "Ludo on Upper Street sells picks to anyone with coin. He made his fortune the other way round, once.",
    ),
    "Tamsin": (
        "Thirty years married, and he goes and dies of a chill. I keep his ring in the chest. I can't bear to sell it.",
        "There was somebody creeping about on the roofs last night, I'm sure of it. Jeb says it was a cat. Jeb says everything's a cat.",
        "Sister Maude comes by every day about eleven. She means well. She brings soup. It's dreadful soup.",
    ),
    "Fenwick": (
        "Eighty-one years in Emberhollow and I've never once been to the Deep. The secret to a long life.",
        "In my day the mine was open and the town had money. Now it has Vanes.",
        "My late wife's locket is somewhere safe. So safe I can't remember where. Under the hearth, maybe.",
    ),
    "Lys": (
        "I help Hilde in the kitchen mornings. Hobb comes round the back at noon and I feed him. Somebody has to.",
        "Dorran carries crates for Barnaby. He says Barnaby counts them twice when he's not looking.",
    ),
    "Dorran": (
        "Crates, crates, crates. I could carry this town to Greyhold and back.",
        "Barnaby keeps the takings under the counter. Don't tell him I know. Don't tell anyone I told you.",
    ),
    "Pell": (
        "I'm going to have a cloak like Lord Vane's one day. A black one. Feathers.",
        "The tower guard - Aldric - he can see the whole town from up there. But he dozes after noon. Everyone knows.",
        "I found a way over the granary roof once. Don't tell Mum. I haven't got a Mum. Don't tell anyone.",
    ),
    "Guard Captain Vance": (
        "Captain Vance, the Emberhollow watch. If something's missing, I'll know who took it by sundown.",
        "The gate stays barred until I say. The roads are thick with the Deep's bandits, and the Saltcove lot aren't fishermen.",
        "Old Bertram served under my father. Best man on the walls, till the arrow. He'll tell you himself. At length.",
    ),
    "Guard Aldric": (
        "From the top of the tower I can see every roof in the town. Every one.",
        "Forty steps up, every morning. Forty down. My knees sound like Bertram's.",
    ),
    "Guard Orrin": (
        "I keep the guard house door. Nobody gets in without a key, or past me.",
        "Wynn brings bread up at noon. Best part of the day. Don't tell the captain.",
    ),
    "Guard Mira": (
        "I walk the night rounds. The streets are quiet. Too quiet, some nights.",
        "Jeb the lamplighter's out all night same as me. He sees more than he says. So do I.",
    ),
    "Guard Bram": (
        "Gate's barred till the captain says otherwise. Sorry, friend.",
        "We change sides every three hours. Keeps the eyes fresh, the captain says. Keeps the feet cold, I say.",
    ),
    "Guard Wenna": (
        "Long night on the gate. Nothing comes up that road but wind.",
        "Smugglers' lanterns off the Saltcove again. Not my gate, not my problem.",
    ),
    "Jailer Hesk": (
        "The cells are empty. I'd like them to stay that way, but I'd like it better if they were full of the right people.",
        "What the watch takes off you goes in the lockbox. You get it back when you've done your time. Mostly.",
    ),
}


def with_stories(npcs: Sequence) -> tuple:
    """The level's people with their stories: each that has none of its own gets its entry here."""
    return tuple(dataclasses.replace(n, lines=STORIES[n.name]) if n.name in STORIES and not n.lines else n for n in npcs)
