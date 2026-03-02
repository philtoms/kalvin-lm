# NLP Glossary

## Part of Speech (POS) - Universal Tags

The 17 Universal POS tags provide a language-independent categorization of word types.

| Flag          | Tag   | Description                             | Example                          |
| ------------- | ----- | --------------------------------------- | -------------------------------- |
| **POS_ADJ**   | ADJ   | Adjective - modifies nouns              | "The _red_ car stopped"          |
| **POS_ADP**   | ADP   | Adposition - prepositions/postpositions | "She walked _through_ the door"  |
| **POS_ADV**   | ADV   | Adverb - modifies verbs/adjectives      | "He ran _quickly_ home"          |
| **POS_AUX**   | AUX   | Auxiliary verb - helping verbs          | "She _is_ running late"          |
| **POS_CCONJ** | CCONJ | Coordinating conjunction                | "Cats _and_ dogs played"         |
| **POS_DET**   | DET   | Determiner - introduces nouns           | "_The_ cat sat down"             |
| **POS_INTJ**  | INTJ  | Interjection - exclamations             | "_Wow_, that's amazing!"         |
| **POS_NOUN**  | NOUN  | Common noun                             | "The _cat_ slept peacefully"     |
| **POS_NUM**   | NUM   | Numeral - numbers                       | "She has _three_ cats"           |
| **POS_PART**  | PART  | Particle - grammatical markers          | "She* 's* going home"            |
| **POS_PRON**  | PRON  | Pronoun - replaces nouns                | "_She_ loves _him_"              |
| **POS_PROPN** | PROPN | Proper noun - names                     | "_Paris_ is beautiful"           |
| **POS_PUNCT** | PUNCT | Punctuation                             | "Hello*,* world*!*"              |
| **POS_SCONJ** | SCONJ | Subordinating conjunction               | "She stayed _because_ it rained" |
| **POS_SYM**   | SYM   | Symbol                                  | "It costs *$*5"                  |
| **POS_VERB**  | VERB  | Verb - actions/states                   | "Birds _fly_ south"              |
| **POS_X**     | X     | Other - foreign words, typos            | "et cetera, _aka_"               |

## Part of Speech Fine (POS_FINE) - Penn Treebank Tags

Fine-grained POS tags provide more specific categorization within each universal category.

### Nouns

| Flag              | Tag  | Description            | Example                   |
| ----------------- | ---- | ---------------------- | ------------------------- |
| **POS_FINE_NN**   | NN   | Noun, singular or mass | "The _cat_ sleeps"        |
| **POS_FINE_NNS**  | NNS  | Noun, plural           | "Three _cats_ play"       |
| **POS_FINE_NNP**  | NNP  | Proper noun, singular  | "_Paris_ is lovely"       |
| **POS_FINE_NNPS** | NNPS | Proper noun, plural    | "The _Alps_ are majestic" |

### Verbs

| Flag             | Tag | Description                           | Example                |
| ---------------- | --- | ------------------------------------- | ---------------------- |
| **POS_FINE_VB**  | VB  | Verb, base form                       | "I can _run_ fast"     |
| **POS_FINE_VBD** | VBD | Verb, past tense                      | "She _walked_ home"    |
| **POS_FINE_VBG** | VBG | Verb, gerund/present participle       | "He is _running_ late" |
| **POS_FINE_VBN** | VBN | Verb, past participle                 | "The cake was _eaten_" |
| **POS_FINE_VBP** | VBP | Verb, non-3rd person singular present | "We _eat_ dinner now"  |
| **POS_FINE_VBZ** | VBZ | Verb, 3rd person singular present     | "She _runs_ every day" |
| **POS_FINE_MD**  | MD  | Modal auxiliary                       | "You _should_ go"      |

### Adjectives

| Flag             | Tag | Description            | Example                |
| ---------------- | --- | ---------------------- | ---------------------- |
| **POS_FINE_JJ**  | JJ  | Adjective, basic form  | "A _red_ car"          |
| **POS_FINE_JJR** | JJR | Adjective, comparative | "She is _taller_"      |
| **POS_FINE_JJS** | JJS | Adjective, superlative | "The _fastest_ runner" |

### Adverbs

| Flag             | Tag | Description         | Example               |
| ---------------- | --- | ------------------- | --------------------- |
| **POS_FINE_RB**  | RB  | Adverb, basic form  | "She ran _quickly_"   |
| **POS_FINE_RBR** | RBR | Adverb, comparative | "He ran _faster_"     |
| **POS_FINE_RBS** | RBS | Adverb, superlative | "She arrived _first_" |
| **POS_FINE_WRB** | WRB | Wh-adverb           | "_Where_ are you?"    |

### Pronouns

| Flag              | Tag  | Description           | Example                 |
| ----------------- | ---- | --------------------- | ----------------------- |
| **POS_FINE_PRP**  | PRP  | Personal pronoun      | "_She_ loves _him_"     |
| **POS_FINE_PRP$** | PRP$ | Possessive pronoun    | "_My_ cat sleeps"       |
| **POS_FINE_WP**   | WP   | Wh-pronoun            | "_Who_ is there?"       |
| **POS_FINE_WP$**  | WP$  | Possessive wh-pronoun | "_Whose_ book is this?" |

### Determiners

| Flag             | Tag | Description   | Example                 |
| ---------------- | --- | ------------- | ----------------------- |
| **POS_FINE_DT**  | DT  | Determiner    | "_The_ cat sat"         |
| **POS_FINE_PDT** | PDT | Predeterminer | "_All_ the dogs barked" |
| **POS_FINE_WDT** | WDT | Wh-determiner | "_Which_ car is yours?" |

### Conjunctions & Prepositions

| Flag            | Tag | Description                           | Example                          |
| --------------- | --- | ------------------------------------- | -------------------------------- |
| **POS_FINE_CC** | CC  | Coordinating conjunction              | "Tea _or_ coffee?"               |
| **POS_FINE_IN** | IN  | Preposition/subordinating conjunction | "She stayed _because_ it rained" |

### Numbers & Quantifiers

| Flag            | Tag | Description         | Example                |
| --------------- | --- | ------------------- | ---------------------- |
| **POS_FINE_CD** | CD  | Cardinal number     | "_Three_ cats slept"   |
| **POS_FINE_EX** | EX  | Existential _there_ | "_There_ is a problem" |

### Particles & Others

| Flag             | Tag | Description       | Example              |
| ---------------- | --- | ----------------- | -------------------- |
| **POS_FINE_POS** | POS | Possessive ending | "John* 's* car"      |
| **POS_FINE_RP**  | RP  | Particle          | "Give _up_ now"      |
| **POS_FINE_TO**  | TO  | Infinitive _to_   | "I want _to_ go"     |
| **POS_FINE_UH**  | UH  | Interjection      | "_Wow_, amazing!"    |
| **POS_FINE_FW**  | FW  | Foreign word      | "Per _se_, it works" |
| **POS_FINE_LS**  | LS  | List item marker  | "_First_, we eat"    |
| **POS_FINE_SYM** | SYM | Symbol            | "It costs *$*5"      |

## Dependency (DEP) Groups

Dependencies describe the grammatical relationships between words in a sentence.

| Group          | Dependencies                                                      | Description                                              | Example                                                    |
| -------------- | ----------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| **DEP_SUBJ**   | nsubj, nsubjpass, csubj, csubjpass, agent                         | Clause subjects - the "doer" of an action                | "_The cat_ sleeps" → "cat" is nsubj                        |
| **DEP_OBJ**    | obj, iobj, dobj                                                   | Direct/indirect objects - recipients of action           | "She gave _him_ a book" → "him" is iobj                    |
| **DEP_OBL**    | obl, obl:\*                                                       | Oblique nominals - adjunct phrases with prepositions     | "She walked _down the street_" → "street" is obl           |
| **DEP_NMOD**   | nmod, nmod:\*                                                     | Nominal modifiers - nouns modifying other nouns          | "The _city's_ lights" → "city's" is nmod                   |
| **DEP_CCOMP**  | ccomp                                                             | Clausal complements - full clauses as complements        | "She said _that she would come_" → "that...come" is ccomp  |
| **DEP_XCOMP**  | xcomp                                                             | Open clausal complements - clauses without subjects      | "She wants _to leave_" → "to leave" is xcomp               |
| **DEP_ADVCL**  | advcl                                                             | Adverbial clause modifiers - clauses modifying verbs     | "She left _when it rained_" → "when...rained" is advcl     |
| **DEP_ACL**    | acl, acl:relcl                                                    | Adnominal clause modifiers - clauses modifying nouns     | "The book _that I read_" → "that I read" is acl:relcl      |
| **DEP_AMOD**   | amod                                                              | Adjectival modifiers - adjectives modifying nouns        | "The _red_ car" → "red" is amod                            |
| **DEP_ADVMOD** | advmod                                                            | Adverbial modifiers - adverbs modifying verbs/adjectives | "She ran _quickly_" → "quickly" is advmod                  |
| **DEP_NUMMOD** | nummod, nummod:\*                                                 | Numeral modifiers - numbers modifying nouns              | "_Three_ cats" → "Three" is nummod                         |
| **DEP_APPOS**  | appos                                                             | Appositional modifiers - noun phrases renaming nouns     | "Paris, _the capital of France_" → "the...France" is appos |
| **DEP_FUNC**   | det, case, mark, aux, auxpass, cop, expl, neg                     | Function words - grammatical markers                     | "_The_ cat _is_ sleeping" → "The" is det, "is" is aux      |
| **DEP_STRUCT** | root, conj, cc, compound, flat, fixed, list, parataxis, discourse | Structural relations - sentence organization             | "Cats _and_ dogs" → "and" is cc (coord)                    |
| **DEP_PUNCT**  | punct, goeswith, reparandum, orphan                               | Punctuation and repairs                                  | "Hello*,* world" → "," is punct                            |

## Morphological (MORPH) Features

Morphological features describe grammatical properties of individual words.

| Flag               | Feature       | Description                              | Example                          |
| ------------------ | ------------- | ---------------------------------------- | -------------------------------- |
| **MORPH_SING**     | Number=Sing   | Singular number - one item               | "_cat_, _dog_, _house_"          |
| **MORPH_PLUR**     | Number=Plur   | Plural number - multiple items           | "_cats_, _dogs_, _houses_"       |
| **MORPH_PAST**     | Tense=Past    | Past tense - completed actions           | "She _walked_, _ran_, _ate_"     |
| **MORPH_PRES**     | Tense=Pres    | Present tense - current/habitual actions | "She _walks_, _runs_, _eats_"    |
| **MORPH_FUT**      | Tense=Fut     | Future tense - will happen               | "She _will walk_"                |
| **MORPH_PASS**     | Voice=Pass    | Passive voice - subject receives action  | "The cake _was eaten_"           |
| **MORPH_PERSON_1** | Person=1      | First person - speaker                   | "_I_ walk, _we_ walk"            |
| **MORPH_PERSON_2** | Person=2      | Second person - addressee                | "_You_ walk"                     |
| **MORPH_PERSON_3** | Person=3      | Third person - other entities            | "_He/she/it_ walks, _they_ walk" |
| **MORPH_PERF**     | Aspect=Perf   | Perfective aspect - completed whole      | "She _has eaten_"                |
| **MORPH_PROG**     | Aspect=Prog   | Progressive aspect - ongoing action      | "She _is eating_"                |
| **MORPH_IND**      | Mood=Ind      | Indicative mood - statements of fact     | "She _walks_ home"               |
| **MORPH_IMP**      | Mood=Imp      | Imperative mood - commands               | "_Walk_ home!"                   |
| **MORPH_INF**      | VerbForm=Inf  | Infinitive verb form                     | "to _walk_, to _eat_"            |
| **MORPH_PART**     | VerbForm=Part | Participle verb form                     | "_walking_, _eaten_, _written_"  |
| **MORPH_GER**      | VerbForm=Ger  | Gerund verb form - noun-like             | "_Swimming_ is fun"              |

## Quick Reference: Combining Features

A single word can have multiple morphological features. Examples:

| Word      | Features                       | Combined Meaning                 |
| --------- | ------------------------------ | -------------------------------- |
| "walked"  | PAST + SING                    | Past tense, singular             |
| "running" | PRES + PROG + SING             | Present progressive, singular    |
| "eaten"   | PAST + PASS + PART             | Past passive participle          |
| "am"      | PRES + IND + PERSON_1          | Present indicative, first person |
| "were"    | PAST + IND + PLUR + PERSON_2/3 | Past indicative plural           |
