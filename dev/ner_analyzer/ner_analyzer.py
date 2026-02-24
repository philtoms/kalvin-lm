#!/usr/bin/env python3
"""
Text Analyzer - Extract named entities and grammatical tags from text files using spaCy.

Reads a JSON file containing text, performs linguistic analysis (NER, POS, noun chunks,
verb lemmas), and saves the results as dictionaries.

GPU Support:
- Apple Silicon: Install with spacy[apple] for MPS acceleration
- CUDA: Install with spacy[cuda11x] for NVIDIA GPU acceleration
- Use --gpu flag to enable GPU acceleration
- Use --model en_core_web_trf for transformer-based model (better GPU utilization)
"""

import argparse
import json
from dataclasses import dataclass, field
from enum import IntFlag
from pathlib import Path

import spacy
from tqdm import tqdm


class NLPType32(IntFlag):
    """
    32-bit NLP type encoding with coarse-grained features.

    Bit layout:
    - Bits 0-16: Coarse POS tags (17 Universal POS tags)
    - Bits 17-24: Simplified dependency groups (8 groups)
    - Bits 25-31: Simplified morphological features (7 features)
    """

    # Coarse POS tags (bits 0-16) - Universal POS Tags
    POS_ADJ = 1 << 0      # Adjective
    POS_ADP = 1 << 1      # Adposition
    POS_ADV = 1 << 2      # Adverb
    POS_AUX = 1 << 3      # Auxiliary
    POS_CCONJ = 1 << 4    # Coordinating conjunction
    POS_DET = 1 << 5      # Determiner
    POS_INTJ = 1 << 6     # Interjection
    POS_NOUN = 1 << 7     # Noun
    POS_NUM = 1 << 8      # Numeral
    POS_PART = 1 << 9     # Particle
    POS_PRON = 1 << 10    # Pronoun
    POS_PROPN = 1 << 11   # Proper noun
    POS_PUNCT = 1 << 12   # Punctuation
    POS_SCONJ = 1 << 13   # Subordinating conjunction
    POS_SYM = 1 << 14     # Symbol
    POS_VERB = 1 << 15    # Verb
    POS_X = 1 << 16       # Other

    # Simplified dependency groups (bits 17-24)
    DEP_SUBJ = 1 << 17    # Subjects: nsubj, nsubjpass, csubj, csubjpass, agent
    DEP_OBJ = 1 << 18     # Objects: obj, iobj, dobj
    DEP_OBL = 1 << 19     # Oblique/adjunct: obl, iobl, nmod (non-possessive)
    DEP_COMP = 1 << 20    # Complements: ccomp, xcomp, advcl, acl, relcl
    DEP_MOD = 1 << 21     # Modifiers: amod, advmod, nummod, appos
    DEP_FUNC = 1 << 22    # Function words: det, case, mark, aux, cop, expl
    DEP_STRUCT = 1 << 23  # Structure: root, conj, cc, compound, flat, fixed, list
    DEP_PUNCT = 1 << 24   # Punctuation/other: punct, goeswith, reparandum

    # Simplified morphological features (bits 25-31)
    MORPH_SING = 1 << 25      # Number=Sing
    MORPH_PLUR = 1 << 26      # Number=Plur
    MORPH_PAST = 1 << 27      # Tense=Past
    MORPH_PRES = 1 << 28      # Tense=Pres
    MORPH_PASS = 1 << 29      # Voice=Pass
    MORPH_PERSON_3 = 1 << 30  # Person=3
    MORPH_PERF = 1 << 31      # Aspect=Perf


class NLPType48(IntFlag):
    """
    48-bit NLP type encoding with finer-grained features.

    Bit layout:
    - Bits 0-16: Coarse POS tags (17 Universal POS tags, same as NLPType32)
    - Bits 17-31: Finer dependency groups (15 groups)
    - Bits 32-47: Finer morphological features (16 features)
    """

    # Coarse POS tags (bits 0-16) - Universal POS Tags (same as NLPType32)
    POS_ADJ = 1 << 0      # Adjective
    POS_ADP = 1 << 1      # Adposition
    POS_ADV = 1 << 2      # Adverb
    POS_AUX = 1 << 3      # Auxiliary
    POS_CCONJ = 1 << 4    # Coordinating conjunction
    POS_DET = 1 << 5      # Determiner
    POS_INTJ = 1 << 6     # Interjection
    POS_NOUN = 1 << 7     # Noun
    POS_NUM = 1 << 8      # Numeral
    POS_PART = 1 << 9     # Particle
    POS_PRON = 1 << 10    # Pronoun
    POS_PROPN = 1 << 11   # Proper noun
    POS_PUNCT = 1 << 12   # Punctuation
    POS_SCONJ = 1 << 13   # Subordinating conjunction
    POS_SYM = 1 << 14     # Symbol
    POS_VERB = 1 << 15    # Verb
    POS_X = 1 << 16       # Other

    # Finer dependency groups (bits 17-31)
    DEP_SUBJ = 1 << 17    # Subjects: nsubj, nsubjpass, csubj, csubjpass, agent
    DEP_OBJ = 1 << 18     # Objects: obj, iobj, dobj
    DEP_OBL = 1 << 19     # Oblique: obl, obl:*
    DEP_NMOD = 1 << 20    # Nominal modifier: nmod, nmod:*
    DEP_CCOMP = 1 << 21   # Clausal complement: ccomp
    DEP_XCOMP = 1 << 22   # Open clausal complement: xcomp
    DEP_ADVCL = 1 << 23   # Adverbial clause: advcl
    DEP_ACL = 1 << 24     # Adnominal clause: acl, acl:relcl
    DEP_AMOD = 1 << 25    # Adjectival modifier: amod
    DEP_ADVMOD = 1 << 26  # Adverbial modifier: advmod
    DEP_NUMMOD = 1 << 27  # Numeral modifier: nummod, nummod:*
    DEP_APPOS = 1 << 28   # Apposition: appos
    DEP_FUNC = 1 << 29    # Function words: det, case, mark, aux, auxpass, cop, expl, neg
    DEP_STRUCT = 1 << 30  # Structure: root, conj, cc, compound, flat, fixed, list, parataxis, discourse
    DEP_PUNCT = 1 << 31   # Punctuation: punct, goeswith, reparandum, orphan

    # Finer morphological features (bits 32-47)
    MORPH_SING = 1 << 32      # Number=Sing
    MORPH_PLUR = 1 << 33      # Number=Plur
    MORPH_PAST = 1 << 34      # Tense=Past
    MORPH_PRES = 1 << 35      # Tense=Pres
    MORPH_FUT = 1 << 36       # Tense=Fut
    MORPH_PASS = 1 << 37      # Voice=Pass
    MORPH_PERSON_1 = 1 << 38  # Person=1
    MORPH_PERSON_2 = 1 << 39  # Person=2
    MORPH_PERSON_3 = 1 << 40  # Person=3
    MORPH_PERF = 1 << 41      # Aspect=Perf
    MORPH_PROG = 1 << 42      # Aspect=Prog
    MORPH_IND = 1 << 43       # Mood=Ind
    MORPH_IMP = 1 << 44       # Mood=Imp
    MORPH_INF = 1 << 45       # VerbForm=Inf
    MORPH_PART = 1 << 46      # VerbForm=Part
    MORPH_GER = 1 << 47       # VerbForm=Ger


# Mapping from fine-grained spaCy dependencies to 32-bit coarse DEP groups
DEP_TO_COARSE32: dict[str, int] = {
    # Subjects
    "nsubj": NLPType32.DEP_SUBJ,
    "nsubjpass": NLPType32.DEP_SUBJ,
    "csubj": NLPType32.DEP_SUBJ,
    "csubjpass": NLPType32.DEP_SUBJ,
    "agent": NLPType32.DEP_SUBJ,

    # Objects
    "obj": NLPType32.DEP_OBJ,
    "iobj": NLPType32.DEP_OBJ,
    "dobj": NLPType32.DEP_OBJ,

    # Oblique/adjunct
    "obl": NLPType32.DEP_OBL,
    "obl:agent": NLPType32.DEP_OBL,
    "obl:tmod": NLPType32.DEP_OBL,
    "iobl": NLPType32.DEP_OBL,
    "nmod": NLPType32.DEP_OBL,
    "nmod:npmod": NLPType32.DEP_OBL,
    "nmod:tmod": NLPType32.DEP_OBL,
    "nmod:poss": NLPType32.DEP_OBL,

    # Complements
    "ccomp": NLPType32.DEP_COMP,
    "xcomp": NLPType32.DEP_COMP,
    "advcl": NLPType32.DEP_COMP,
    "acl": NLPType32.DEP_COMP,
    "acl:relcl": NLPType32.DEP_COMP,
    "relcl": NLPType32.DEP_COMP,

    # Modifiers
    "amod": NLPType32.DEP_MOD,
    "advmod": NLPType32.DEP_MOD,
    "nummod": NLPType32.DEP_MOD,
    "nummod:gov": NLPType32.DEP_MOD,
    "nummod:entity": NLPType32.DEP_MOD,
    "appos": NLPType32.DEP_MOD,

    # Function words
    "det": NLPType32.DEP_FUNC,
    "det:predet": NLPType32.DEP_FUNC,
    "case": NLPType32.DEP_FUNC,
    "mark": NLPType32.DEP_FUNC,
    "aux": NLPType32.DEP_FUNC,
    "auxpass": NLPType32.DEP_FUNC,
    "cop": NLPType32.DEP_FUNC,
    "expl": NLPType32.DEP_FUNC,
    "neg": NLPType32.DEP_FUNC,

    # Structure
    "root": NLPType32.DEP_STRUCT,
    "conj": NLPType32.DEP_STRUCT,
    "cc": NLPType32.DEP_STRUCT,
    "cc:preconj": NLPType32.DEP_STRUCT,
    "compound": NLPType32.DEP_STRUCT,
    "compound:prt": NLPType32.DEP_STRUCT,
    "flat": NLPType32.DEP_STRUCT,
    "flat:foreign": NLPType32.DEP_STRUCT,
    "fixed": NLPType32.DEP_STRUCT,
    "list": NLPType32.DEP_STRUCT,
    "parataxis": NLPType32.DEP_STRUCT,
    "discourse": NLPType32.DEP_STRUCT,
    "vocative": NLPType32.DEP_STRUCT,
    "dislocated": NLPType32.DEP_STRUCT,

    # Punctuation/other
    "punct": NLPType32.DEP_PUNCT,
    "goeswith": NLPType32.DEP_PUNCT,
    "reparandum": NLPType32.DEP_PUNCT,
    "orphan": NLPType32.DEP_PUNCT,
}

# Mapping from fine-grained morph features to 32-bit coarse MORPH flags
MORPH_TO_COARSE32: dict[str, int] = {
    # Number
    "Number=Sing": NLPType32.MORPH_SING,
    "Number=Plur": NLPType32.MORPH_PLUR,

    # Tense
    "Tense=Past": NLPType32.MORPH_PAST,
    "Tense=Pres": NLPType32.MORPH_PRES,

    # Voice
    "Voice=Pass": NLPType32.MORPH_PASS,

    # Person
    "Person=3": NLPType32.MORPH_PERSON_3,

    # Aspect
    "Aspect=Perf": NLPType32.MORPH_PERF,
}

# POS tag to NLPType32 mapping (shared between 32-bit and 48-bit)
POS_TO_COARSE: dict[str, int] = {
    "ADJ": NLPType32.POS_ADJ,
    "ADP": NLPType32.POS_ADP,
    "ADV": NLPType32.POS_ADV,
    "AUX": NLPType32.POS_AUX,
    "CCONJ": NLPType32.POS_CCONJ,
    "DET": NLPType32.POS_DET,
    "INTJ": NLPType32.POS_INTJ,
    "NOUN": NLPType32.POS_NOUN,
    "NUM": NLPType32.POS_NUM,
    "PART": NLPType32.POS_PART,
    "PRON": NLPType32.POS_PRON,
    "PROPN": NLPType32.POS_PROPN,
    "PUNCT": NLPType32.POS_PUNCT,
    "SCONJ": NLPType32.POS_SCONJ,
    "SYM": NLPType32.POS_SYM,
    "VERB": NLPType32.POS_VERB,
    "X": NLPType32.POS_X,
}


# ============================================================================
# 48-bit NLP type mappings (finer-grained)
# ============================================================================

# Mapping from fine-grained spaCy dependencies to 48-bit finer DEP groups
DEP_TO_COARSE48: dict[str, int] = {
    # Subjects
    "nsubj": NLPType48.DEP_SUBJ,
    "nsubjpass": NLPType48.DEP_SUBJ,
    "csubj": NLPType48.DEP_SUBJ,
    "csubjpass": NLPType48.DEP_SUBJ,
    "agent": NLPType48.DEP_SUBJ,

    # Objects
    "obj": NLPType48.DEP_OBJ,
    "iobj": NLPType48.DEP_OBJ,
    "dobj": NLPType48.DEP_OBJ,

    # Oblique
    "obl": NLPType48.DEP_OBL,
    "obl:agent": NLPType48.DEP_OBL,
    "obl:tmod": NLPType48.DEP_OBL,

    # Nominal modifiers
    "nmod": NLPType48.DEP_NMOD,
    "nmod:npmod": NLPType48.DEP_NMOD,
    "nmod:tmod": NLPType48.DEP_NMOD,
    "nmod:poss": NLPType48.DEP_NMOD,

    # Clausal complements
    "ccomp": NLPType48.DEP_CCOMP,
    "xcomp": NLPType48.DEP_XCOMP,

    # Clauses
    "advcl": NLPType48.DEP_ADVCL,
    "acl": NLPType48.DEP_ACL,
    "acl:relcl": NLPType48.DEP_ACL,
    "relcl": NLPType48.DEP_ACL,

    # Modifiers (separate groups)
    "amod": NLPType48.DEP_AMOD,
    "advmod": NLPType48.DEP_ADVMOD,
    "nummod": NLPType48.DEP_NUMMOD,
    "nummod:gov": NLPType48.DEP_NUMMOD,
    "nummod:entity": NLPType48.DEP_NUMMOD,
    "appos": NLPType48.DEP_APPOS,

    # Function words
    "det": NLPType48.DEP_FUNC,
    "det:predet": NLPType48.DEP_FUNC,
    "case": NLPType48.DEP_FUNC,
    "mark": NLPType48.DEP_FUNC,
    "aux": NLPType48.DEP_FUNC,
    "auxpass": NLPType48.DEP_FUNC,
    "cop": NLPType48.DEP_FUNC,
    "expl": NLPType48.DEP_FUNC,
    "neg": NLPType48.DEP_FUNC,

    # Structure
    "root": NLPType48.DEP_STRUCT,
    "conj": NLPType48.DEP_STRUCT,
    "cc": NLPType48.DEP_STRUCT,
    "cc:preconj": NLPType48.DEP_STRUCT,
    "compound": NLPType48.DEP_STRUCT,
    "compound:prt": NLPType48.DEP_STRUCT,
    "flat": NLPType48.DEP_STRUCT,
    "flat:foreign": NLPType48.DEP_STRUCT,
    "fixed": NLPType48.DEP_STRUCT,
    "list": NLPType48.DEP_STRUCT,
    "parataxis": NLPType48.DEP_STRUCT,
    "discourse": NLPType48.DEP_STRUCT,
    "vocative": NLPType48.DEP_STRUCT,
    "dislocated": NLPType48.DEP_STRUCT,

    # Punctuation/other
    "punct": NLPType48.DEP_PUNCT,
    "goeswith": NLPType48.DEP_PUNCT,
    "reparandum": NLPType48.DEP_PUNCT,
    "orphan": NLPType48.DEP_PUNCT,
}

# Mapping from fine-grained morph features to 48-bit finer MORPH flags
MORPH_TO_COARSE48: dict[str, int] = {
    # Number
    "Number=Sing": NLPType48.MORPH_SING,
    "Number=Plur": NLPType48.MORPH_PLUR,

    # Tense
    "Tense=Past": NLPType48.MORPH_PAST,
    "Tense=Pres": NLPType48.MORPH_PRES,
    "Tense=Fut": NLPType48.MORPH_FUT,

    # Voice
    "Voice=Pass": NLPType48.MORPH_PASS,

    # Person
    "Person=1": NLPType48.MORPH_PERSON_1,
    "Person=2": NLPType48.MORPH_PERSON_2,
    "Person=3": NLPType48.MORPH_PERSON_3,

    # Aspect
    "Aspect=Perf": NLPType48.MORPH_PERF,
    "Aspect=Prog": NLPType48.MORPH_PROG,

    # Mood
    "Mood=Ind": NLPType48.MORPH_IND,
    "Mood=Imp": NLPType48.MORPH_IMP,

    # VerbForm
    "VerbForm=Inf": NLPType48.MORPH_INF,
    "VerbForm=Part": NLPType48.MORPH_PART,
    "VerbForm=Ger": NLPType48.MORPH_GER,
}


def compute_nlp_type32(pos: str, dep: str, morph: str) -> int:
    """
    Compute 32-bit NLP type from coarse-grained features.

    Args:
        pos: Coarse POS tag (e.g., "NOUN", "VERB")
        dep: Dependency label (e.g., "nsubj", "obj")
        morph: Pipe-separated morph features (e.g., "Number=Sing|Tense=Past")

    Returns:
        32-bit integer encoding all features.
    """
    nlp_type = 0

    # Map POS
    if pos in POS_TO_COARSE:
        nlp_type |= int(POS_TO_COARSE[pos])

    # Map dependency (use coarse group)
    if dep in DEP_TO_COARSE32:
        nlp_type |= int(DEP_TO_COARSE32[dep])
    elif dep:  # Unknown dependency maps to STRUCT
        nlp_type |= int(NLPType32.DEP_STRUCT)

    # Map morph features
    if morph:
        for feature in morph.split("|"):
            feature = feature.strip()
            if feature in MORPH_TO_COARSE32:
                nlp_type |= int(MORPH_TO_COARSE32[feature])

    return nlp_type


def compute_nlp_type48(pos: str, dep: str, morph: str) -> int:
    """
    Compute 48-bit NLP type from finer-grained features.

    Args:
        pos: Coarse POS tag (e.g., "NOUN", "VERB")
        dep: Dependency label (e.g., "nsubj", "obj")
        morph: Pipe-separated morph features (e.g., "Number=Sing|Tense=Past")

    Returns:
        48-bit integer encoding all features.
    """
    nlp_type = 0

    # Map POS (same as 32-bit)
    if pos in POS_TO_COARSE:
        nlp_type |= int(POS_TO_COARSE[pos])

    # Map dependency (use finer groups)
    if dep in DEP_TO_COARSE48:
        nlp_type |= int(DEP_TO_COARSE48[dep])
    elif dep:  # Unknown dependency maps to STRUCT
        nlp_type |= int(NLPType48.DEP_STRUCT)

    # Map morph features (finer)
    if morph:
        for feature in morph.split("|"):
            feature = feature.strip()
            if feature in MORPH_TO_COARSE48:
                nlp_type |= int(MORPH_TO_COARSE48[feature])

    return nlp_type


def get_nlp_type32_legend() -> dict[str, int]:
    """Return legend mapping flag names to bit values for NLPType32."""
    return {flag.name: int(flag.value) for flag in NLPType32}


def get_nlp_type48_legend() -> dict[str, int]:
    """Return legend mapping flag names to bit values for NLPType48."""
    return {flag.name: int(flag.value) for flag in NLPType48}


# ============================================================================
# Fine-grained NLP type system (dynamic, unlimited bits)
# ============================================================================

@dataclass
class NLPFineTypeRegistry:
    """
    Registry for collecting unique NLP feature values during processing.

    After all texts are processed, this is used to build the NLPType IntFlag
    and assign bit patterns to each word.
    """
    pos: set[str] = field(default_factory=set)
    pos_fine: set[str] = field(default_factory=set)
    dep: set[str] = field(default_factory=set)
    morph: set[str] = field(default_factory=set)  # Individual morph features like "Number=Sing"
    # Lookup maps (populated after build_nlp_type_class is called)
    _feature_map: dict = field(default_factory=dict)  # Maps (category, value) -> flag_name
    _name_map: dict = field(default_factory=dict)  # Maps flag_name -> (category, value)

    def add_token(self, pos: str, pos_fine: str, dep: str, morph: str) -> None:
        """Add token features to the registry."""
        if pos:
            self.pos.add(pos)
        if pos_fine:
            self.pos_fine.add(pos_fine)
        if dep:
            self.dep.add(dep)
        # Parse morph features (e.g., "Number=Sing|Tense=Past" -> ["Number=Sing", "Tense=Past"])
        if morph:
            for feature in morph.split("|"):
                if feature.strip():
                    self.morph.add(feature.strip())

    def build_nlp_fine_type_class(self) -> type:
        """
        Dynamically build an IntFlag class with all discovered NLP features.

        Returns:
            A dynamically created IntFlag class with flags for each feature.
        """
        # Collect all features with their prefix
        all_features: list[tuple[str, str]] = []  # (flag_name, display_name)

        # Add POS tags
        for val in sorted(self.pos):
            flag_name = f"POS_{val}"
            all_features.append((flag_name, val))

        # Add fine POS tags
        for val in sorted(self.pos_fine):
            flag_name = f"POS_FINE_{val}"
            all_features.append((flag_name, val))

        # Add dependency labels
        for val in sorted(self.dep):
            flag_name = f"DEP_{val.upper()}"
            all_features.append((flag_name, val))

        # Add morph features (convert "Number=Sing" to "MORPH_NUMBER_SING")
        for val in sorted(self.morph):
            if "=" in val:
                key, value = val.split("=", 1)
                flag_name = f"MORPH_{key.upper()}_{value.upper()}"
            else:
                flag_name = f"MORPH_{val.upper()}"
            all_features.append((flag_name, val))

        # Create IntFlag using functional API
        # IntFlag('ClassName', [('NAME1', 1), ('NAME2', 2), ...])
        flag_values = [(flag_name, 1 << i) for i, (flag_name, _) in enumerate(all_features)]
        NLPFineType = IntFlag("NLPFineType", flag_values)

        # Build lookup maps and store in registry (not on the class itself)
        self._feature_map = {}  # Maps (category, value) -> flag_name
        self._name_map = {}  # Maps flag_name -> (category, value)

        for flag_name, display_name in all_features:
            # Determine category from flag name prefix
            if flag_name.startswith("POS_FINE_"):
                category = "pos_fine"
            elif flag_name.startswith("POS_"):
                category = "pos"
            elif flag_name.startswith("DEP_"):
                category = "dep"
            else:
                category = "morph"
            self._feature_map[(category, display_name)] = flag_name
            self._name_map[flag_name] = (category, display_name)

        return NLPFineType

    def get_flag_value(self, NLPType: type, category: str, value: str) -> int:
        """Get the bit value for a given category and value."""
        key = (category, value)
        if key in self._feature_map:
            flag_name = self._feature_map[key]
            return int(getattr(NLPType, flag_name))
        return 0

    def total_flags(self) -> int:
        """Return total number of unique flags that will be created."""
        return len(self.pos) + len(self.pos_fine) + len(self.dep) + len(self.morph)


def setup_gpu(use_gpu: bool, verbose: bool = False) -> bool:
    """
    Configure spaCy to use GPU if available.

    Returns:
        True if GPU is being used, False otherwise.
    """
    if not use_gpu:
        return False

    # Try to enable GPU
    gpu_available = spacy.prefer_gpu(0)  # type: ignore[attr-defined]
    if gpu_available:
        if verbose:
            print("GPU acceleration enabled")
        return True
    else:
        if verbose:
            print("GPU not available, using CPU")
        return False


def load_texts_from_json(file_path: Path) -> list[str]:
    """Load texts from JSON file. Handles array of objects with 'summary' field."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # Extract 'summary' field from each object
        return [item.get("summary", "") for item in data if "summary" in item]
    elif isinstance(data, dict) and "summary" in data:
        return [data["summary"]]
    elif isinstance(data, dict) and "summaries" in data:
        return [item.get("summary", "") for item in data["summaries"] if "summary" in item]
    elif isinstance(data, str):
        return [data]
    else:
        raise ValueError(f"Unsupported JSON structure in {file_path}")


def load_texts_from_file(file_path: Path) -> list[str]:
    """Load texts from a file (JSON or plain text)."""
    if file_path.suffix.lower() == ".json":
        return load_texts_from_json(file_path)
    else:
        with open(file_path, encoding="utf-8") as f:
            return [f.read()]


def load_existing_dict(file_path: Path) -> dict | None:
    """Load an existing dictionary from a JSON file."""
    if not file_path.exists():
        return None
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def load_existing_grammar(file_path: Path) -> dict[str, dict]:
    """Load existing grammar dictionary from JSON file."""
    data = load_existing_dict(file_path)
    if data is None:
        return {}
    return {k: v for k, v in data.items()}  # Type-safe conversion


def load_existing_count_dict(file_path: Path) -> dict[str, int]:
    """Load existing count dictionary (verbs, noun_chunks) from JSON file."""
    data = load_existing_dict(file_path)
    if data is None:
        return {}
    return {k: int(v) for k, v in data.items()}


@dataclass
class LinguisticAnalysis:
    """Container for all linguistic analysis results."""
    # Named entities: entity text -> NER label
    ner: dict[str, str] = field(default_factory=dict)
    # Combined grammatical table: word -> {pos, pos_fine, dep, morph, count, frequency_pct, nlp_type, nlp_fine_type}
    grammar: dict[str, dict] = field(default_factory=dict)
    # Noun chunks: chunk text -> count
    noun_chunks: dict[str, int] = field(default_factory=dict)
    # Verb lemmas: lemma -> count
    verbs: dict[str, int] = field(default_factory=dict)
    # Total word count for frequency calculation
    total_word_count: int = 0
    # Registry for fine-grained NLP feature types (collected during processing)
    nlp_fine_registry: NLPFineTypeRegistry = field(default_factory=NLPFineTypeRegistry)
    # The dynamically built NLPFineType class (set after processing)
    NLPFineType: type | None = None
    # Stats for tracking new entries
    new_grammar_entries: int = 0
    new_ner_entries: int = 0
    new_verb_entries: int = 0
    new_noun_chunk_entries: int = 0

    def top_frequency(self, n: int = 100) -> list[tuple[str, int, float]]:
        """
        Return top N most frequent words.

        Returns:
            List of (word, count, frequency_pct) tuples sorted by count descending.
        """
        items = [
            (word, data["count"], data["frequency_pct"])
            for word, data in self.grammar.items()
        ]
        return sorted(items, key=lambda x: -x[1])[:n]

    def build_nlp_types(self) -> None:
        """
        Build NLP type bit patterns for all words.

        Computes both:
        - nlp_type: 32-bit coarse-grained encoding (hardcoded)
        - nlp_fine_type: Dynamic fine-grained encoding (unlimited bits)

        Must be called after all texts have been processed.
        """
        # Build the fine-grained IntFlag class from collected features
        self.NLPFineType = self.nlp_fine_registry.build_nlp_fine_type_class()

        # Assign nlp_type32, nlp_type48, and nlp_fine_type to each word in grammar
        for word, data in self.grammar.items():
            pos = data.get("pos", "")
            pos_fine = data.get("pos_fine", "")
            dep = data.get("dep", "")
            morph = data.get("morph", "")

            # Compute 32-bit coarse nlp_type32
            data["nlp_type32"] = compute_nlp_type32(pos, dep, morph)

            # Compute 48-bit finer nlp_type48
            data["nlp_type48"] = compute_nlp_type48(pos, dep, morph)

            # Compute fine-grained nlp_fine_type
            nlp_fine_type = 0
            if pos:
                nlp_fine_type |= self.nlp_fine_registry.get_flag_value(self.NLPFineType, "pos", pos)
            if pos_fine:
                nlp_fine_type |= self.nlp_fine_registry.get_flag_value(self.NLPFineType, "pos_fine", pos_fine)
            if dep:
                nlp_fine_type |= self.nlp_fine_registry.get_flag_value(self.NLPFineType, "dep", dep)
            if morph:
                for feature in morph.split("|"):
                    feature = feature.strip()
                    if feature:
                        nlp_fine_type |= self.nlp_fine_registry.get_flag_value(self.NLPFineType, "morph", feature)

            data["nlp_fine_type"] = nlp_fine_type

    def get_nlp_type32_legend(self) -> dict[str, int]:
        """
        Return legend for 32-bit NLPType32 flags.

        Useful for decoding nlp_type32 values.
        """
        return get_nlp_type32_legend()

    def get_nlp_type48_legend(self) -> dict[str, int]:
        """
        Return legend for 48-bit NLPType48 flags.

        Useful for decoding nlp_type48 values.
        """
        return get_nlp_type48_legend()

    def get_nlp_fine_type_legend(self) -> dict[str, int]:
        """
        Return legend for dynamic fine-grained NLPFineType flags.

        Useful for decoding nlp_fine_type values.
        """
        if self.NLPFineType is None:
            return {}
        legend = {}
        for flag_name in self.nlp_fine_registry._name_map:
            legend[flag_name] = int(getattr(self.NLPFineType, flag_name))
        return legend


def analyze_texts(
    nlp: spacy.Language,
    texts: list[str],
    batch_size: int = 50,
    verbose: bool = False,
    existing_grammar: dict[str, dict] | None = None,
    existing_ner: dict[str, str] | None = None,
    existing_verbs: dict[str, int] | None = None,
    existing_noun_chunks: dict[str, int] | None = None,
) -> LinguisticAnalysis:
    """
    Perform comprehensive linguistic analysis on texts using spaCy.

    Extracts NER, POS tags, noun chunks, verb lemmas, dependency labels, and morphology.
    Can extend existing dictionaries with new entries.
    """
    analysis = LinguisticAnalysis()

    # Initialize with existing dictionaries if provided
    if existing_grammar:
        analysis.grammar = existing_grammar.copy()
    if existing_ner:
        analysis.ner = existing_ner.copy()
    if existing_verbs:
        analysis.verbs = existing_verbs.copy()
    if existing_noun_chunks:
        analysis.noun_chunks = existing_noun_chunks.copy()

    # Use nlp.pipe for efficient batch processing
    iterator = tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts), disable=not verbose)

    for doc in iterator:
        # Extract named entities
        for ent in doc.ents:
            if ent.text not in analysis.ner:
                analysis.ner[ent.text] = ent.label_
                analysis.new_ner_entries += 1

        # Extract grammatical info for each token into combined table
        for token in doc:
            # Skip whitespace-only tokens
            if not token.text.strip():
                continue

            analysis.total_word_count += 1

            # Get morph string
            morph_str = str(token.morph) if token.morph else ""

            # Register features for NLPFineType building
            analysis.nlp_fine_registry.add_token(
                pos=token.pos_,
                pos_fine=token.tag_,
                dep=token.dep_,
                morph=morph_str
            )

            # Add or update word in grammar dictionary
            if token.text not in analysis.grammar:
                analysis.grammar[token.text] = {
                    "pos": token.pos_,
                    "pos_fine": token.tag_,
                    "dep": token.dep_,
                    "morph": morph_str,
                    "count": 1,
                }
                analysis.new_grammar_entries += 1
            else:
                analysis.grammar[token.text]["count"] += 1

            # Verb lemmas (always increment count, but track new ones)
            if token.pos_ == "VERB":
                if token.lemma_ not in analysis.verbs:
                    analysis.verbs[token.lemma_] = 1
                    analysis.new_verb_entries += 1
                else:
                    analysis.verbs[token.lemma_] += 1

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if chunk.text not in analysis.noun_chunks:
                analysis.noun_chunks[chunk.text] = 1
                analysis.new_noun_chunk_entries += 1
            else:
                analysis.noun_chunks[chunk.text] += 1

    # Calculate frequency percentages
    if analysis.total_word_count > 0:
        for word_data in analysis.grammar.values():
            count = word_data.get("count", 0)
            word_data["frequency_pct"] = round((count / analysis.total_word_count) * 100, 4)

    # Build NLPType IntFlag and assign bit patterns
    analysis.build_nlp_types()

    return analysis


def save_as_json_binary(data: dict, output_path: Path) -> None:
    """Save dictionary as JSON in binary format."""
    with open(output_path, "wb") as f:
        json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        f.write(json_bytes)


def save_analysis(analysis: LinguisticAnalysis, output: Path, stem: str) -> dict[str, Path]:
    """Save all analysis results to separate JSON files."""
    saved_files = {}

    files_to_save = {
        "ner": analysis.ner,
        "grammar": analysis.grammar,
        "noun_chunks": analysis.noun_chunks,
        "verbs": analysis.verbs,
    }

    for name, data in files_to_save.items():
        if data:  # Only save non-empty dictionaries
            output_path = output / f"{stem}_{name}.json"
            save_as_json_binary(data, output_path)
            saved_files[name] = output_path

    # Save 32-bit NLP type legend (coarse flags)
    nlp_type32_legend = analysis.get_nlp_type32_legend()
    if nlp_type32_legend:
        legend_path = output / f"{stem}_nlp_type32.json"
        save_as_json_binary(nlp_type32_legend, legend_path)
        saved_files["nlp_type32"] = legend_path

    # Save 48-bit NLP type legend (finer flags)
    nlp_type48_legend = analysis.get_nlp_type48_legend()
    if nlp_type48_legend:
        legend_path = output / f"{stem}_nlp_type48.json"
        save_as_json_binary(nlp_type48_legend, legend_path)
        saved_files["nlp_type48"] = legend_path

    # Save fine-grained NLP type legend (dynamic flags)
    nlp_fine_legend = analysis.get_nlp_fine_type_legend()
    if nlp_fine_legend:
        legend_path = output / f"{stem}_nlp_fine_types.json"
        save_as_json_binary(nlp_fine_legend, legend_path)
        saved_files["nlp_fine_types"] = legend_path

    return saved_files


def print_summary(analysis: LinguisticAnalysis) -> None:
    """Print summary statistics for the analysis."""
    print("\nAnalysis Summary:")
    print(f"  Total Words: {analysis.total_word_count:,}")
    print(f"  Named Entities (NER): {len(analysis.ner)} (+{analysis.new_ner_entries} new)")
    print(f"  Unique Words (Grammar): {len(analysis.grammar)} (+{analysis.new_grammar_entries} new)")
    print(f"  Noun Chunks: {len(analysis.noun_chunks)} (+{analysis.new_noun_chunk_entries} new)")
    print(f"  Verb Lemmas: {len(analysis.verbs)} (+{analysis.new_verb_entries} new)")

    # NLP fine type stats
    total_flags = analysis.nlp_fine_registry.total_flags()
    print(f"  NLP Fine Type Flags: {total_flags} unique features")

    # NER type breakdown
    if analysis.ner:
        ner_counts: dict[str, int] = {}
        for label in analysis.ner.values():
            ner_counts[label] = ner_counts.get(label, 0) + 1
        print("\n  NER type breakdown:")
        for label, count in sorted(ner_counts.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")

    # POS tag breakdown (from combined grammar table)
    if analysis.grammar:
        pos_counts: dict[str, int] = {}
        for entry in analysis.grammar.values():
            pos = entry.get("pos", "")
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        print("\n  POS tag breakdown:")
        for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {pos}: {count}")

    # Top frequency words
    if analysis.grammar:
        print("\n  Top 15 words by frequency:")
        for word, count, freq_pct in analysis.top_frequency(15):
            print(f"    {word}: {count} ({freq_pct}%)")

    # Top verbs
    if analysis.verbs:
        print("\n  Top 15 verbs:")
        for lemma, count in sorted(analysis.verbs.items(), key=lambda x: -x[1])[:15]:
            print(f"    {lemma}: {count}")

    # Top noun chunks
    if analysis.noun_chunks:
        print("\n  Top 15 noun chunks:")
        for chunk, count in sorted(analysis.noun_chunks.items(), key=lambda x: -x[1])[:15]:
            print(f"    {chunk}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze text file with spaCy and generate linguistic dictionaries"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("/Volumes/USB-Backup/ai/data/tidy-ts/tinystories.v1.summary.json"),
        help="Input text or JSON file (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for JSON files (default: same as input file)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use (default: %(default)s). Use 'en_core_web_trf' for transformer model with better GPU utilization."
    )
    parser.add_argument(
        "--gpu", "-g",
        action="store_true",
        help="Enable GPU acceleration (requires spacy[apple] for macOS or spacy[cuda11x] for NVIDIA)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "-e", "--existing",
        type=Path,
        default=None,
        help="Existing text or JSON file"
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output is None:
        args.output = args.input.parent

    # Load existing dictionaries if --existing flag is set
    existing_grammar = None
    existing_ner = None
    existing_verbs = None
    existing_noun_chunks = None

    if args.existing and not args.existing.isdir():
        stem = args.existing.stem if args.existing else args.input.stem
        base_dir = args.existing.parent

        grammar_path = base_dir / f"{stem}_grammar.json"
        ner_path = base_dir / f"{stem}_ner.json"
        verbs_path = base_dir / f"{stem}_verbs.json"
        noun_chunks_path = base_dir / f"{stem}_noun_chunks.json"

        if grammar_path.exists():
            if args.verbose:
                print(f"Loading existing grammar: {grammar_path}")
            existing_grammar = load_existing_grammar(grammar_path)
            if args.verbose:
                print(f"  Loaded {len(existing_grammar)} existing grammar entries")

        if ner_path.exists():
            if args.verbose:
                print(f"Loading existing NER: {ner_path}")
            existing_ner = load_existing_dict(ner_path)
            if args.verbose and existing_ner:
                print(f"  Loaded {len(existing_ner)} existing NER entries")

        if verbs_path.exists():
            if args.verbose:
                print(f"Loading existing verbs: {verbs_path}")
            existing_verbs = load_existing_count_dict(verbs_path)
            if args.verbose:
                print(f"  Loaded {len(existing_verbs)} existing verb entries")

        if noun_chunks_path.exists():
            if args.verbose:
                print(f"Loading existing noun chunks: {noun_chunks_path}")
            existing_noun_chunks = load_existing_count_dict(noun_chunks_path)
            if args.verbose:
                print(f"  Loaded {len(existing_noun_chunks)} existing noun chunk entries")

    # Load spaCy model
    if args.verbose:
        print(f"Loading spaCy model: {args.model}")

    # Setup GPU if requested
    if args.gpu:
        gpu_enabled = setup_gpu(use_gpu=True, verbose=args.verbose)
        if not gpu_enabled and args.verbose:
            print("Warning: GPU requested but not available, falling back to CPU")

    nlp = spacy.load(args.model)

    # Load texts
    if args.verbose:
        print(f"Reading file: {args.input}")
    texts = load_texts_from_file(args.input)

    if args.verbose:
        total_chars = sum(len(t) for t in texts)
        print(f"Loaded {len(texts)} text segments ({total_chars:,} characters)")

    # Perform linguistic analysis
    if args.verbose:
        print("Performing linguistic analysis...")
    analysis = analyze_texts(
        nlp, texts,
        batch_size=args.batch_size,
        verbose=args.verbose,
        existing_grammar=existing_grammar,
        existing_ner=existing_ner,
        existing_verbs=existing_verbs,
        existing_noun_chunks=existing_noun_chunks,
    )

    # Save results
    stem = args.input.stem
    saved_files = save_analysis(analysis, args.output, stem)

    print(f"\nSaved {len(saved_files)} analysis files:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")

    # Print summary with new entries info
    if args.verbose:
        print_summary(analysis)


if __name__ == "__main__":
    main()
