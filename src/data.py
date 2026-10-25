"""
Data preprocessing, BIO label alignment, and entity definition handling
for Just Pass Twice (JPT) Named Entity Recognition.
"""

import os
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# Standard canonical definitions (Channel 1: Single Definition per entity type)
CANONICAL_DEFINITIONS = {
    # CoNLL-2003 Entity Types
    "O": "A token or punctuation that is not part of any named entity.",
    "PER": "A named person, human individual, public figure, or fictional character.",
    "ORG": "An organization, company, institution, agency, or governing body.",
    "LOC": "A location, geographical place, city, country, mountain, or landmark.",
    "MISC": "A miscellaneous entity such as a nationality, event, or creative work.",
    
    # CrossNER Domain Entity Types
    "POLITICIAN": "A politician, government official, diplomat, or head of state.",
    "POLITICALPARTY": "A political party, political movement, or parliamentary coalition.",
    "COUNTRY": "A sovereign country, nation-state, or recognized territory.",
    "ELECTION": "A political election, referendum, plebiscite, or vote.",
    "EVENT": "A historical, political, scientific, or cultural event.",
    "RESEARCHER": "An artificial intelligence researcher, scientist, or computer scientist.",
    "FIELD": "An academic discipline, subfield of AI, or scientific study area.",
    "TASK": "A machine learning or computational task, objective, or benchmark.",
    "PRODUCT": "A software library, AI framework, hardware product, or commercial tool.",
    "ALGORITHM": "A specific machine learning algorithm, neural architecture, or mathematical technique.",
    "METRIC": "An evaluation metric, scoring criterion, or statistical measurement in computing.",
    "METRICS": "An evaluation metric, scoring criterion, or statistical measurement in computing.",
    "CONFERENCE": "An academic conference, symposium, workshop, or scientific convention.",
    "PROGRAMLANG": "A programming language, coding syntax, or formal computer language.",
    "UNIVERSITY": "A university, college, higher education institution, or academic research campus.",
    "AWARD": "An honor, prize, medal, trophy, or formal recognition.",
    "BOOK": "A published book, novel, literary volume, or textbook.",
    "LITERARYGENRE": "A genre of literature such as fiction, tragedy, science fiction, or realism.",
    "MAGAZINE": "A periodical magazine, gazette, or regular journal publication.",
    "POEM": "A poem, poetic piece, verse, or lyrical literary work.",
    "WRITER": "An author, novelist, poet, playwright, or literary essayist.",
    "ALBUM": "A music album, studio record, EP, or audio disc release.",
    "BAND": "A musical band, music group, rock band, or orchestra.",
    "MUSICALARTIST": "A musician, singer, composer, vocalist, or musical performer.",
    "MUSICALINSTRUMENT": "A musical instrument such as guitar, piano, drums, or violin.",
    "MUSICGENRE": "A musical genre such as jazz, rock, classical, or hip hop.",
    "SONG": "A song, musical track, single, or vocal composition.",
    "ACADEMICJOURNAL": "A peer-reviewed scientific journal, scholarly periodical, or academic review.",
    "ASTRONOMICALOBJECT": "An astronomical object, star, planet, galaxy, comet, or celestial body.",
    "CHEMICALCOMPOUND": "A chemical compound, chemical substance, molecule, or synthetic material.",
    "CHEMICALELEMENT": "A chemical element on the periodic table such as oxygen, iron, or carbon.",
    "DISCIPLINE": "A scientific discipline, branch of science, or academic field of inquiry.",
    "ENZYME": "A biological enzyme, catalytic macromolecule, or metabolic catalyst.",
    "PROTEIN": "A biological protein, amino acid chain, or peptide structure.",
    "SCIENTIST": "A natural scientist, physicist, chemist, biologist, or experimentalist.",
    "THEORY": "A scientific theory, physical law, scientific model, or mathematical hypothesis."
}

# Paraphrased definitions for testing the multi-definition aggregation hypothesis
PARAPHRASED_DEFINITIONS = {
    "O": [
        "A token or punctuation that is not part of any named entity.",
        "Common non-entity word, punctuation, or functional syntactic element.",
        "General vocabulary item representing neither names, places, nor organizations."
    ],
    "PER": [
        "A named person, human individual, public figure, or fictional character.",
        "Proper names of people, historical figures, or individual humans.",
        "An individual's personal name, surname, or fictional entity persona."
    ],
    "ORG": [
        "An organization, company, institution, agency, or governing body.",
        "A formal collective group, corporation, university, or administrative association.",
        "Commercial companies, non-profit institutions, clubs, and official organizations."
    ],
    "LOC": [
        "A location, geographical place, city, country, mountain, or landmark.",
        "Names of physical territories, geographic points, administrative regions, or nations.",
        "Any geographical spot, town, state, continent, or physical address."
    ],
    "MISC": [
        "A miscellaneous entity such as a nationality, event, or creative work.",
        "Special proper names including national affiliations, ethnic groups, and media titles.",
        "Other named entities that do not fit into person, organization, or location."
    ],
    "POLITICIAN": [
        "A politician, government official, diplomat, or head of state.",
        "Elected representatives, ministers, presidents, and political candidates.",
        "Individual persons engaged in public policy, government governance, and politics."
    ],
    "POLITICALPARTY": [
        "A political party, political movement, or parliamentary coalition.",
        "Organized political factions, registered parties, and ideological coalitions.",
        "Partisan political organizations competing in governance and electoral politics."
    ],
    "COUNTRY": [
        "A sovereign country, nation-state, or recognized territory.",
        "National state, recognized sovereign geopolitical entity, or sovereign homeland.",
        "Independent nations, states, and internationally recognized countries."
    ],
    "ELECTION": [
        "A political election, referendum, plebiscite, or vote.",
        "Formal voting events, parliamentary elections, and democratic ballots.",
        "Organized democratic contest or ballot to choose public leaders."
    ],
    "EVENT": [
        "A historical, political, scientific, or cultural event.",
        "Named occurrences, conferences, wars, tournaments, or significant historical incidents.",
        "Key milestone occurrences, conflicts, gatherings, or historical events."
    ],
    "RESEARCHER": [
        "An artificial intelligence researcher, scientist, or computer scientist.",
        "Academic scholars, investigators, and computer scientists authoring research.",
        "Individual scientists and researchers investigating AI and related fields."
    ],
    "FIELD": [
        "An academic discipline, subfield of AI, or scientific study area.",
        "Specialized branch of computer science, research domains, and academic disciplines.",
        "Area of study or technical scientific domain in artificial intelligence."
    ],
    "TASK": [
        "A machine learning or computational task, objective, or benchmark.",
        "Specific NLP or computer vision problem, sequence labeling task, or predictive objective.",
        "Technical problem statement or objective that algorithms are designed to solve."
    ],
    "PRODUCT": [
        "A software library, AI framework, hardware product, or commercial tool.",
        "Engineered software tools, machine learning packages, libraries, or commercial products.",
        "Manufactured items, software applications, developer toolkits, or platforms."
    ],
    "ALGORITHM": [
        "A specific machine learning algorithm, neural architecture, or mathematical technique.",
        "Computational method, network architecture, optimization rule, or algorithm.",
        "Formal algorithmic procedure or neural model structure used in computations."
    ],
    "METRIC": [
        "An evaluation metric, scoring criterion, or statistical measurement in computing.",
        "Benchmark evaluation measurement, numerical performance metric, or error indicator.",
        "Quantitative measure such as accuracy, F1-score, or precision used to assess models."
    ],
    "METRICS": [
        "An evaluation metric, scoring criterion, or statistical measurement in computing.",
        "Benchmark evaluation measurement, numerical performance metric, or error indicator.",
        "Quantitative measure such as accuracy, F1-score, or precision used to assess models."
    ],
    "CONFERENCE": [
        "An academic conference, symposium, workshop, or scientific convention.",
        "Scholarly gathering, academic meeting, or formal research convention.",
        "Peer-reviewed scientific proceeding venue or symposium event."
    ],
    "PROGRAMLANG": [
        "A programming language, coding syntax, or formal computer language.",
        "Software programming language, scripting tool, or computational coding language.",
        "Formal programming code system such as Python, C++, or Java."
    ],
    "UNIVERSITY": [
        "A university, college, higher education institution, or academic research campus.",
        "Academic tertiary institution, degree-granting college, or educational research center.",
        "Higher learning academy, university campus, or academic research faculty."
    ],
    "AWARD": [
        "An honor, prize, medal, trophy, or formal recognition.",
        "Distinction, scientific prize, medal of honor, or artistic award.",
        "Formal accolade, trophy, citation, or commemorative award."
    ],
    "BOOK": [
        "A published book, novel, literary volume, or textbook.",
        "Literary publication, printed book, monograph, or written volume.",
        "Published manuscript, novel, or scholarly book title."
    ],
    "LITERARYGENRE": [
        "A genre of literature such as fiction, tragedy, science fiction, or realism.",
        "Category of literary writing, stylistic tradition, or narrative format.",
        "Literary form, artistic style, or genre category of written literature."
    ],
    "MAGAZINE": [
        "A periodical magazine, gazette, or regular journal publication.",
        "Regular print magazine, periodical, or serial gazette publication.",
        "Monthly or weekly published magazine or journalistic periodical."
    ],
    "POEM": [
        "A poem, poetic piece, verse, or lyrical literary work.",
        "Poetic composition, stanza verse, sonnet, or lyrical written work.",
        "Metered poetry, verse creation, or poetic literature piece."
    ],
    "WRITER": [
        "An author, novelist, poet, playwright, or literary essayist.",
        "Literary author, creative novelist, or published writer.",
        "Individual person authoring books, poetry, plays, or literary essays."
    ],
    "ALBUM": [
        "A music album, studio record, EP, or audio disc release.",
        "Recorded audio collection, musical LP, studio record, or release.",
        "Official musical record, track compilation, or studio album."
    ],
    "BAND": [
        "A musical band, music group, rock band, or orchestra.",
        "Ensemble of musicians, performing music group, or pop/rock band.",
        "Musical group, orchestra, quartet, or collaborative band."
    ],
    "MUSICALARTIST": [
        "A musician, singer, composer, vocalist, or musical performer.",
        "Recording artist, vocal performer, songwriter, or instrumental musician.",
        "Individual creative music artist, singer, or instrumental performer."
    ],
    "MUSICALINSTRUMENT": [
        "A musical instrument such as guitar, piano, drums, or violin.",
        "Acoustic or electronic instrument designed to play musical notes.",
        "Physical sound-producing device used in musical performance."
    ],
    "MUSICGENRE": [
        "A musical genre such as jazz, rock, classical, or hip hop.",
        "Stylistic musical tradition, genre, or sound classification.",
        "Category of musical composition, artistic tradition, or genre style."
    ],
    "SONG": [
        "A song, musical track, single, or vocal composition.",
        "Recorded audio track, musical piece, single, or vocal song.",
        "Individual composition or song released by a musician."
    ],
    "ACADEMICJOURNAL": [
        "A peer-reviewed scientific journal, scholarly periodical, or academic review.",
        "Refereed academic publication, scientific journal, or scholarly periodical.",
        "Formal scientific research journal publishing peer-reviewed papers."
    ],
    "ASTRONOMICALOBJECT": [
        "An astronomical object, star, planet, galaxy, comet, or celestial body.",
        "Celestial physical entity, planet, asteroid, nebula, or galaxy.",
        "Cosmic body, star system, planet, or deep space astronomical feature."
    ],
    "CHEMICALCOMPOUND": [
        "A chemical compound, chemical substance, molecule, or synthetic material.",
        "Molecular chemical structure, compound, or chemical agent.",
        "Pure chemical substance composed of bonded atoms or molecules."
    ],
    "CHEMICALELEMENT": [
        "A chemical element on the periodic table such as oxygen, iron, or carbon.",
        "Fundamental chemical element, atomic species, or elemental substance.",
        "Specific atomic element identified by atomic number on the periodic table."
    ],
    "DISCIPLINE": [
        "A scientific discipline, branch of science, or academic field of inquiry.",
        "Specialized scientific branch, field of study, or academic discipline.",
        "Organized domain of knowledge, science branch, or research field."
    ],
    "ENZYME": [
        "A biological enzyme, catalytic macromolecule, or metabolic catalyst.",
        "Protein catalyst speeding up biochemical metabolic reactions.",
        "Biochemical enzyme responsible for cellular organic catalysis."
    ],
    "PROTEIN": [
        "A biological protein, amino acid chain, or peptide structure.",
        "Polypeptide macromolecule, structural protein, or biological complex.",
        "Organic molecule composed of amino acid sequences forming cellular proteins."
    ],
    "SCIENTIST": [
        "A natural scientist, physicist, chemist, biologist, or experimentalist.",
        "Professional researcher, experimental physicist, or scientific investigator.",
        "Individual person conducting empirical research in natural sciences."
    ],
    "THEORY": [
        "A scientific theory, physical law, scientific model, or mathematical hypothesis.",
        "Well-substantiated scientific explanation, mathematical law, or theoretical model.",
        "Established physical theorem, scientific hypothesis, or theoretical principle."
    ]
}


def normalize_tag(tag: str) -> str:
    """Normalize raw BIO tags across CoNLL and CrossNER formats to canonical uppercase."""
    tag = tag.strip()
    if not tag or tag == "O":
        return "O"
    
    parts = tag.split("-", 1)
    if len(parts) == 2:
        bio, ent_type = parts[0].upper(), parts[1].upper()
        # Canonical mappings
        if ent_type in ("ORGANISATION", "ORGANIZATION"):
            ent_type = "ORG"
        elif ent_type == "PERSON":
            ent_type = "PER"
        elif ent_type == "LOCATION":
            ent_type = "LOC"
        return f"{bio}-{ent_type}"
    return tag.upper()


def extract_entity_types_from_tags(tag_list: List[str]) -> List[str]:
    """Extract distinct entity type names (e.g., ['PER', 'ORG', 'LOC', 'MISC']) excluding 'O'."""
    types = set()
    for tag in tag_list:
        if tag.startswith("B-") or tag.startswith("I-"):
            types.add(tag.split("-", 1)[1])
    return sorted(list(types))


def read_conll_file(file_path: str) -> List[Tuple[List[str], List[str]]]:
    """
    Read a CoNLL/CrossNER format file.
    Sentences are separated by empty lines.
    Each non-empty line has: token <TAB/SPACE> label (or token pos chunk label).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    sentences = []
    current_tokens = []
    current_tags = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                continue
            
            # Skip doc boundary lines in CoNLL if present
            if line.startswith("-DOCSTART-"):
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = normalize_tag(parts[-1])
                current_tokens.append(token)
                current_tags.append(tag)
                
        if current_tokens:
            sentences.append((current_tokens, current_tags))
            
    return sentences


def get_dataset_statistics(sentences: List[Tuple[List[str], List[str]]]) -> Dict:
    """Compute sentence counts, token counts, entity mention counts by type."""
    num_sentences = len(sentences)
    num_tokens = sum(len(tokens) for tokens, _ in sentences)
    type_counts = {}
    
    for tokens, tags in sentences:
        for tag in tags:
            if tag.startswith("B-"):
                ent_type = tag.split("-", 1)[1]
                type_counts[ent_type] = type_counts.get(ent_type, 0) + 1
                
    return {
        "num_sentences": num_sentences,
        "num_tokens": num_tokens,
        "entity_mentions": type_counts,
        "total_mentions": sum(type_counts.values())
    }


class JPTNERDataset(Dataset):
    """
    PyTorch Dataset supporting:
    1. Single-pass baseline (standard causal or token sequence)
    2. Two-pass JPT (input sequence duplicated, loss applied only to second pass)
    """
    def __init__(
        self,
        sentences: List[Tuple[List[str], List[str]]],
        tokenizer: AutoTokenizer,
        label_to_id: Dict[str, int],
        two_pass: bool = True,
        max_length: int = 128
    ):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.two_pass = two_pass
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens, tags = self.sentences[idx]
        
        # Subword tokenization for pass 1
        subwords = []
        word_to_subword_indices = []
        
        for word in tokens:
            # Prepend space for GPT-2 BPE tokenizer to match natural word boundaries
            word_subwords = self.tokenizer.tokenize(" " + word)
            if not word_subwords:
                word_subwords = [self.tokenizer.unk_token or ""]
            word_to_subword_indices.append((len(subwords), len(subwords) + len(word_subwords)))
            subwords.extend(word_subwords)
            
        subword_ids = self.tokenizer.convert_tokens_to_ids(subwords)
        
        # Truncate sequence if too long
        max_single_len = (self.max_length // 2 - 2) if self.two_pass else (self.max_length - 2)
        if len(subword_ids) > max_single_len:
            subword_ids = subword_ids[:max_single_len]
            # Adjust word spans
            valid_word_indices = []
            for start, end in word_to_subword_indices:
                if start < max_single_len:
                    valid_word_indices.append((start, min(end, max_single_len)))
            word_to_subword_indices = valid_word_indices
            tokens = tokens[:len(word_to_subword_indices)]
            tags = tags[:len(word_to_subword_indices)]

        sep_id = getattr(self.tokenizer, "eos_token_id", 50256) or 50256
        
        if self.two_pass:
            # Concatenate: [Pass 1 tokens, SEP, Pass 2 tokens]
            # Pass 2 tokens are the ones we extract representations from and compute loss on
            pass1_ids = subword_ids
            pass2_ids = subword_ids
            input_ids = pass1_ids + [sep_id] + pass2_ids
            attention_mask = [1] * len(input_ids)
            
            # Second pass offset
            pass2_offset = len(pass1_ids) + 1
            
            # Build target labels aligned to the second pass
            # -100 indicates ignored positions (all pass 1 tokens, separator, padding)
            labels = [-100] * len(input_ids)
            eval_mask = [0] * len(input_ids)
            
            for (w_start, w_end), tag in zip(word_to_subword_indices, tags):
                # The first subword gets the entity tag id, subsequent subwords get -100
                first_subword_idx = pass2_offset + w_start
                if first_subword_idx < len(labels):
                    labels[first_subword_idx] = self.label_to_id.get(tag, self.label_to_id.get("O", 0))
                    eval_mask[first_subword_idx] = 1
                    
        else:
            # Single-pass baseline
            input_ids = subword_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(input_ids)
            eval_mask = [0] * len(input_ids)
            
            for (w_start, w_end), tag in zip(word_to_subword_indices, tags):
                if w_start < len(labels):
                    labels[w_start] = self.label_to_id.get(tag, self.label_to_id.get("O", 0))
                    eval_mask[w_start] = 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eval_mask": torch.tensor(eval_mask, dtype=torch.long),
            "num_words": torch.tensor(len(tokens), dtype=torch.long)
        }


def collate_jpt_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 50256) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences dynamically to the batch maximum length."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_eval_mask = []
    
    for item in batch:
        cur_len = item["input_ids"].size(0)
        pad_len = max_len - cur_len
        
        batch_input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        batch_attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        batch_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        batch_eval_mask.append(
            torch.cat([item["eval_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        
    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
        "eval_mask": torch.stack(batch_eval_mask)
    }


def create_definition_embeddings(
    entity_types: List[str],
    definitions_dict: Dict,
    tokenizer: AutoTokenizer,
    backbone_model: torch.nn.Module,
    device: str = "cpu",
    is_multi: bool = False
) -> torch.Tensor:
    """
    Encode entity definition strings using the backbone's token embeddings.
    For each definition, tokenizes the text and computes the mean pooled embedding.
    Returns:
    - If is_multi=False: tensor of shape [num_types, hidden_size]
    - If is_multi=True: tensor of shape [num_types, num_paraphrases, hidden_size]
    """
    backbone_model.eval()
    embed_layer = backbone_model.get_input_embeddings()
    hidden_size = backbone_model.config.hidden_size

    type_embeddings = []

    with torch.no_grad():
        for ent_type in entity_types:
            defs = definitions_dict.get(ent_type, ["An entity mention."])
            if isinstance(defs, str):
                defs = [defs]

            paraphrase_embs = []
            for d in defs:
                tokens = tokenizer.encode(" " + d, add_special_tokens=False)
                if not tokens:
                    tokens = [tokenizer.eos_token_id or 0]
                tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
                emb = embed_layer(tok_tensor).mean(dim=0)  # [hidden_size]
                paraphrase_embs.append(emb)

            if is_multi:
                stacked = torch.stack(paraphrase_embs)
                type_embeddings.append(stacked)
            else:
                type_embeddings.append(paraphrase_embs[0])

    if is_multi:
        max_p = max(t.size(0) for t in type_embeddings)
        padded = []
        for t in type_embeddings:
            if t.size(0) < max_p:
                pad_t = t[-1:].repeat(max_p - t.size(0), 1)
                t = torch.cat([t, pad_t], dim=0)
            padded.append(t)
        return torch.stack(padded).to(device)
    else:
        return torch.stack(type_embeddings).to(device)

