import json
import os
import re
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import spacy
import spacy_alignments
from cached_path import cached_path
from spacy.matcher import Matcher
from spacy.tokens import Doc, DocBin, Span, Token
from tqdm.auto import tqdm

RE_MULTIPLE_SPACES = re.compile(r" {2,}")

CAPTIONS_DIR = os.path.join(os.environ["SCRATCH_DIR"], "captions")

spacy.prefer_gpu()
NLP = spacy.load("en_core_web_trf")

Doc.set_extension("video_id", default=None)
Token.set_extension("start_time", default=None)
Token.set_extension("end_time", default=None)


def _list_caption_paths(dir_path: str) -> Iterator[str]:
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".json"):  # noqa
                yield entry.path  # noqa


def _captions_to_text(caption_full_dict: Mapping[str, Any]) -> str:
    return RE_MULTIPLE_SPACES.sub(" ", " ".join(d["alternatives"][0]["transcript"].strip()
                                                for d in caption_full_dict["results"][:-1])).strip()


def _parse_caption_time(s: str) -> float:
    return float(s[:-1])


def _load_caption(path: str) -> Optional[Mapping[str, Any]]:
    with open(path) as file:
        caption_full_dict = json.load(file)

        if results := caption_full_dict["results"]:
            tokens_info = results[-1]["alternatives"][0]["words"]
        else:
            tokens_info = None

        if tokens_info:
            return {  # Save some memory by just keeping what we actually use.
                "text": _captions_to_text(caption_full_dict),
                "video_id": os.path.basename(path).rsplit(".", maxsplit=1)[0],
                "tokens_info": [{
                    "word": wi["word"],
                    "start_time": _parse_caption_time(wi["startTime"]),
                    "end_time": _parse_caption_time(wi["endTime"]),
                } for wi in tokens_info],
            }
        else:
            return None  # There are around 750/150k that fall here for different reasons.


def _add_caption_info_to_doc(doc: Doc, tokens_info: Sequence[Mapping[str, Any]]) -> Doc:
    spacy2caption = spacy_alignments.get_alignments([t.text for t in doc], [w["word"] for w in tokens_info])[0]

    for token, caption_token_indices in zip(doc, spacy2caption):
        token._.start_time = tokens_info[caption_token_indices[0]]["start_time"]
        token._.end_time = tokens_info[caption_token_indices[-1]]["end_time"]

    return doc


def _create_docs() -> Iterator[Doc]:
    caption_paths = list(_list_caption_paths(CAPTIONS_DIR))
    # caption_paths = random.sample(caption_paths, min(len(caption_paths), 100))
    # We don't keep the captions in memory as it can be a lot.
    caption_it = (caption for path in caption_paths if (caption := _load_caption(path)))

    doc_and_context_it = NLP.pipe(((c["text"], (c["video_id"], c["tokens_info"]))  # noqa
                             for c in caption_it), as_tuples=True)

    for doc, (video_id, tokens_info) in tqdm(doc_and_context_it, total=len(caption_paths), desc="Parsing"):
        doc._.trf_data = None  # It takes up a lot of memory.
        doc._.video_id = video_id
        yield _add_caption_info_to_doc(doc, tokens_info)


def _load_cached_docs() -> Iterator[Doc]:
    print("Loading cached docs…")
    with open(cached_path("parsed_docs"), "rb") as file:
        return DocBin().from_bytes(file.read()).get_docs(NLP.vocab)


def _save_docs(docs: Iterable[Doc]) -> None:
    print("Saving cached docs…")
    with open("/tmp/.docs", "wb") as file:
        file.write(DocBin(store_user_data=True, docs=docs).to_bytes())


if os.environ.get("LOAD_CACHED_DOCS", "0").lower() in {"1", "true", "y"}:
    DOCS = list(_load_cached_docs())
else:
    DOCS = list(_create_docs())

if os.environ.get("SAVE_CACHED_DOCS", "0").lower() in {"1", "true", "y"}:
    _save_docs(DOCS)

print("Docs ready")


def search_in_subtitles(pattern: Iterable[Mapping[str, Any]]) -> Iterator[Span]:
    matcher = Matcher(NLP.vocab)
    matcher.add("search", [pattern])
    for doc in DOCS:
        for m in matcher(doc):
            yield doc[m[1]:m[2]]
