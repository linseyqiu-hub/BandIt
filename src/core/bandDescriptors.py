"""
Distilled IELTS band descriptors — paraphrased from official rubric.
4 criteria × bands 1–9. Used in RAG feedback prompt.
Window: overall ± 1 bands injected per request.
"""

BAND_DESCRIPTORS = {
    1: {
        "TR": "Content is wholly unrelated to the prompt. No addressable ideas present.",
        "CC": "Writing fails to communicate any message. No organisational features evident.",
        "LR": "No usable vocabulary resource. Only isolated words at most.",
        "GRA": "No rateable language. No sentence forms evident.",
    },
    2: {
        "TR": "Content barely related to the prompt. No identifiable position. One or two glimpses of ideas with no development.",
        "CC": "Little relevant message; response may be entirely off-topic. No control of organisational features.",
        "LR": "Extremely limited resource. Few recognisable strings beyond memorised phrases. No control of word formation or spelling.",
        "GRA": "Little or no evidence of sentence forms, except in memorised phrases.",
    },
    3: {
        "TR": "Prompt not adequately addressed or misunderstood. No identifiable position. Few ideas, mostly irrelevant or undeveloped.",
        "CC": "No apparent logical organisation. Ideas discernible but hard to relate. Minimal cohesive devices; sequencers used without logical purpose. Paragraphing attempts are unhelpful.",
        "LR": "Inadequate resource; possible over-dependence on memorised language. Very limited control of word choice and spelling; errors frequently impede meaning.",
        "GRA": "Sentence forms attempted but grammar and punctuation errors dominate. Most meaning fails to come through. Response may be too short to assess.",
    },
    4: {
        "TR": "Prompt only minimally or tangentially addressed; position hard to identify. Ideas lack relevance, clarity, or support. Heavy repetition.",
        "CC": "No clear progression. Cohesive devices basic, inaccurate, or repetitive. Paragraphing absent or with no clear main topic.",
        "LR": "Basic vocabulary, possibly repetitive or unrelated to task. Memorised chunks and inappropriate word choice common. Errors frequently impede meaning.",
        "GRA": "Simple sentences dominate; complex structures rare and error-prone. Frequent grammatical errors impede meaning. Punctuation often faulty.",
    },
    5: {
        "TR": "Main parts incompletely addressed. Position stated but development unclear. Ideas limited, underdeveloped, or irrelevant. Some repetition.",
        "CC": "Some organisation evident but not consistently logical; no clear overall progression. Ideas loosely linked, not fluently connected. Cohesive devices limited or overused. Paragraphing may be missing.",
        "LR": "Limited vocabulary but minimally adequate. Frequent lapses in word choice and flexibility. Spelling and formation errors noticeable and may cause reader difficulty.",
        "GRA": "Structures limited and repetitive. Complex sentences attempted but tend to be faulty; greatest accuracy on simple sentences. Errors frequent and cause some reader difficulty. Punctuation may be faulty.",
    },
    6: {
        "TR": "Main parts addressed though unevenly covered. Position relevant but conclusions may be unclear or unjustified. Ideas relevant but some underdeveloped or weakly supported.",
        "CC": "Generally coherent with clear overall progression. Cohesive devices used with some effect but may be faulty, mechanical, or overused. Paragraphing not always logical.",
        "LR": "Generally adequate vocabulary for the task. Meaning clear despite restricted range or imprecise word choice. Some spelling and formation errors that do not impede communication.",
        "GRA": "Mix of simple and complex structures but flexibility limited. Complex structures less accurate than simple ones. Errors in grammar and punctuation occur but rarely impede communication.",
    },
    7: {
        "TR": "Main parts addressed with a clear, developed position. Ideas extended and supported but may over-generalise or lack precision in supporting material.",
        "CC": "Logically organised with clear progression; minor lapses. Cohesive devices used flexibly though with some inaccuracy or over/under-use. Paragraphing generally effective.",
        "LR": "Sufficient range for flexibility and precision. Some less common and idiomatic items used. Style and collocation awareness evident despite some inappropriacies. Few spelling or formation errors that do not detract from clarity.",
        "GRA": "Variety of complex structures with some flexibility and accuracy. Grammar and punctuation generally well controlled; error-free sentences frequent. Minor errors persist but do not impede communication.",
    },
    8: {
        "TR": "Prompt sufficiently addressed with a clear, well-developed position. Ideas relevant, well extended, and supported. Only occasional omissions or lapses in content.",
        "CC": "Message easy to follow; ideas logically sequenced and cohesion well managed. Occasional lapses. Paragraphing sufficient and appropriate.",
        "LR": "Wide resource used fluently and flexibly to convey precise meaning. Skilful use of uncommon and idiomatic items despite occasional inaccuracies in word choice or collocation. Spelling and formation errors minimal.",
        "GRA": "Wide range of structures used flexibly and accurately. Majority of sentences error-free; punctuation well managed. Occasional non-systematic errors with minimal communicative impact.",
    },
    9: {
        "TR": "Prompt fully and deeply explored. Clear, fully developed position that directly answers the question. Ideas fully extended, well supported, and highly relevant.",
        "CC": "Message followed effortlessly. Cohesion seamless and rarely attracts attention. Any lapses in coherence or cohesion are minimal. Paragraphing skilfully managed.",
        "LR": "Full flexibility and precise use widely evident. Wide vocabulary used accurately and appropriately with sophisticated control. Spelling and formation errors extremely rare with minimal communicative impact.",
        "GRA": "Wide range of structures with full flexibility and control. Grammar and punctuation used appropriately throughout. Errors extremely rare and have minimal communicative impact.",
    },
}


def get_window_descriptors(overall: int) -> dict:
    """
    Return descriptors for bands overall-1, overall, overall+1.
    Clamps to [1, 9].
    """
    lo = max(1, overall - 1)
    hi = min(9, overall + 1)
    return {band: BAND_DESCRIPTORS[band] for band in range(lo, hi + 1)}


def format_descriptors_for_prompt(overall: int) -> str:
    """
    Format windowed descriptors as a compact prompt block.
    """
    window = get_window_descriptors(overall)
    lines = ["## IELTS Band Descriptors (reference window)"]
    for band, criteria in window.items():
        lines.append(f"\n### Band {band}")
        for criterion, desc in criteria.items():
            lines.append(f"**{criterion}:** {desc}")
    return "\n".join(lines)