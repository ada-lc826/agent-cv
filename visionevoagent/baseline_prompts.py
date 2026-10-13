"""Prompts of the re-implemented comparison methods, adapted to our module search space.

Each block follows the structure of the original method (paper + official code, see
reference/对比方法_prompt补齐.md in the article folder) but is written for our search space and
returns the same JSON as the Design Agent. Deviations from the originals are listed in README.md
("Comparison methods") and must be reported in the paper as "re-implemented under our search space".
"""
from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------- GENIUS (Zheng et al., 2023)
# One conversation; the first turn encodes the problem with step-by-step guidance, every later turn
# reports the accuracy of the last proposal and asks for a better one (repo: nas_bench_201.py).
GENIUS_SYSTEM = "You are an expert in neural architecture search. Return JSON only."

GENIUS_PROBLEM = """Your task is to help me choose modules to insert into an existing vision model so that \
its {metric} on {task} becomes as high as possible.

The model is a fixed, already trained-from-scratch baseline. A candidate is defined by four choices:
{space}

The baseline is {baseline}.

Let's think step by step. First analyse the available options. Then consider where in the network each \
option acts and how it changes the features and the gradient flow. Then propose one candidate that you \
expect to perform best, ignoring model size.

After you propose a candidate I will train it and report its {metric}. We will iterate on the design \
together; do not propose a design that was already tested."""

GENIUS_FEEDBACK = """With this model we achieved {result}. Please recommend a new model that outperforms \
all previous ones, based on the experiments so far, and explain why it should be better."""

# ---------------------------------------------------------------- EvoPrompting (Chen et al., NeurIPS 2023)
# Few-shot prompt of parent programs with their metrics; the last block states target metrics
# (params = 90% of the smallest parent, accuracy = 102% of the best parent) and asks for the child.
EVOPROMPTING_SYSTEM = ("You generate new neural architectures by combining and varying the example "
                       "architectures you are given. Return JSON only.")


def evoprompting_prompt(task: str, space: str, parents: List[Dict[str, Any]], target: Dict[str, float]) -> str:
    shots = "\n\n".join(f"Metrics: {{\"num_params_M\": {p['params']:.2f}, \"val_accuracy\": {p['score']:.2f}}}\n"
                        f"Architecture: {p['arch']}" for p in parents)
    return (f"Task: {task}. Each architecture is a choice of modules from this space:\n{space}\n\n"
            f"{shots}\n\n"
            f"Metrics: {{\"num_params_M\": {target['params']:.2f}, \"val_accuracy\": {target['score']:.2f}}}\n"
            "Architecture: ?\n\nWrite the architecture that achieves the last metrics.")


# ---------------------------------------------------------------- LLMatic (Nasir et al., GECCO 2024)
# Mutation = current network + one instruction from a fixed list; crossover = two networks + "combine".
# The official code (umair-nasir14/LLMatic) has 13 instructions; below they are re-worded for our space.
LLMATIC_SYSTEM = ("You modify neural architectures following an instruction. Every architecture is a choice "
                  "of modules from a fixed search space. Return JSON only.")

LLMATIC_INSTRUCTIONS = [
    "Add a module to improve the network.",
    "Remove a module to improve the network.",
    "Improve the network.",
    "Improve the network by making it much smaller.",
    "Improve the network by making it much larger.",
    "Improve the network by changing how the last stages' features are combined.",
    "Improve the network by changing its convolutions.",
    "Improve the network by changing where its modules are placed.",
    "Improve the network by adding attention.",
    "Improve the network by applying its modules to more stages.",
    "Improve the network by making its modules lighter.",
    "Improve the network by adding channel recalibration.",
    "Improve the network by enlarging the receptive field.",
]

LLMATIC_MUTATE = """Search space:
{space}

Network:
{arch}

Instruction: {instruction}
Give the modified network."""

LLMATIC_CROSSOVER = """Search space:
{space}

Network A:
{a}

Network B:
{b}

Combine the two networks above into a third network."""

# ---------------------------------------------------------------- LAPT (Zhou et al., AAAI 2025)
LAPT_SYSTEM = ("You are an expert in neural network design. You learn design principles from evaluated "
               "architectures and turn them into a smaller search space. Return JSON only.")

LAPT_LEARN = """Each architecture is a choice of modules from this search space:
{space}

Evaluated architectures ({metric}, higher is better):
{table}

Let's break this down step by step. First analyse how each choice acts in the network. Then find the \
common patterns among the better architectures and explain them. Finally infer design principles, and \
according to them list the options to keep for each dimension (at most {k} per dimension)."""

LAPT_UPDATE_IMPROVED = """The current principles are:
{principles}

Within the options {allowed}, these architectures were found ({metric}, higher is better):
{table}

Update the design principles based on these architectures, then list the options to keep for each \
dimension (at most {k} per dimension) to build better architectures."""

LAPT_UPDATE_STUCK = """The current principles are:
{principles}

With the options {allowed} it is hard to find better architectures. Describe the effect of options that \
are not covered by the current principles, and list a different set of options to keep for each \
dimension (at most {k} per dimension)."""


def lapt_schema(space: Dict[str, List[str]], dims: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "principles": {"type": "array", "items": {"type": "string"}},
            "allowed": {
                "type": "object",
                "properties": {d: {"type": "array", "items": {"type": "string", "enum": space[d]}} for d in dims},
                "required": list(dims),
                "additionalProperties": False,
            },
        },
        "required": ["principles", "allowed"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------- NADER (Yang et al., CVPR 2025)
NADER_SYSTEM = ("You are a computer vision research expert with deep insight into neural architecture design. "
                "Return JSON only.")

# Without the Reader, the original lets the LLM propose its own inspirations (PROMPT_PROPOSE_INSPIRATION_INIT).
NADER_INSPIRATIONS = """The model to be improved is {baseline}, built from this search space:
{space}

Propose {n} inspirations that could improve its {metric}.
Constraints: each must concern the network structure, be at most 50 words, and the set must be diverse."""

# Proposer (PROMPT_PROPOSER_MOG): pick a (model, inspiration) pair from the optimisation tree.
NADER_PROPOSE = """Model optimisation tree. Each node is a model with its {metric}; "parent --[i]--> child" means the child was obtained from the parent by applying inspiration i.
{tree}

Candidate inspirations (index: text):
{inspirations}

Choose one model of the tree and one inspiration so that the modified model is most likely to outperform every model in the tree. Look carefully at each model, at the effect each inspiration had where it was applied, and at the modification paths; weaker models may still improve with the right inspiration."""

# Modifier (prompt_modify_block): apply the inspiration, keep everything unrelated unchanged, follow the
# experiences retrieved for this inspiration.
NADER_MODIFY = """Modify the model according to the inspiration. Keep every choice that is unrelated to the inspiration as it is.

Search space:
{space}

Inspiration: {inspiration}

Model: {arch}
{experience}"""

NADER_EXPERIENCE = """
Suggestions learned from earlier failed modifications (follow them to get better performance):
{items}
"""

NADER_REFLECT = """A model was modified and its {metric} dropped.
Original model: {raw} ({raw_score:.2f})
Modified model: {new} ({new_score:.2f})
Inspiration that was applied: {inspiration}

Think step by step about why the modification hurt, then give one general suggestion (at most 50 words, \
about network structure) to avoid this mistake in later modifications."""


def nader_inspirations_schema() -> Dict[str, Any]:
    return {"type": "object", "properties": {"inspirations": {"type": "array", "items": {"type": "string"}}},
            "required": ["inspirations"], "additionalProperties": False}


NADER_PROPOSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"parent_round": {"type": "integer"}, "inspiration_index": {"type": "integer"},
                   "rationale": {"type": "string"}},
    "required": ["parent_round", "inspiration_index", "rationale"],
    "additionalProperties": False,
}


NADER_SUGGESTION_SCHEMA: Dict[str, Any] = {
    "type": "object", "properties": {"analysis": {"type": "string"}, "suggestion": {"type": "string"}},
    "required": ["analysis", "suggestion"], "additionalProperties": False}
