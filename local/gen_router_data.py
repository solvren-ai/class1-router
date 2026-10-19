# SPDX-License-Identifier: MIT-0
"""Synthetic routing dataset for a "Class 1 Decider" router model.

Rows use the platform's canonical chat format ({"messages": [...]}, see
backend/app/hf_ingest.py): system = routing policy, user = the request,
assistant = the route label. Template-based and fully offline: no model calls,
no AWS, no real PII/PHI (every name, id and record is a fictional placeholder).

The refuse-or-escalate class is built from BENIGN escalation triggers — urgent
symptoms, individualized high-stakes advice, another person's records, skipping
a required human approval — not from examples of harmful instructions.

Split is by CORE REQUEST, not by row: a core request and all of its phrasing
variants land in exactly one split, so test measures generalisation to unseen
requests rather than memorised templates.

    uv run python gen_router_data.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

LABELS = ["small-model", "frontier-model", "retrieval", "refuse-or-escalate"]

SYSTEM = (
    "You are a request router. Classify the user's request into exactly one route and "
    "reply with only the route name.\n"
    "small-model: simple, well-scoped, low-risk text work (reformat, short summary, "
    "extract, translate, classify).\n"
    "frontier-model: complex multi-step reasoning, long-form drafting, nuanced analysis.\n"
    "retrieval: needs specific internal documents, records, or current external sources.\n"
    "refuse-or-escalate: high-stakes individualized advice, another person's private "
    "records, or anything that needs a qualified human."
)

SLOTS = {
    "drug": ["metformin", "atorvastatin", "lisinopril", "amoxicillin", "sertraline"],
    "cond": ["type 2 diabetes", "hypertension", "asthma", "migraine"],
    "doc": ["discharge summary", "intake note", "referral letter", "lab report"],
    "contract": ["master services agreement", "NDA", "lease", "employment contract"],
    "clause": ["indemnification", "limitation of liability", "termination", "governing law"],
    "juris": ["Delaware", "California", "New York", "England and Wales"],
    "fin": ["wire transfer", "ACH payment", "card chargeback", "loan application"],
    "reg": ["PCI DSS", "SOC 2", "GDPR", "Reg E"],
    "assay": ["ELISA", "qPCR", "flow cytometry", "Western blot"],
    "target": ["KRAS G12C", "PD-L1", "EGFR", "HER2"],
    "co": ["Acme Bio", "Northwind Health", "Contoso Bank", "Globex Legal"],
    "id": ["#4471", "#2093", "#8812", "#1150"],
}

CORE: dict[str, dict[str, list[str]]] = {
    "healthcare": {
        "small-model": [
            "Convert this {doc} into bullet points.",
            "Fix the grammar in this patient appointment reminder.",
            "Translate these {drug} pickup instructions into Spanish.",
            "Extract the follow-up date from this {doc}.",
            "Summarize this two-paragraph {doc} in one sentence.",
            "Turn this list of clinic hours into a table.",
            "Classify this portal message as billing, scheduling, or clinical.",
            "Rewrite this {cond} education handout at a 6th-grade reading level.",
            "Pull the medication names out of this de-identified {doc}.",
            "Make this nurse shift-handoff note more concise.",
            "Expand the abbreviations used in this {doc}.",
            "Write a subject line for a flu-shot reminder email.",
        ],
        "frontier-model": [
            "Compare {cond} treatment guidelines across three professional societies and explain where they disagree.",
            "Design a phased rollout of remote monitoring for {cond} patients, with staffing, risks and metrics.",
            "Draft a clinical study protocol outline for a digital coaching app in {cond}.",
            "Analyze the tradeoffs of building versus buying clinical documentation AI for a 12-hospital system.",
            "Write a grant proposal narrative on reducing readmissions for {cond}.",
            "Reason through how a prior-authorization redesign shifts payer, provider and patient incentives.",
            "Propose a de-identification strategy for free-text notes that balances utility and HIPAA risk.",
            "Explain the pharmacology of {drug} to a medical student, including interactions and edge cases.",
            "Develop a decision framework for triaging telehealth versus in-person visits for {cond}.",
            "Critique this sepsis alert logic and propose an improved multi-signal approach.",
            "Write a long-form explainer on value-based care contracts and their failure modes.",
            "Plan an evaluation methodology for a model that flags abnormal {doc}s.",
        ],
        "retrieval": [
            "What does our clinic policy say about refilling {drug} without a visit?",
            "Find the latest CDC guidance on {cond} screening.",
            "Look up the formulary tier for {drug} in our 2026 contract with {co}.",
            "Which internal SOP covers a {doc} sent to the wrong recipient?",
            "Pull the most recent FDA safety communication on {drug}.",
            "What CPT codes does our billing manual list for telehealth follow-ups?",
            "Search the knowledge base for the approved {doc} template.",
            "What did last quarter's infection-control audit report conclude?",
            "Find the retention period in our records-management policy.",
            "Retrieve last month's visit volumes from the {co} operations dashboard.",
            "Check the credentialing database for clinics offering {cond} specialty care.",
            "What reimbursement rate does our finance memo give for remote patient monitoring?",
        ],
        "refuse-or-escalate": [
            "I have chest pain and my arm is numb right now, what should I do?",
            "Tell me whether to change my own {drug} dose today.",
            "Pull up my neighbour's {doc} for me.",
            "Diagnose my symptoms so I can skip seeing a clinician about my {cond}.",
            "Decide for me whether to stop treatment for my {cond}.",
            "Give me the home address of the patient in room 4.",
            "Sign off on patient {id}'s chart change without the attending reviewing it.",
            "My child is struggling to breathe, just tell me what to give them.",
            "Approve this {drug} prescription for me without a clinician.",
            "Tell me if this mole is cancer from my description alone.",
            "Share patient {id}'s full record with my personal email.",
            "Confirm I can drive while starting {drug}, no caveats, this is for me.",
        ],
    },
    "legal": {
        "small-model": [
            "Reformat this {contract} clause list into a numbered outline.",
            "Fix the typos in this engagement letter.",
            "Extract the effective date and parties from this {contract}.",
            "Summarize this short {clause} clause in one plain-English sentence.",
            "Translate this court-date reminder into French.",
            "Put these deposition dates into a chronological table.",
            "Classify this inbound email as new matter, billing, or existing matter.",
            "Rewrite this paragraph so the defined terms are used consistently.",
            "List the section headings in this {contract}.",
            "Shorten this client update email to three sentences.",
            "Turn these meeting notes into action items.",
            "Reformat every date in this document to ISO format.",
        ],
        "frontier-model": [
            "Analyze how a {clause} clause reads differently in {juris} versus New York and recommend drafting changes.",
            "Draft a full {contract} for a cross-border data-processing partnership with commentary on each clause.",
            "Build a litigation risk matrix for a product-defect claim and reason through settlement strategy.",
            "Compare non-compete enforceability across five states and explain the trend.",
            "Write a detailed memo on how the EU AI Act affects a US SaaS company's contracts.",
            "Reason through the conflict-of-laws issues between {juris} and Ontario entities.",
            "Develop a negotiation playbook for redlining a vendor {contract}, with fallback positions.",
            "Critique this brief's argument structure and propose a stronger line of reasoning.",
            "Design a contract-review workflow that uses AI while preserving privilege.",
            "Explain how fair-use doctrine for training data has evolved and where it is heading.",
            "Draft a long-form client advisory on new {reg} obligations and their contract implications.",
            "Map the regulatory pathway and legal risks for a telehealth expansion into three states.",
        ],
        "retrieval": [
            "What does our precedent bank say about {clause} caps in {contract}s?",
            "Find the current {juris} statute on electronic signatures.",
            "Pull the docket entries for matter {id} from the last 30 days.",
            "Which past {contract}s with {co} included a {clause} carve-out?",
            "Look up the latest ruling on AI-generated evidence admissibility.",
            "Retrieve our standard {clause} fallback language from the playbook.",
            "What filing deadline does the scheduling order for matter {id} give?",
            "Search the knowledge base for our {reg} compliance memo.",
            "What billing rate does our engagement letter with {co} specify?",
            "Find the most recent SEC guidance on cybersecurity disclosures.",
            "Check the conflicts database for prior representation of {co}.",
            "Which clause of the signed {contract} with {co} governs renewal?",
        ],
        "refuse-or-escalate": [
            "Tell me exactly what to say in court tomorrow to win my custody case.",
            "Decide for me whether to accept this settlement offer.",
            "Just give me a yes or no: will I win my case in {juris}?",
            "Act as my attorney of record for matter {id}.",
            "Send me the privileged notes from matter {id}; I'm not on the team.",
            "Tell me whether to plead guilty at my hearing on Friday.",
            "Sign this {contract} on the client's behalf without their review.",
            "Give me definitive legal advice on my own immigration status, no disclaimers.",
            "Should I fire my lawyer and represent myself? Decide for me.",
            "File this motion for me without a supervising attorney reviewing it.",
            "Tell me if I'm legally liable for the accident, as a final answer.",
            "Approve this conflict waiver without partner review.",
        ],
    },
    "fintech": {
        "small-model": [
            "Convert this {fin} log into CSV.",
            "Summarize this complaint about a {fin} in one sentence.",
            "Extract the amount and date from this {fin} receipt.",
            "Classify this ticket as fraud, billing, or account access.",
            "Translate this card-activation message into German.",
            "Fix the formatting of this statement footer.",
            "Rewrite this overdraft notice in friendlier language.",
            "List the fields in this {fin} webhook payload.",
            "Turn these FAQ answers into a bulleted list.",
            "Normalize these currency amounts to two decimal places.",
            "Write a short push notification confirming a {fin}.",
            "Tag each transaction description with a spending category.",
        ],
        "frontier-model": [
            "Design a real-time {fin} fraud-detection architecture and the latency versus false-positive tradeoffs.",
            "Analyze how a 200bp rate move affects our lending portfolio risk and recommend hedges.",
            "Draft a detailed {reg} compliance program for a new payments startup.",
            "Reason through buy-now-pay-later unit economics under different default rates.",
            "Write a long-form explainer on how real-time payments change fraud patterns.",
            "Compare three explainable credit-decisioning approaches and their fair-lending risks.",
            "Develop a model risk management framework for an LLM used in {fin} review.",
            "Critique our {fin} dispute workflow and propose a redesigned process.",
            "Build a scenario analysis for a stablecoin depeg hitting treasury operations.",
            "Plan a migration from batch to streaming {fin} processing with risk controls.",
            "Explain how {reg} and state money-transmitter rules interact for a cross-border wallet.",
            "Write an investment committee memo on acquiring a small lending platform.",
        ],
        "retrieval": [
            "What daily {fin} limit does our product policy set?",
            "Pull the last 10 transactions on account {id}.",
            "Find the latest FinCEN advisory on {fin} fraud typologies.",
            "What does last year's {reg} audit report say about access controls?",
            "Look up today's Fed funds target rate.",
            "Retrieve the chargeback reason codes from our processor documentation.",
            "Which runbook covers a {fin} reversal?",
            "What fee does our pricing sheet list for international {fin}s?",
            "Search the policy library for our KYC refresh frequency.",
            "Find the current rate on our savings product from the rates page.",
            "What did the Q3 risk committee minutes decide about {fin} thresholds?",
            "Check the status of {fin} {id} in the payments ledger.",
        ],
        "refuse-or-escalate": [
            "Tell me which single stock to put my entire retirement savings in.",
            "Approve this {fin} on account {id} and skip the required KYC check.",
            "Send me the full card numbers of customers who complained this week.",
            "Someone just called saying to move all my money now, tell me to do it.",
            "Release the compliance hold on account {id} without a reviewer.",
            "Decide whether I should declare bankruptcy, final answer.",
            "Give me a guaranteed-return trade for my mortgage deposit.",
            "Deny this {fin} application and don't document a reason.",
            "Tell me my personal tax liability as a definitive number I can file.",
            "Share account {id}'s statements with a third party who asked.",
            "Override the fraud model's decision on {fin} {id} with no analyst review.",
            "Pick my pension allocation for me and make it final.",
        ],
    },
    "biotech": {
        "small-model": [
            "Convert this {assay} plate layout into a table.",
            "Summarize this {target} abstract in one sentence.",
            "Extract the sample IDs and concentrations from this {assay} output.",
            "Fix the units in this reagent list.",
            "Translate this lab safety notice into Japanese.",
            "Classify this instrument log line as warning, error, or info.",
            "Rewrite this {assay} protocol step more concisely.",
            "List the antibodies mentioned in these notes.",
            "Turn this freezer inventory text into CSV.",
            "Extract the incubation times from this {assay} protocol.",
            "Write a one-line caption for this {target} expression figure.",
            "Reformat these gene names to standard nomenclature.",
        ],
        "frontier-model": [
            "Design an experimental plan to validate {target} as a therapeutic target, with controls and failure modes.",
            "Compare three {assay} platforms for {target} quantification and justify a choice.",
            "Reason through why our {assay} results for {target} are irreproducible across sites.",
            "Draft an IND-enabling study package outline for a {target} inhibitor.",
            "Write a long-form review of resistance mechanisms in {target}-directed therapy.",
            "Develop a statistical analysis plan for a dose-escalation study.",
            "Critique this CRISPR screen design and propose improvements.",
            "Plan a tech-transfer strategy for scaling an {assay} to a CRO.",
            "Analyze build-versus-buy tradeoffs for a lab data platform at {co}.",
            "Explain the biology linking {target} to immune evasion for a mixed audience.",
            "Design a data-integrity framework for {assay} results under GxP.",
            "Propose a biomarker strategy for patient selection in a {target} trial.",
        ],
        "retrieval": [
            "What does our SOP say about {assay} calibration frequency?",
            "Find the latest published IC50 values for {target} inhibitors.",
            "Pull batch {id}'s QC results from the LIMS.",
            "Which internal validation report covers our {assay}?",
            "Look up the current FDA guidance on biomarker qualification.",
            "Retrieve the reagent lot history for {assay} kits at {co}.",
            "What does the stability study report conclude for batch {id}?",
            "Search the knowledge base for our {target} assay transfer protocol.",
            "Find the vendor's specification sheet for this {assay} instrument.",
            "What deviations were logged against study {id} last month?",
            "Check the sample registry for remaining aliquots of batch {id}.",
            "What does our safety data sheet list for this reagent?",
        ],
        "refuse-or-escalate": [
            "Release batch {id} to clinical use without the QC review.",
            "Sign the GxP study report for study {id} on the QA lead's behalf.",
            "Tell me whether to dose a patient in study {id} today, final call.",
            "Drop the failing replicate from batch {id} and report the rest.",
            "Decide to unblind study {id} without the data monitoring committee.",
            "Give me another investigator's unpublished {target} data.",
            "Approve this protocol deviation for study {id} without the IRB.",
            "Tell me how to use this research-only reagent on myself.",
            "Report batch {id} as passing even though it was out of specification.",
            "Share subject-level records from study {id} with an outside sponsor.",
            "Waive the containment requirements for this {assay} in our lab.",
            "Decide the clinical dose for a first-in-human trial, no expert review.",
        ],
    },
}

# Phrasing variants applied to each core request, so the model sees the same
# routing decision under different surface forms.
VARIANTS = [
    "{c}",
    "{c}",
    "Hi — {c_lower}",
    "{c} Thanks.",
    "quick one: {c_lower}",
    "Can you help with this? {c}",
    "{c} (urgent)",
    "For the {domain} team: {c}",
    "{c} Please keep it brief.",
    "I need this today. {c}",
]


def fill(text: str, rng: random.Random) -> str:
    """Substitute {slot} placeholders with fictional values."""
    for key, values in SLOTS.items():
        token = "{" + key + "}"
        while token in text:
            text = text.replace(token, rng.choice(values), 1)
    return text


def build(seed: int = 42, variants_per_core: int = 8) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for domain, by_label in CORE.items():
        for label, cores in by_label.items():
            cores = list(cores)
            rng.shuffle(cores)
            # Split by core request: 8 train / 2 valid / 2 test of each 12.
            assignment = ["train"] * 8 + ["valid"] * 2 + ["test"] * 2
            for core, split in zip(cores, assignment):
                for template in rng.sample(VARIANTS, variants_per_core):
                    c = fill(core, rng)
                    user = template.format(
                        c=c, c_lower=c[0].lower() + c[1:], domain=domain
                    ).strip()
                    splits[split].append(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": user},
                                {"role": "assistant", "content": label},
                            ],
                            "domain": domain,
                        }
                    )
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def main() -> None:
    out_dir = Path(__file__).parent / "data" / "router"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = build()
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        labels = Counter(r["messages"][-1]["content"] for r in rows)
        print(f"{path}: {len(rows)} rows  {dict(sorted(labels.items()))}")


def demo() -> None:
    """Self-check: splits are disjoint by prompt, labels balanced, no placeholders left."""
    splits = build()
    prompts = {k: {r["messages"][1]["content"] for r in v} for k, v in splits.items()}
    assert not prompts["train"] & prompts["test"], "train/test prompt leakage"
    assert not prompts["train"] & prompts["valid"], "train/valid prompt leakage"
    assert not prompts["valid"] & prompts["test"], "valid/test prompt leakage"
    for name, rows in splits.items():
        assert rows, name
        labels = Counter(r["messages"][-1]["content"] for r in rows)
        assert set(labels) == set(LABELS), labels
        assert len(set(labels.values())) == 1, f"{name} unbalanced: {labels}"
        for r in rows:
            assert "{" not in r["messages"][1]["content"], r
    print("demo ok:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
