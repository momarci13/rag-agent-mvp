# EU AI Act (Regulation (EU) 2024/1689) -- obligations relevant to model validation

> **SUMMARY / NOT VERBATIM.** Hand-written summary of provisions of the EU
> Artificial Intelligence Act that are relevant when validating AI or
> machine-learning models in financial institutions. Not the legal text.
> Attach the official text from EUR-Lex for article-level citations and
> check application dates for each obligation.

## Risk classification (Art. 6 and Annex III)

- AI systems listed in Annex III are high-risk unless an Art. 6(3)
  derogation applies and is documented.
- Annex III point 5(b): AI systems intended to evaluate the creditworthiness
  of natural persons or establish their credit score are high-risk, with the
  exception of systems used to detect financial fraud.
- Annex III point 5(c): risk assessment and pricing for natural persons in
  life and health insurance is high-risk.

## Risk management system (Art. 9)

A continuous, iterative risk management process runs through the whole
lifecycle: identify and analyse known and foreseeable risks, estimate and
evaluate them, adopt risk management measures, and test the system against
defined metrics and probabilistic thresholds before placing it on the market.

## Data and data governance (Art. 10)

Training, validation and testing data sets are subject to data governance
covering design choices, collection processes, preparation (annotation,
cleaning, aggregation), assumptions, availability and suitability,
examination for possible biases, and identification of data gaps. Data sets
are relevant, sufficiently representative and, to the best extent possible,
free of errors and complete for the intended purpose.

## Technical documentation (Art. 11 and Annex IV)

Documentation is drawn up before placing on the market and kept up to date.
Annex IV elements include: general description and intended purpose; design
specifications, architecture and computational resources; data requirements
and data sheets; human oversight measures; validation and testing procedures
with metrics used to measure accuracy, robustness and potentially
discriminatory impacts, and test logs; the risk management system; changes
over the lifecycle; and the post-market monitoring plan.

## Record-keeping (Art. 12)

High-risk AI systems technically allow automatic recording of events (logs)
over their lifetime, to support traceability, post-market monitoring and
identification of risk situations.

## Transparency to deployers (Art. 13)

Instructions for use state the intended purpose, the level of accuracy
(including metrics), robustness and cybersecurity tested, known or
foreseeable circumstances that may lead to risks, performance for specific
groups of persons, input data specifications, and human oversight measures.

## Human oversight (Art. 14)

Systems are designed so natural persons can understand capacities and
limitations, monitor operation, remain aware of automation bias, interpret
output correctly, decide not to use or override the output, and interrupt
the system.

## Accuracy, robustness and cybersecurity (Art. 15)

Appropriate levels of accuracy are declared in the instructions for use.
Systems are resilient to errors, faults and inconsistencies, and to attempts
to alter their use or performance (e.g. data poisoning, adversarial
examples). Systems that continue learning mitigate feedback loops.

## Deployer obligations for credit scoring (Art. 26 and Art. 27)

Deployers use the system per the instructions, assign human oversight to
competent persons, monitor operation and keep logs. Deployers that evaluate
creditworthiness of natural persons perform a fundamental rights impact
assessment before first use (Art. 27).
