# Data

## Azure LLM Inference traces (optional)

`AzureLLMInferenceTrace_conv.csv`, `AzureLLMInferenceTrace_code.csv` — a sample
of real Microsoft Azure LLM inference service traces (timestamp, context
tokens, generated tokens). ThermaLoop runs without these (synthetic workload is
the default); they are an optional real-workload input loaded via
`thermaloop.workload.azure`.

**Source:** Microsoft Azure Public Dataset — https://github.com/Azure/AzurePublicDataset
**License:** CC-BY 4.0. These trace files remain under their original Creative
Commons license, separate from this repository's MIT license (which covers the
code and docs only).
**Citation:** Patel et al., *Splitwise: Efficient Generative LLM Inference Using
Phase Splitting*, ISCA 2024.
