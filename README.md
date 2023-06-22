Pretty hacky addition of Penn Treebank, and some code to run the benchmarks. Entrypoint here is `evaluate.py`. You should be able to clone this repo, run `pip install -e .` and then `python evaluate.py`.


# Possible Problems
- General stages of grief ([here](https://github.com/stanford-crfm/mistral/issues/12))
  - > Basically no one knows how to reproduce the numbers
  - > it does seem concerning that in the thread Sidd linked there are 10-20 ppl differences when we are talking about ppls around 10-20... And one wonders why old-school language modelling/specific perplexities was fraught...
  - https://twitter.com/ArmenAgha/status/1669084129261162497
  - https://twitter.com/BlancheMinerva/status/1666852161131626497?s=20
- raw vs processed WT2
- cloze LMs (BERT) vs non-masked language modeling
- base of the exponent (e generally, 2 for huggingface-of-6-months-ago)
  - > It depends on the dataset and we use whatever metric results have previously been reported with. So for WikiText or PTB we report perplexities with base e / nats while for enwik8 it's base 2 / bits. ([link](https://github.com/openai/gpt-2/issues/78#issuecomment-481412845))
- perplexity-per-word (more common a few years ago) vs per token (more common now)
- What do you divide by? Tokens in the original dataset, or tokens in the tokenized dataset
  - > I think it should be num_original_tokens, otherwise your perplexity would be affected by how you tokenize the data right? For fair comparison everyone should ideally report perplexity over the same number of token outputs (regardless of how it's tokenized internally). ([link](https://github.com/openai/gpt-2/issues/78#issuecomment-481409772))
- Stride window ([see](https://huggingface.co/docs/transformers/perplexity#:~:text=Running%20this%20with%20the%20stride%20length%20equal%20to%20the%20max%20input%20length%20is%20equivalent%20to%20the%20suboptimal%2C%20non%2Dsliding%2Dwindow%20strategy%20we%20discussed%20above.%20The%20smaller%20the%20stride%2C%20the%20more%20context%20the%20model%20will%20have%20in%20making%20each%20prediction%2C%20and%20the%20better%20the%20reported%20perplexity%20will%20typically%20be.))
  - > Another possibility is that for evaluation, instead of chunking the dataset and shoving it through, they pass it in 1 token at a time the same way they do for sample generation. This would significantly reduce the loss because the model wouldn't "forget" what it was doing at the beginning of each sequence. I'll try this later today. [link](https://github.com/huggingface/transformers/issues/483#issuecomment-483456603)
- "Note that on PTB and WT-2, both AWD-LSTM-MoS-BERTVocab and AWDLSTM-MoS-GPTVocab outperform the original AWS-LSTM-MoS models by 17.8 and 10.6 perplexity points respectively. This is likely due to the change in word vocabulary to a sub-word vocabulary." ([here](https://arxiv.org/pdf/1904.09408.pdf))
- "overlapping evaluation" as [here](https://arxiv.org/pdf/2103.10360.pdf).
- What's the actual dataset?
  - > Another source of confusion: wikitext-2 and wikitext-103 have the same validation and test set, but the different gpt-2 models have different scores in table 3. Which dataset did you report on in the paper? Training? But the quoted SotA for wikitext-2 in the table comes from a paper that evaluates on the test set. The paper for SotA for wikitext-103 isn't specified. ([link](https://github.com/openai/gpt-2/issues/78#issuecomment-484361129))
- Preprocessing of the dataset
  - > We discuss this in 3.1 Language Modeling section. Essentially we run invertible "de-tokenizers" to massage into a more friendly format, and scale losses according to the token ratio. ([link](https://github.com/openai/gpt-2/issues/78#issuecomment-467199421))
  - > When OpenAI created GPT-2, they also created a custom, non-standard lambada evaluation dataset. OpenAI also changed the metric for evaluation by counting number of times the last BPE token is predicted incorrectly instead of the last word. This produces a huge difference in performance score, totally over 10%. ([link](https://github.com/EleutherAI/lm-evaluation-harness/issues/356))
