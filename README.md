This code is originaly adopted from Andrej Karpathy implementation of the GPT paper. The only difference is that this is implemeted in tensorflow not pytorch.I thought it would be some sort of a challenge
if i tried to use a different library, and indeed it was quite a challange but fairly simple. Such type of implementations where you translate from one library to another helps alot in buiding
a deeper understanding of the framework/library you are using. If you encounter any issues while training the model feel free to comment on the issue. The dataset used is a dataset that contains conversational dialogue that i got from Kaggle


For the gpt_testbed_on_tiktoken you see that i have not decoded the sequence and that's because this error indicates that TikToken cannot find a mapping (key) for a specific value (encoded token) in its decoding dictionary. This could happen due to:

Missing Tokens: The encoded tokens in pred_tokens might not be present in the vocabulary TikToken was trained on.
Incorrect Encoding/Decoding: There might be a mismatch between the encoding and decoding processes

which is really interesting because I was wondering why Andrej Karpathy did not use the TikToken library which is used in training state-of-the-art models like GPT-4.The tiktoken library will only decode sequences that are present in its vocabulary during training so if a model produces jibberish it will raise an error like this 'PanicException: no entry found for key'
