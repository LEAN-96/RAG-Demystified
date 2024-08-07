# RAG-Demystified

Are you intrigued by the concept of  Retrieval-Augmented-Generation (RAG)  but find it intimidating due to its association with Artificial Intelligence (AI) and Natural Langauge Processing (NLP)? Look no further! RAG-Demystified is your gateway to understanding this groundbreaking technology without the need for prior expertise in AI.    

In this repository, our mission is simple: to break down RAG into easily digestible explanations that anyone can understand. Whether you're a student exploring the realms of artificial intelligence, a professional seeking to broaden your knowledge, or simply curious about the latest advancements in technology, RAG-Demystified is designed with you in mind.    

But we don't stop there. Recognizing that the AI community often assumes familiarity with concepts like  Large Language Models (LLMs) and the transformer architecture, we go the extra mile to ensure that even the basics are accessible to everyone.    

Join us on a journey of discovery as we unravel the mysteries of RAG and provide you with the tools to navigate this exciting field with confidence. By the end of your exploration with RAG-Demystified, you'll have a solid understanding of not only RAG itself but also the foundational concepts that underpin it. Welcome to RAG-Demystified, where complexity meets clarity. 


![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/fd5f90d8-c481-4200-8c9e-d48250f5dba6) 



# Fundamentals of Large Language Models (LLMs) 

Large Language Models (LLMs) refer to advanced types of AI models, typically based on Transformer architectures, that have hundreds of billions (or more) parameters. These models are trained on vast amounts of text data, examples being GPT-3, PaLM, Galactica, and LLaMA. LLMs have impressive capabilities in understanding natural language and solving complex tasks through text generation. 

LLMs are not just static tools; they exhibit remarkable emergent abilities. This means that as these models interact with users and tasks, they demonstrate unexpected skills or talents that weren't explicitly programmed. Here are some key emergent abilities: 

1. Instruction Following: 

LLMs can follow given instructions well during inference (when they are generating outputs). 
Example: If you ask an LLM to write a story in the style of a famous author, it can produce a surprisingly accurate piece even if it wasn't specifically trained for that task. 

2. In-Context Learning: 

LLMs can refine their responses based on the context provided within the prompt itself. This helps them produce more relevant and coherent outputs. 
Example: When given a few examples of a specific task within the prompt, like translating sentences, the model can learn and apply the pattern immediately. 

3. Step-by-Step Reasoning: 

LLMs can perform complex tasks involving multiple reasoning steps, such as solving mathematical word problems. 
Example: Through techniques like chain-of-thought prompting, LLMs can break down and solve problems that require several logical steps. 

Source: [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) 

# LLM Architecture: Transformer 
The Transformer architecture, introduced by [Vaswani et al. in 2017](https://arxiv.org/abs/1706.03762), revolutionized natural language processing (NLP) by enabling efficient parallelization and handling long-range dependencies in text. It has become the foundation for many state-of-the-art language models. 

The Transformer consists usually of an encoder and a decoder, each composed of multiple identical layers. The encoder-decoder architecture is a powerful neural network design used for sequence-to-sequence tasks, such as machine translation.

The encoder processes the input sequence and generates a set of continuous representations, while the decoder uses these representations (depicted as state) to produce the output sequence. 
Each encoder layer has two main components: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder layers are similar but include an additional multi-head attention mechanism that attends to the encoder's output. 

![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/fa005044-d1ab-4ac3-b55f-9f8ab51456a7) 

Before LLMs can generate text, they must process the input text in a language they can understand. Lets break down the encoder-decoder process step-by-step: 

## Encoder

1. The input sequence (e.g., a sentence in English) is tokenized into individual words or subwords. This process of converting a sequence of text into tokens is called Tokenization. The goal is to break down the text into manageable pieces that the model can process. 

2. Each token is converted into an embedding, a numerical representation that captures its meaning. In other words, they mapped to a high-dimensional vectors - or a long lists of values. The values in an embedding represent the linguistic characteristics of a word. This layer is essentially a lookup table in the training data where each token has a corresponding vector. Tokens or words with similar meaning their vectors are closer to each other.

3. Positional Encoding is crucial because Transformers do not have a built-in sense of the order of tokens. Positional encodings are added to the token embeddings to provide information about the position of each token in the sequence. 

4. The embeddings with positional encodings are passed through multiple layers of self-attention and feed-forward networks in the encoder. Self-Attention Mechanism allows the model to weigh the importance of different tokens in the sequence when encoding a particular token. The self-attention mechanism is key to understanding context. By allowing each word to attend to every other word in the sentence, the Transformer can capture long-range dependencies and relationships regardless of their distance in the sequence. The context is now added as "encoded state" and ready to be passed along to the decoder. These embeddings are also called context vectors or transformer embeddings.

## Decoder
1. The decoder takes the context vectors from the encoder and starts generating the output sequence. It uses self-attention to focus on the previously generated words and encoder-decoder attention to focus on the input sentence. At each step, the decoder uses the context vectors and the previously generated tokens to predict the next token. This process continues until the entire output sequence (e.g., a sentence in French) is generated.

2. The final output sequence is produced, which can be a translated sentence, a summary, or any other sequence-based task.


The Transformer architecture is a powerful model that processes text efficiently and understands context through self-attention. By breaking down the input into embeddings, adding positional information, and using multiple layers of attention and feed-forward networks, it can generate accurate and contextually relevant outputs. Here how the entire architecture is illustrated in the initial transformer [paper](https://arxiv.org/abs/1706.03762):

![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/812bdc85-050c-4aa7-ad59-7644694c237f)


## Decoding strategies

Decoding strategies are methods used by language models to generate text based on a given input prompt. These strategies determine how the model selects the next word in a sequence. Let's focus on two popular decoding strategies: Greedy Search and Beam Search.

### Greedy Search

At each step, the model looks at the probabilities of all possible next words and selects the one with the highest probability. Once the most probable word is chosen, it becomes part of the output sequence, and the process repeats for the next word. This method is computationally efficient because it only considers the most likely word at each step, without evaluating other potential sequences. However, Greedy search might miss out on better overall sequences because it doesn't consider the long-term context. It only focuses on the immediate next word, which can lead to less coherent or meaningful sentences.

### Beam Search

Instead of selecting just one word at each step, beam search keeps track of the top N most likely sequences (where N is the beam size). At each step, the model expands each of the N sequences by considering all possible next words. It then keeps the top N sequences based on their combined probabilities. This process continues until the model generates an end-of-sequence token or reaches the maximum length. The sequence with the highest overall score is chosen as the final output. Beam search is more computationally intensive than greedy search because it evaluates multiple sequences simultaneously.

![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/dc6914dd-fbee-4bc3-9735-f855795fa6f9)


[Figure](https://heidloff.net/article/greedy-beam-sampling/)

Sources:

[D2L - Encoder-Decoder](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html)

[D2L - Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html)

[Generative AI](https://ig.ft.com/generative-ai/)

[A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)

[Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)

# Limitations of Large Language Models (LLMs)

While Large Language Models (LLMs) are powerful tools, they come with several important limitations:

1. Hallucinations and Truthfulness

LLMs can sometimes generate information that is incorrect or entirely made up, a phenomenon known as "hallucination." This issue becomes more problematic when combined with other concerns like bias, user privacy, and security risks. Moreover, LLMs act as "black boxes," meaning we can't easily understand how they make their predictions. Because LLMs do not have a notion of "truth" and are often trained on a mix of reliable and unreliable content, they can produce very plausible but incorrect answers.

2. Fixed Knowledge

LLMs have a fixed amount of knowledge based on what they learned during their training (known as parametric memory). They can't adapt to new information, making them less effective for tasks requiring the latest or highly detailed knowledge, especially in specialized fields. This limitation can lead to irrelevant or even harmful content. Expanding an LLM's knowledge through fine-tuning is expensive and resource-intensive.

3. Technical Restrictions
   
LLMs have technical limitations, such as the amount of text they can process at once (token limits). This affects their ability to handle large blocks of text, making it challenging to use them for bigger projects.

4. Lack of State/Memory
   
LLMs do not have memory or state. They cannot remember information from previous prompts, which is a significant limitation for applications that require continuity or context over multiple interactions.

5. Stochastic Nature

LLMs are stochastic or probabilistic. Sending the same prompt multiple times can yield different responses each time. While parameters like "temperature" can limit this variability, it remains an inherent property of their training.

6. Resource-Intensive

LLMs are generally very large, requiring many costly GPU machines for training and serving. This size can also lead to poor service level agreements (SLAs), particularly in terms of latency.

Source: [Large Language Models: A Survey](https://arxiv.org/html/2402.06196v2#S4)

# Optimize LLMs output
There are four approaches to enhance the output of LLM´s, where each of them ranges from easy and cheap to difficult and expensive to deploy. Here we will explain three of them and what sets them apart, before we discuss the fourth, Retrieval-Augmented Generation in the next chapter.


![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/db891ba4-687c-4d40-b4dd-34aa13f1b54d)

[Figure](https://www.fiddler.ai/blog/four-ways-that-enterprises-deploy-llms?WT.mc_id=academic-105485-koreyst)

## Pre-Training
Pre-Training is like a student attending a comprehensive school where they learn a wide range of subjects. During this phase, the student (or the model) absorbs a vast amount of general knowledge from various textbooks, articles, and other educational materials. This process equips the student with a broad understanding of many topics, even though they might not be an expert in any specific area.

In the context of LLMs, pre-training involves training the model on massive amounts of text data from the internet, books, and other sources. The model learns to predict the next word in a sentence, which helps it understand language patterns, grammar, and general knowledge. This phase is crucial because it builds the foundational knowledge that the model will use later. However, Pre-Training a model from scratch is really expensive and resource-intensive. 

## Fine-Tuning
Fine-Tuning is like a student who, after attending general school, decides to specialize in a particular subject, such as mathematics or history. The student now focuses on specific textbooks and materials related to their chosen field, refining their knowledge and skills to become an expert in that area. 

For LLMs, fine-tuning involves taking the pre-trained model and training it further on a smaller, task-specific dataset. This process helps the model adapt its general knowledge to perform well on specific tasks, such as answering questions about medical information or generating code. Fine-tuning adjusts the model's parameters to better suit the particular requirements of the task at hand. Although, fine-tuning is a good approach to optimize LLMs for a specific task, it can be very resource intensive and time consuming.

## Prompt Engineering 

A prompt is like a question or a set of instructions you give to an AI model to get a specific response. Prompt engineering is the art of crafting these questions or instructions to get better and more accurate answers from the AI. It involves minimal changes to the model and external information, concentrating on leveraging the inherent capabilities of LLMs.

Think of prompt engineering as giving directions. There are various techniques to structure these prompts. Some might involve clear, step-by-step instructions, while others might use examples to illustrate what you mean. Sometimes, these techniques can get quite sophisticated, involving conditional steps or branching paths.

Providing clear and specific prompts helps the AI understand exactly what you want. This context can be given in various ways, such as by including examples or detailed instructions within the prompt itself.

In-Context Learning (ICL) is a technique where the model is taught by providing examples or instructions directly within the prompt. This method allows the model to learn and adapt to specific tasks without the need for additional training. ICL can be particularly useful for tasks that require the model to understand and replicate specific patterns or formats.
Giving the model examples is referred as Shot Prompting.

One-Shot Prompting: You give the AI one example to help it understand what you want.

Few-Shot Prompting: You give the AI a few examples to make things even clearer.

These examples are included in the prompt to provide the necessary context, helping the AI give more accurate answers.

Source: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/pdf/2406.06608):


## Retrieval-Augmented Generation (RAG)
*Spoiler: The following terms, formulas and explaination are based on the [original RAG paper](http://arxiv.org/abs/2005.11401) and [DPR paper](https://arxiv.org/abs/2004.04906) released by a research team from Facebook AI.*
![RAG_Origin_Terms_Heatmap](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/83d8af41-cc3d-47da-87da-24cda82c1c44)
*Bibliometric RAG-Terms Heatmap based on the main papers mentioned in this repository using the tool VOSViewer.*


To address the limitations of LLMs, the field of AI has introduced a standardized approach to improve the performance of pre-trained language models on various knowledge-intensive tasks called Retrieval-Augmented Generation (RAG). RAG was introduced by [Lewis et. al](http://arxiv.org/abs/2005.11401) as a novel approach to natural language processing (NLP) tasks that require access to external knowledge sources (Non-parametric knowledge).  The authors uses the term "general-purpose fine-tuning recipe", which reflects RAG's utility in broadly enhancing pre-trained models across various tasks by integrating retrieval mechanisms.

RAG builds upon the advancements in large language models (LLMs) like GPT (Generative Pre-trained Transformer) models, integrating retrieval-based methods to enhance the generation process and to overcome the fixed amount of knowledge (Parametric knowledge). Understanding RAG begins with grasping the main ideas of retrieval-based and generation-based approaches in NLP. RAG works similarly to a typical sequence-to-sequence (seq2seq) model, where it takes one sequence x as input and produces a corresponding sequence y as output. Generation-based models in an traditional seq2seq model or LLM focus on creating text solely based on the input x without looking at external sources to produce output y. So, the output is solely generated based on the knowledge (parametric knowledge) the model was trained on. However, what sets RAG apart is that it adds an extra step. Instead of directly sending the input x to the generator, RAG uses retrieval-based methods, which involve finding useful information from document z, like databases, to help with generating text. 

RAG combines these two methods by using embeddings models, a vector index to find relevant context from external sources and a generator component to produce text based on both the input and retrieved context. For this purpose two independent embedding models (bi-encoder architecture), in this case [BERT models](https://arxiv.org/abs/1810.04805) were used: An document encoder [BERT](https://huggingface.co/google-bert/bert-base-uncased) and a fine-tuned query encoder BERT. The authors refer to this as [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (DPR). [BERT](https://arxiv.org/pdf/1810.04805), short for “Bidirectional Encoder Representations from Transformers,” is a transformer-based language model trained with massive datasets to understand languages like humans do. Like Word2Vec, BERT can create word embeddings from input data it was trained with. Additionally, BERT can differentiate contextual meanings of words when applied to different phrases. Therfore, it understands words not just on their own but in the context they’re used in. For example, BERT creates different embeddings for ‘play’ as in “I went to a play” and “I like to play.” This makes it better than models like Word2Vec, which don’t consider the words around them. Plus, BERT can handle the position of words really well, which is important.

Let's explore RAG through the analogy of a student attending an exam:
Think of a pre-trained Large Language Model (LLM) as a closed-book exam where the student relies solely on their memorized knowledge without referring to any materials. They're expected to answer questions based on what they've learned beforehand. Now, picture RAG as an open-book exam for the same student, but with a twist: they haven't studied! In this scenario, the student can access a textbook during the exam, similar to how RAG integrates an external knowledge base with the language model. In essence, RAG is like an open-book exam without studying, where the student (or the model) can access additional resources but must still navigate through them to find the correct information, while pre-trained LLMs are more like closed-book exams where the model can only use what it already knows.

![RAG_Analogy](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/92b40321-be9d-44c0-9c25-96900ad67ef4)



## Key components used


+ Generator = Encoder-Decoder pre-trained Seq2Seq Transformer, [BART-large](https://arxiv.org/abs/1910.13461).

+ Query Encoder = Dense Passage Retrieval (DPR), Fine-Tuned [BERT](https://huggingface.co/google-bert/bert-base-uncased) 

+ Document Encoder = [BERT](https://huggingface.co/google-bert/bert-base-uncased) 

+ Parametric knowledge = Knowledge thats implictly stored in the weights of the neural network (BART-large)

+ Non-Parametric knowledge = [FAISS](https://github.com/facebookresearch/faiss) Vector Index, consisting of 22 million * 100 encoded words chunks from wikipedia



Sources:

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)

[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)

# Benefits of RAG

RAG improves the quality of the text generated by the model and ensures that it's accurate and up-to-date. This approach also helps to reduce the problem of generating incorrect information, known as "hallucinations," by making the model rely on existing documentation. With RAG, we enhance the abilities of a large language model (LLM) by adding relevant context from a knowledge base. This context is inserted into the prompt, leveraging the LLM’s in-context learning abilities to produce more accurate and relevant responses.

Additionally, RAG enhances transparency and helps with error checking and copyright issues by clearly citing the sources of information used in the generated text. It also allows for private or specialized data to be incorporated into the text, ensuring that the output is tailored to specific needs or is more current.

One advantage is its ability to reduce the need for frequent retraining (fine-tuning) of the model. Unlike traditional approaches, where the entire model must be retrained with new documents to change what it knows, RAG simplifies this process. By updating the external database with new information, RAG eliminates the need for the model to memorize everything, saving time and computational resources. This flexibility allows us to control what the model knows simply by swapping out the documents used for knowledge retrieval.

Studies also show that RAG is outperforming fine-tuning:

![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/ab339e5d-be98-4b38-ab0a-8dd19760bc0c)
[Figure](https://arxiv.org/pdf/2312.05934)


Sources:

[Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

[Fine-Tuning or Retrieval?Comparing Knowledge Injection in LLMs](https://arxiv.org/pdf/2312.05934)


# Naive RAG
In this repository we will focus mainly on the Naive RAG approach. 

*The following description of the RAG-process is taken from a [survey](http://arxiv.org/abs/2312.10997) and the terminilogy with citatations from the original papers is applied to it. Since for the two papers two formula expressions were used both are written down.*

The basic RAG process involves three phases: Indexing, retrieval, and generation. The entire RAG-Process is often referred as "RAG-Pipeline".

**High-Level RAG Architecture:**

![RAG_Updated](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/4ed80cc8-388c-44db-87d9-547cdb8fb757)


## Indexing
Transforming unstructured data into a format that AI models can efficiently process is a key step in setting up RAG. Therefore, Indexing is a crucial step in preparing data. It involves several key steps:

1. Data Transformation: We start by getting our data ready. This involves cleaning up and extracting information from different formats like PDFs or web pages. Then, we make sure everything is in a standard text format that the model can understand.

2. Chunking: Next, we break the text into smaller pieces, kind of like cutting a big cake into slices. This helps the model handle the information better, especially since it can only process a certain amount at a time (Limited context window). Chunk size plays a role in how our system understands information.
The ideal length of your chunk (chunk size) depends on your use case. For question answering, shorter, more specific chunks are often needed. For summarization, longer chunks may be more suitable. If a chunk is too short, it may lack sufficient context, while if it's too long, it might include too much irrelevant information.
So, finding the right balance is key to getting relevant results.

3. Encoding and Vectorization: Now comes the tricky part: Turning words into numbers! We use embedding models to do this, which help our computer understand the meaning behind the words and how they're related to each other. These models, also known as encoding models or bi-encoders, are trained on a large corpus of data, making them powerful enough to encode chunks of documents into single vector embeddings. Vector embeddings are dense representations of objects in a continuous high-dimensional vector space, capturing semantic relationships between objects through distance and direction. Referencing back to [Dense passage retriever model](https://github.com/facebookresearch/DPR) from Facebook AI, the index encoding model utilized the document encoder BERT.

"*d(z) is a dense representation of a document produced by a BERT_BASE document encoder*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$\text{d(z)= BERT}_{d}(z)$$
==
$$\text{E}_{P}(p)$$
 
4. Storage in Vector Database: Finally, we store these encoded chunks in a vector database. This specialized database is designed to manage and search embedded vectors. This makes it easy for the retriever to find what it needs quickly when we ask it questions later on. Similar passages are stored closer to each other as dense vector representations. Researcher from Facebook used an [FAISS](https://github.com/facebookresearch/faiss) index as vector database.

"*During inference time, we apply the passage encoder EP (p) to all the passages and index them using FAISS ofﬂine. FAISS is an extremely efﬁcient, open-source library for similarity search and clustering of dense vectors, which can easily be applied to billions of vectors.*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)

The entire index process is described as followed:

"*Given a collection of M text passages, the goal of our dense passage retriever (DPR) is to index all the passages in a low-dimensional and continuous space, such that it can retrieve efﬁciently the top k passages relevant to the input question for the reader at run-time. Our dense passage retriever (DPR) uses a dense encoder EP (·) which maps any text passage to a d- dimensional real-valued vectors and builds an index for all the M passages that we will use for retrieval.*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)


![VectorDB](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/d4d93974-2a75-4496-bfc0-24854da8137b)

[Figure](https://weaviate.io/blog/what-is-a-vector-database)


## Index Visualization
When it comes to seeing our index in action, we rely on a handy tool called [RAGmap](https://github.com/JGalego/RAGmap). RAGmap is a simple RAG visualization tool for exploring document chunks and queries in embedding space. It enables to create a vector store and reduces high-dimensional vectors to 2D and 3D vector space. For our embedding model, we also used [BERT_BASE](https://huggingface.co/google-bert/bert-base-uncased) and indexed the original [RAG paper](http://arxiv.org/abs/2005.11401) to demonstrate how it works. Each chunk contains 256 characters and a chunk overlap of 25 characters. t-SNE algorithm is breaking down the encoded 768 dimensions of BERTd(z) and BERTq(x) into a 2-dimensional space.

Here's what the visualization in a 2D space looks like:

![BERT_RAG_Index_2D_t-SNE_Top3_chunk256 (1)](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/b896b4ee-54ba-4827-a140-f6b5f380970b)

In contrast, we ingested the same document using a different embedding model with only 384 dimensions, [BGE-Small](https://huggingface.co/BAAI/bge-small-en-v1.5). As you can see the similarity distribution differs from the previous BERT embeddings. 
![RAG_Index_2D_t-SNE_Top3_chunk256_1_out](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/04c55cb0-ffdd-4278-be41-5a018f103733)

The visualizations indicate, that BERT_Base seems to provide better embeddings for the given text chunks. It shows more distinct and compact clusters, indicating better preservation of both local and global structures in the data. BGE-Small shows a more spread-out distribution with less distinct clustering. The data points are more uniformly distributed, which might indicate less effective separation of different groups. However, choosing the right embedding model also depends on the task and the data you want to retrieve. A collection and benchmark of different embedding models can be found [here]([Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

## Retrieval
Retrieval in RAG involves fetching highly relevant context from a retriever. Here is how it works:

1. Encoding of User Query: The user query is processed and encoded into a representation that the system can work with. The retriever transforms the question into a vector using the fine-tuned query encoder BERT.

"*q(x) is a query representation produced by a query encoder, also based on BERT_BASE.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$\text{q(x)= BERT}_{q}(x)$$
==
$$\text{E}_{Q}(q)$$

"*Given a question q at run-time, we derive its embedding vq = EQ(q) and retrieve the top k passages with embeddings closest to vq.*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)


2. Document Retrieval: Using the encoded query, the system searches a large corpus of information to retrieve relevant documents or passages. This search, also called vector search or similarity search, finds top K document chunks within the indexed corpus by calculating similarity scores between the query vector and the document chunk vectors. For this purpose mathematical operations like Euclidean distance, Cosine similarity, Manhattan distance or Dot product can be applied measuring the distance between vector representations to determine their similarity. In the paper the similarity between the question and the document passage is using the dot product, a Maximum Inner Product Search (MIPS) algorithm.  This step is implemented within the FAISS index and prepares the relevant documents for the generation process.

![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/394f7ab3-e2cc-4abf-8f0d-fba5ad1d6b8a)

[Figure](https://weaviate.io/blog/what-is-a-vector-database)

"*Calculating top-k(pη(·|x)), the list of k documents z with highest prior probability pη(z|x), is a Maximum Inner Product Search (MIPS) problem, which can be approximately solved in sub-linear time.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

"*At run-time, DPR applies a different encoder EQ(·) that maps the input question to a d-dimensional vector, and retrieves k passages of which vectors are the closest to the question vector. We deﬁne the similarity between the question and the passage using the dot product of their vectors:*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)

$$p_{\eta}(z|x) \propto \exp(d(z)^{\top}q(x))$$
==
$$\text{sim}(q, p) = E_Q(q)^\top E_P(p)$$

$$\text{sim}(q, p) = p_{\eta}(z|x) \propto \exp(d(z)^{\top}q(x))$$$$

Both formulas are about finding the best match between a question and some passages.

Let's break down this formula step by step:
==
This symbol tells us that the probability of a document z given a query x is proportional to the similarity between the document and the query. In other words, the more similar they are, the higher the probability that the document is relevant to the query.

$$p_{\eta}(z|x) \propto$$
==
This part calculates how similar the document's representation d(z) is to the query's representation q(x). Think of it like comparing two pictures to see how much they look alike.

$$\\exp(d(z)^{\top}q(x))$$
==
It's a way to represent the document's content in a dense format.

$$\text{d}(z)$$
==
Similarly, this represents the query in a dense format. It's also encoded into a format that the system can understand easily.
 
$$\text{q}(x)$$
==

So, in simpler terms, this formula is like a way for the system to figure out which documents are most likely to match a given question. It does this by comparing how similar each document is to the question. And by doing this comparison, it can quickly find the top documents that are the best match to the query.


## Retrieval Visualization  

For the BERT embedding model the Top-3 results based on our initial query looks like following:
![BERT_RAG_Retrieval_2D_t-SNE_Top3_chunk256 (1)](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/ff7a7f90-7834-4f75-839e-3385d85d7364)

For demonstration purposes we can also display our query, retrieved text chunks and their corresponding 2-D vectors as a table:

| id | x | y | chunk |
|-----------------|-----------------|-----------------|-----------------|
| 0 (Query) | -20.651572193624503 | 2.5121936727809024 | What is Retrieval-Augmented Generation (RAG)? |
| 11 | -20,372099632067037 | 2,6042801988217015 | memory with non-parametric (i.e., retrieval-based) memories [ 20,26,48] can address some of these<br>issues because knowledge can be directly revised and expanded, and accessed knowledge can be |
| 4 | -23,154010314955553 | 3,0623941846106097 | memory have so far been only investigated for extractive downstream tasks. We explore a general-<br>purpose ﬁne-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-<br>trained parametric and non-parametric mem- |
| 70 | -20,614559198905972 | 1,5474185954633568 | documents, relying primarily on non-parametric knowledge. We also compare to “Closed-Book QA”<br>approaches [ 52], which, like RAG, generate answers, but which do not exploit retrieval, instead |

Same visualization for the BGE-Small embedding model. For this model all three retrieved results seems to be close to the query:
![RAG_Retrieval_2D_t-SNE_Top3_chunk256_1_out](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/c1fe9d3d-1151-4151-9c30-3b893fb1eae0)

Before we jump to the generation phase a short visualized recap about the formulas we learned so far for indexing and retrieval:
![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/b77511bb-85c4-49d5-844c-948995dd8c2a)
[Figure](https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval)

## Generation 
For the generator the authors used [BART-large](https://arxiv.org/abs/1910.13461), a pre-trained seq2seq transformer with 400M parameters. However, for applying RAG any generator (LLM) can be utilzed.


1. Integration of Context: Once the documents are encoded, they are ready to be combined with the encoded query. This expanded context is then incorporated into the prompt and given to the LLM for generating a response. The model weighs each of the documents using a retrieval score when generating the answer.

"*To combine the input x with the retrieved content z when generating from BART, we simply concatenate them.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

"*To decode, we can plug p′θ(yi|x, y1:i−1) into a standard beam decoder.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$p_{\theta}(y_i|x, z, y_{1:i-1})$$
==

In practice, the question and retrieved documents are injected in a prompt and given to the generator (LLM) to answer the question based on the retrieved documents. This is called prompt injection. For the [Dense Passage Retriever (DPR)](https://github.com/facebookresearch/DPR) the authors used a JSON format which looks like following:

    [
        {
            "question": "What is Retrieval-Augmented Generation (RAG)?",
            "answers": ["memory have so far been only investigated for extractive downstream tasks. We explore a general-<br>purpose ﬁne-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-<br>trained parametric and non-parametric mem-", "memory with non-parametric (i.e., retrieval-based) memories [ 20,26,48] can address some of these<br>issues because knowledge can be directly revised and expanded, and accessed knowledge can be", "documents, relying primarily on non-parametric knowledge. We also compare to “Closed-Book QA”<br>approaches [ 52], which, like RAG, generate answers, but which do not exploit retrieval, instead" ],
            "ctxs": [
                {
                    "id": "4", "11", "70" # passage id from database tsv file
                    "title": "RAG for Knowledge-Intensive NLP Tasks",
                    "text": "....",
                    "score": "...",  # retriever score
                    "has_answer": true|false
         },
    ]

However, nowadays tools like [LangChain](https://www.langchain.com/) and [LlamaIndex](https://www.llamaindex.ai/) uses prompt templates, where the question (query) and context is passed to the generator through variables. The context contains the top-k relevant documents based on the question:

      Answer the question based on the context:
      
      Question: {question}

      Context: {context}

      Answer:

For the generation part the authors propose two RAG model variants to decode from this set of latent documents and for producing distributions over the generated text : 

"*We marginalize the latent documents with a top-K approximation, either on a per-output basis (assuming the same document is responsible for all tokens) or a per-token basis (where different documents are responsible for different tokens).*"

"*RAG employs a form of late fusion to integrate knowledge from all retrieved documents, meaning it makes individual answer predictions for document-question pairs and then aggregates the final prediction scores. Critically, using late fusion allows us to back-propagate error signals in the output to the retrieval mechanism, which can substantially improve the performance of the end-to-end system.*" (Lewis P, Riedel S, Kiela D, Piktus A [Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/))

## RAG-Token Model

Instead of picking one document and generating the entire answer from it, the RAG-Token model looks at multiple documents for each word (or token) to decide the best word to use next in the answer. Imagine you are writing an answer word by word. For the first word, the model considers all K documents and decides the best word based on the information from all of them. After choosing the first word, it repeats the process for the second word, again considering the top K documents to pick the best one. For each word, the model creates a distribution (a set of possible words) from each document based on the retrieval score and the previously generated token. It then combines these distributions to decide the final word. This process, a standard beam search, ensures the answer is well-informed by considering all the retrieved documents. This method allows the model to draw information from different sources for each word, making the answer more comprehensive and accurate. It doesn't rely on just one document for the entire answer but uses multiple documents dynamically for each step.


"*The RAG-Token model can be seen as a standard, autoregressive seq2seq generator with transition probability:*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)


$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i}^N \sum_{z \in \text{top-k}(p(⋅|x))} p_{\eta}(z|x) p_{\theta}(y_i | x, z, y_{1:i-1}) ]$$
==

Let's break down this formula step by step:

The formula calculates the probability of generating a sequence of words 𝑦 given an input 𝑥.

$$p_{\text{RAG-Token}}(y|x)$$
==

This product notation indicates that we are going to perform the following steps for each token 𝑦𝑖 in the sequence. Here, 𝑖 ranges from 1 to 𝑁, where 𝑁 is the length of the sequence.

$$\approx \prod_{i}^N$$
==
For each token 𝑦𝑖, we sum over the probabilities for each of the top K documents 𝑧. This means we consider each document's contribution to generating the current token.

$$\sum_{z \in \text{top-k}(p(z|x))}$$
==

This term represents the probability that a document 𝑧 is relevant to the input 𝑥. It tells us how likely each document is to contain the information we need. This document probability is derived from the retrieval score of the document.

$$p_n(z|x)$$
==
This term represents the probability of generating the token 𝑦𝑖 given the input 𝑥, the document 𝑧, and all the previous tokens 𝑦1:𝑖−1. This captures how well the document 𝑧 helps in generating the current token.

$$p_{\theta}(y_i|x, z_i, y_{1:i-1})$$
==

This process allows the RAG-Token model to effectively combine retrieval and generation on a token-by-token basis.




## RAG-Sequence Model

The RAG-Sequence model uses the same retrieved document to generate the entire output sequence.These document act as a consistent source of context throughout the generation process. This process can be seen as a modified beam search, that means beam search is run for each document z.
The decoding process in RAG-Sequence can be approached in two ways, where both run a initial beam search for each retrieved document:

1. This strategy involves running additional forward passes for each document where the hypothesis does not appear in the beam search. The generator probability is then multiplied by the retriever probability p η (z∣x), and the probabilities are summed across all beams to get the final marginal probabilities.
"*We refer to this decoding procedure as “Thorough Decoding.*” (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

2. This strategy approximates the probability p θ (y∣x,zi) as zero for any hypothesis not generated during the initial beam search. This method avoids the need for additional forward passes, making it more efficient, especially for longer output sequences.
"*We refer to this decoding procedure as “Fast Decoding.*” (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)



![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/850af0df-ce34-4f57-9d44-1da520298807)

[Figure](https://sharif-llm.ir/assets/lectures/LLM-RAG.pdf)


$$p_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-k}(p(⋅|x))} p_n(z|x) p_θ(y|x, z) = \sum_{z \in \text{top-k}(p(⋅|x))} p_n(z|x) \prod_{i}^N p_θ(y_i | x, z, y_{1:i-1})$$


## Lets break it down again:

This represents the probability of generating the sequence y  given the input 𝑥

$$p_{\text{RAG-Sequence}}(y|x)$$
==
This sums over the top-k relevant documents 𝑧 retrieved based on the input 𝑥

$$\sum_{z \in \text{top-k}(p(z|x))}$$
==
This term represents the probability that a document 𝑧 is relevant to the input 𝑥. It tells us how likely each document is to contain the information we need.


$$p_n(z|x)$$
==
This term represents the probability of generating the entire sequence ( 𝑦 ) given the input 𝑥 and the retrieved document 𝑧. It captures how well the document 𝑧 helps in generating the answer. This document probability is derived from the retrieval score of the document.


$$p_{\theta}(y|x, z)$$
==
Here, ( 𝑦 ) is broken down into individual tokens (words or characters). For each token ( 𝑦𝑖 ) in the sequence, we calculate its probability given the input 𝑥, the document 𝑧, and all the previous tokens 
𝑦1:𝑖−1. We multiply these probabilities together to get the overall probability for the sequence. Each of these probabilities is weigthed by the parameters of the BART model (p θ) as well as the similarity that is determined the prior probability on that document which was retrieved.

$$\prod_{i=1}^{N} p_{\theta}(y_i | x, z, y_{1:i-1})$$



In summary, the RAG-Sequence and RAG-Token models provide robust solutions for knowledge-intensive generation tasks by leveraging both parametric and non-parametric memory. The RAG-Sequence model uses the same retrieved document to generate the entire output sequence, ensuring consistency and coherence in the generated text. On the other hand, the RAG-Token model allows for more flexibility by using different documents for each token, which can enhance the diversity and specificity of the generated content.Both models have shown to outperform traditional parametric-only models like BART in various tasks, including open-domain question answering and Jeopardy question generation. RAG-Sequence tends to produce more coherent and contextually consistent outputs, while RAG-Token excels in generating responses that combine information from multiple sources, leading to more factual and diverse outputs.

The entire RAG process summarized by the authors:

"*We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We combine these components in a probabilistic model trained end-to-end. The retriever (Dense Passage Retriever, henceforth DPR) provides latent documents conditioned on the input, and the seq2seq model (BART) then conditions on these latent documents together with the input to generate the output. We marginalize the latent documents with a top-K approximation, either on a per-output basis (assuming the same document is responsible for all tokens) or a per-token basis (where different documents are responsible for different tokens).*"

Sources:

[Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)

[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)

# Applications of RAG

Retrieval-Augmented Generation (RAG) has a wide range of use-cases that leverage its ability to retrieve and generate relevant information dynamically. In customer support chatbots, RAG can provide accurate and contextually relevant responses by pulling in information from a company's knowledge base. For documents Q&A and internal knowledge Q&A, RAG excels in retrieving specific information from vast document repositories, making it invaluable for corporate environments and research institutions. Agents powered by RAG can perform complex tasks by understanding and acting on user queries with high accuracy. In multimodal applications, RAG integrates text with other data forms like images and videos, enhancing the richness of interactions. Query analysis benefits from RAG’s capability to interpret and reformulate user queries for better search results. Additionally, RAG aids in structured data extraction by identifying and extracting key pieces of information from unstructured text, streamlining data processing workflows.

Sources: 

[LangChain](https://python.langchain.com/v0.1/docs/use_cases/)

[LlamaIndex](https://docs.llamaindex.ai/en/stable/use_cases/)

# Advanced RAG
Naive RAG, has some drawbacks. It may suffer from low precision, meaning it sometimes includes irrelevant information in the response, and low recall, where it may miss relevant information. Another issue is the possibility of the model receiving outdated information, leading to inaccurate responses and the risk of generating content that doesn't align with the user's needs. This can result in confusing or incorrect answers. Referencing back to our initial example of an student however, just like the student faces the challenge of finding the right answers amidst the sea of information in the textbook, Naive RAG may struggle to discern which information is relevant. 

Advanced RAG improves upon Naive RAG by enhancing retrieval quality through pre-retrieval and post-retrieval strategies. Pre-Retrieval strategies focuses on optimizing both the indexing structure and the original query. Key strategies include enhancing data granularity, optimizing index structures, adding metadata, and using query optimization techniques like query rewriting and expansion. In the post-retrieval process, the focus is on re-ranking the retrieved information to highlight the most relevant content and compressing the context to avoid information overload, ensuring only essential details are processed by the model.

The research field of RAG is moving fast, that´s why a lot of advanced RAG strategies are recently developed. If you want to dive deeper into the field I recommend the following material:

[Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

[Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473)

[RAG for LLMs](https://www.promptingguide.ai/research/rag)

# Modular RAG

Modular RAG is an advanced and flexible approach that builds on the principles of Naive and Advanced RAG. It introduces new modules and patterns to improve retrieval and generation processes. This modular architecture includes specialized components like a Search module for direct searches across diverse data sources, a Memory module leveraging LLM memory for better alignment with data distribution, and a Task Adapter module tailoring RAG for various downstream tasks. Modular RAG also supports iterative and adaptive retrieval processes, allowing dynamic reconfiguration and integration of new modules to address specific challenges, ultimately enhancing the quality and relevance of the generated content.

Source: [Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

# Conclusion

Retrieval-Augmented Generation (RAG) is an effective method for enhancing Large Language Models (LLMs) with domain-specific data without the need for frequent retraining or fine-tuning. This approach allows us to interact with data effortlessly while leveraging the capabilities of LLMs. Tools and frameworks such as LangChain, LlamaIndex, Flowise, and Langflow make it straightforward to implement RAG pipelines, from basic to advanced levels. As the field continues to evolve, we can expect more enterprises and organizations to adopt RAG within their operations. The progression towards agentic and multimodal RAG will extend the utility beyond text-based Q&A applications.

Recent studies highlight the shift towards multimodal RAG approaches. For example, a search for "RAG" or "Retrieval-Augmented Generation" on Lens.org, followed by a bibliometric analysis using [VOSViewer](https://www.vosviewer.com/), reveals terms like "query image", "augmented image and "visual question answering." These findings emphasize the potential of RAG in interacting with images, paving the way for innovative applications beyond traditional text-based interfaces.

![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/14043b0d-80a7-4d27-9a65-db688f362972)

# Resources and Further Reading
This is a collection of further readings and RAG application tools.

## Transformer
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Visual introduction to Transformers](https://www.youtube.com/watch?v=ISPId9Lhc1g)

[Visual introduction to Attention mechanism](https://www.youtube.com/watch?v=eMlx5fFNoYc)

[Generative AI](https://ig.ft.com/generative-ai/)

## Large Language Models

[A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)

[Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)

[LLM Visualization](https://bbycroft.net/llm)

[Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)

## RAG

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)

[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)

[Retrieval Augmentation Reduces Hallucination in Conversation](https://arxiv.org/abs/2104.07567)

[A Survey on Retrieval-Augmented Text Generation](https://arxiv.org/abs/2202.01110)

[Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

[Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473)

[Retrieval-Augmented Generation for AI-Generated Content: A Survey (Repository)](https://github.com/hymie122/RAG-Survey)

[RAG for LLMs](https://www.promptingguide.ai/research/rag)



# RAG Orchestration Frameworks & Tools

| Frameworks | Tools | Evaluation & Observability| Visualization |
|-----------------|-----------------|-----------------|-----------------|
| [LangChain](https://www.langchain.com/) | [FlowiseAI](https://flowiseai.com/) | [RAGAS](https://github.com/explodinggradients/ragas) | [RAGexplorer](https://github.com/gabrielchua/RAGxplorer/) | 
| [LlamaIndex](https://www.llamaindex.ai/) | [LangFlow](https://www.langflow.org/) | [Langsmith](https://www.langchain.com/langsmith) |[RAGmap](https://github.com/JGalego/RAGmap) |
| [Haystack](https://haystack.deepset.ai/) | [Verba](https://github.com/weaviate/Verba) | [Langfuse](https://langfuse.com/) | [ChunkVisualizer](https://huggingface.co/spaces/Nymbo/chunk_visualizer) |
| [Canopy](https://github.com/pinecone-io/canopy) |  [VectorShift](https://vectorshift.ai/) | [RAG-Arena](https://github.com/mendableai/rag-arena) | [ChunkViz](https://chunkviz.up.railway.app/) |
| [RAGFlow](https://github.com/infiniflow/ragflow?tab=readme-ov-file) | [Meltano](https://meltano.com/)  | [Auto-RAG](https://github.com/Marker-Inc-Korea/AutoRAG) | |
| [DSPy](https://github.com/stanfordnlp/dspy) | [Cohere Coral](https://cohere.com/coral)  | [RAGTune](https://github.com/misbahsy/RAGTune) | |
| [Cognita](https://github.com/truefoundry/cognita) |[Verta](https://www.verta.ai/rag)  |[Lunary](https://lunary.ai/) | 
| [Dify](https://github.com/langgenius/dify) | [Quivr](https://docs.quivr.app/intro) | [DeepEval](https://github.com/confident-ai/deepeval) |  |
|  | [RAGapp](https://github.com/ragapp/ragapp) | [Trulens](https://www.trulens.org/) |  |
|  |  |[RagaAI](https://docs.raga.ai/raga-llm-hub/test-execution/evaluation)  |  |
## LLM Stack

[LetsBuildAI](https://letsbuild.ai/)
