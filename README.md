# RAG-Demystified

Are you intrigued by the concept of **Retrieval-Augmented-Generation (RAG)** but find it intimidating due to its association with Artificial Intelligence (AI) and Natural Langauge Processing (NLP)? Look no further! RAG-Demystified is your gateway to understanding this groundbreaking technology without the need for prior expertise in AI.

In this repository, our mission is simple: to break down RAG into easily digestible explanations that anyone can understand. Whether you're a student exploring the realms of artificial intelligence, a professional seeking to broaden your knowledge, or simply curious about the latest advancements in technology, RAG-Demystified is designed with you in mind.

But we don't stop there. Recognizing that the AI community often assumes familiarity with concepts like **Large Language Models (LLMs)** and the transformer architecture, we go the extra mile to ensure that even the basics are accessible to everyone.

Join us on a journey of discovery as we unravel the mysteries of RAG and provide you with the tools to navigate this exciting field with confidence. By the end of your exploration with RAG-Demystified, you'll have a solid understanding of not only RAG itself but also the foundational concepts that underpin it. Welcome to RAG-Demystified, where complexity meets clarity.


![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/fd5f90d8-c481-4200-8c9e-d48250f5dba6)



# Fundamentals of Large Language Models (LLMs):

Large Language Models (LLMs) are not just static tools but exhibit remarkable emergent abilities. When we talk about emergent abilities we're referring to the unexpected talents or skills these models demonstrate as they interact with users or tasks. These capabilities can sometimes surprise us, showing functionalities that weren't explicitly programmed into the model. LLMs demonstrate various emergent abilities, such as instruction following, in-context learning and step-by-step reasoning. These capabilities include problem-solving, where LLMs provide insightful solutions to tasks they weren't trained on, adaptation to user inputs, and contextual understanding.

LLMs perform well when given instructions during inference, and they excel when provided with contextual reference data. They prioritize contextual knowledge over data they've been trained on, leading to more accurate responses.

In-context learning is another important ability, allowing LLMs to refine their responses based on specific contexts. This enhances the relevance and coherence of their outputs.

The emergence of these abilities distinguishes LLMs from smaller models. For example, LLMs like GPT-3 exhibit strong in-context learning abilities, allowing them to generate expected outputs without additional training. They can also follow instructions for new tasks without explicit examples, improving their generalization ability.

Furthermore, LLMs can solve complex tasks involving multiple reasoning steps, such as mathematical word problems, through strategies like the chain-of-thought prompting. This ability, along with others, emerges as LLMs scale up in size, contributing to their performance gains. Overall, emergent abilities in LLMs showcase their adaptability and intelligence, making them versatile tools for various tasks and applications. [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)

# Introduction to Transformers:

BERT, short for “Bidirectional Encoder Representations from Transformers,” is a transformer-based language model trained with massive datasets to understand languages like humans do. Like Word2Vec, BERT can create word embeddings from input data it was trained with. Additionally, BERT can differentiate contextual meanings of words when applied to different phrases. Therfore, it understands words not just on their own but in the context they’re used in. For example, BERT creates different embeddings for ‘play’ as in “I went to a play” and “I like to play.” 
This makes it better than models like Word2Vec, which don’t consider the words around them. Plus, BERT can handle the position of words really well, which is important.
BERT Transformer uses bidirectional self-attention.
[BERTBASE](https://huggingface.co/google-bert/bert-base-uncased) (L=12, H=768, A=12, Total Parameters=110M)

# Limitations of Large Language Models (LLMs):
Despite their impressive abilities, LLMs have notable limitations, especially when applied in real-world situations. One major issue is that they sometimes generate information that is incorrect or entirely made up, which is called "hallucination." This problem becomes worse when combined with issues like bias, user privacy concerns, and security risks. Moreover, LLMs acting as "black box" meaning we cannot provide insight into their predictions.

Another important limitation is that LLMs have a fixed amount of knowledge (Parametric memory). They only know what they learned during their training and can't adapt to new information. This makes them less effective in tasks that require the latest and most detailed knowledge, especially in specialized areas. In practical terms, this might mean they produce irrelevant or even harmful content. Expanding the LLM´s memory through fine-tuning is expensive and resource-intensive.

Additionally, LLMs have technical restrictions, such as limits on the amount of text they can process at once (Token limits). This can affect their ability to handle large blocks of text and may make it harder to scale their use for bigger projects. 

Parametric knowledge = Knowledge thats implictly stored in the weights of the neural network

Non-Parametric knowledge = Index 22 million * 100 encoded words chunks from wikipedia

# Introduction to RAG:

![RAG_Origin_Terms_Heatmap](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/83d8af41-cc3d-47da-87da-24cda82c1c44)



To address these limitations, the field of AI has introduced a general-purpose fine-tuning recipe called Retrieval-Augmented Generation (RAG). RAG was first introduced by [Lewis et. al](http://arxiv.org/abs/2005.11401) in 2020. The authors introduced RAG as a novel approach to natural language processing (NLP) tasks that require access to external knowledge sources (Non-parametric knowledge). RAG builds upon the advancements in large language models (LLMs) like GPT (Generative Pre-trained Transformer) models, integrating retrieval-based methods to enhance the generation process and to overcome the fixed amount of knowledge (Parametric knowledge).

Let's explore RAG through the analogy of a student attending an exam:
Think of a pre-trained Large Language Model (LLM) as a closed-book exam where the student relies solely on their memorized knowledge without referring to any materials. They're expected to answer questions based on what they've learned beforehand.

Now, picture RAG as an open-book exam for the same student, but with a twist: they haven't studied! In this scenario, the student can access a textbook during the exam, similar to how RAG integrates an external knowledge base with the language model. 

In essence, RAG is like an open-book exam without studying, where the student (or the model) can access additional resources but must still navigate through them to find the correct information, while pre-trained LLMs are more like closed-book exams where the model can only use what it already knows.

![RAG_Analogy](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/92b40321-be9d-44c0-9c25-96900ad67ef4)


Understanding RAG begins with grasping the main ideas of retrieval-based and generation-based approaches in NLP. RAG works similarly to a typical sequence-to-sequence (seq2seq) model, where it takes one sequence x as input and produces a corresponding sequence y as output. Generation-based models in an traditional seq2seq model  focus on creating text solely based on the input x without looking at external sources to produce output y. However, what sets RAG apart is that it adds an extra step. Instead of directly sending the input x to the generator, RAG uses retrieval-based methods, which involve finding useful information from document z, like databases, to help with generating text. 

$$\text{Sequence output}(y) = \ \text{Input query}(x) + \text{Retrieved documents}(z)$$
==

RAG combines these two methods by using [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (DPR) which is based on bi-encoder architecture to find relevant context from external sources and a generator component to produce text based on both the input and retrieved context. For this purpose two independent [BERT models](https://arxiv.org/abs/1810.04805) were used: An document encoder BERT and a fine-tuned query encoder BERT.


*d(z) is a dense representation of a document produced by a [BERT_BASE](https://huggingface.co/google-bert/bert-base-uncased) document encoder, and q(x) is a query representation produced by a query encoder, also based on [BERT_BASE](https://huggingface.co/google-bert/bert-base-uncased).* (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$\text{d}(z) = \text{BERT}_d(z), \quad\text{BERT}_q(x)$$
==
 
For the generator component the authors used a encoder-decoder pre-trained seq2seq transformer, [BART-large](https://arxiv.org/abs/1910.13461).

Sources:

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)

[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)

# Benefits of RAG:

RAG improves the quality of the text generated by the model and ensures that it's accurate and up-to-date. This approach also helps to reduce the problem of generating incorrect information, known as "hallucinations," by making the model rely on existing documentation. With RAG, we enhance the abilities of a large language model (LLM) by adding relevant context from a knowledge base. This context is inserted into the prompt, leveraging the LLM’s in-context learning abilities to produce more accurate and relevant responses.

Additionally, RAG enhances transparency and helps with error checking and copyright issues by clearly citing the sources of information used in the generated text. It also allows for private or specialized data to be incorporated into the text, ensuring that the output is tailored to specific needs or is more current.

One advantage is its ability to reduce the need for frequent retraining (fine-tuning) of the model. Unlike traditional approaches, where the entire model must be retrained with new documents to change what it knows, RAG simplifies this process. By updating the external database with new information, RAG eliminates the need for the model to memorize everything, saving time and computational resources. This flexibility allows us to control what the model knows simply by swapping out the documents used for knowledge retrieval.


# Naive RAG

**High-Level RAG Architecture:**

![RAG_Updated](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/4ed80cc8-388c-44db-87d9-547cdb8fb757)


The basic RAG process involves indexing, retrieval, and generation. In simple terms, a user's input x is used to search for relevant documents z, which are then combined with a prompt and given to the model to create a final output y. If the interaction involves multiple turns, previous conversation history can be included in the prompt. The entire RAG-Process is often referred as "RAG-Pipeline".

However, this approach, known as Naive RAG, has some drawbacks. It may suffer from low precision, meaning it sometimes includes irrelevant information in the response, and low recall, where it may miss relevant information. Another issue is the possibility of the model receiving outdated information, leading to inaccurate responses and the risk of generating content that doesn't align with the user's needs. This can result in confusing or incorrect answers.

## Indexing:
Transforming unstructured data into a format that AI models can efficiently process is a key step in setting up RAG. Therefore, Indexing is a crucial step in preparing data. It involves several key steps:

1. Data Transformation: We start by getting our data ready. This involves cleaning up and extracting information from different formats like PDFs or web pages. Then, we make sure everything is in a standard text format that the model can understand.

2. Chunking: Next, we break the text into smaller pieces, kind of like cutting a big cake into slices. This helps the model handle the information better, especially since it can only process a certain amount at a time (Limited context window). Chunk size plays a role in how our system understands information. When a chunk contains multiple ideas, the semantic relationship based on the whole chunk might not precisely show the importance of the main information. This could lead to confusion or inaccurate results.
Choosing a smaller chunk gives a clearer context, focusing on specific details. On the other hand, larger chunks include more information, but some of it might not be directly related to what the user is looking for. So, finding the right balance is key to getting accurate and relevant results.

3. Encoding and Vectorization: Now comes the tricky part: turning words into numbers! We use embedding models to do this, which help our computer understand the meaning behind the words and how they're related to each other. These models, also known as encoding models or **bi-encoders**, are trained on a large corpus of data, making them powerful enough to encode chunks of documents into single vector embeddings. Vector embeddings are dense representations of objects in a continuous high-dimensional vector space, capturing semantic relationships between objects through distance and direction. Referencing back to [Dense passage retriever model](https://github.com/facebookresearch/DPR) from Facebook AI, the index encoding model utilized the document encoder BERT.

"*d(z) is a dense representation of a document produced by a BERT_BASE document encoder*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$\text{d(z)= BERT}_{d}(z)$$
==
 
4. Storage in Vector Database: Finally, we store these encoded chunks in a vector database. This specialized database is designed to manage and search embedded vectors. This makes it easy for the retriever to find what it needs quickly when we ask it questions later on. Researcher from Facebook used an FAISS index as vector database.

"*During inference time, we apply the passage encoder EP (p) to all the passages and index them using FAISS ofﬂine. FAISS is an extremely efﬁcient, open-source library for similarity search and clustering of dense vectors, which can easily be applied to billions of vectors.*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)

$$\text{E}_{P}(p)$$
==
$$\text{d(z)= BERT}_{d}(z)$$

The entire index process is described as followed:

"*Given a collection of M text passages, the goal of our dense passage retriever (DPR) is to index all the passages in a low-dimensional and continuous space, such that it can retrieve efﬁciently the top k passages relevant to the input question for the reader at run-time. Our dense passage retriever (DPR) uses a dense encoder EP (·) which maps any text passage to a d- dimensional real-valued vectors and builds an index for all the M passages that we will use for retrieval.*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)

## Index Visualization
When it comes to seeing our index in action, we rely on a handy tool called [RAGmap](https://github.com/JGalego/RAGmap). RAGmap is a simple RAG visualization tool for exploring document chunks and queries in embedding space. It enables to create a vector store and reduces high-dimensional vectors to 2D and 3D vector space. For our embedding model, we also used [BERT_BASE](https://huggingface.co/google-bert/bert-base-uncased) and indexed the original [RAG paper](http://arxiv.org/abs/2005.11401) to demonstrate how it works. Each chunk contains 256 characters and a chunk overlap of 25 characters. t-SNE algorithm is breaking down the encoded 768 dimensions of BERTd(z) and BERTq(x) into a 2-dimensional space.

Here's what the visualization in a 2D space looks like:

![BERT_RAG_Index_2D_t-SNE_Top3_chunk256 (1)](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/b896b4ee-54ba-4827-a140-f6b5f380970b)

In ordner to compare we ingested the same document with the same metrics using a different embedding model, [BGE-Small](https://huggingface.co/BAAI/bge-small-en-v1.5):
![RAG_Index_2D_t-SNE_Top3_chunk256_1_out](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/04c55cb0-ffdd-4278-be41-5a018f103733)



## Retrieval:
Retrieval in RAG involves fetching highly relevant context from a retriever. Here is how it works:

1. Encoding of User Query: The user query is processed and encoded into a representation that the system can work with. The retriever transforms the question into a vector using the fine-tuned query encoder BERT.

"*q(x) is a query representation produced by a query encoder, also based on BERT_BASE.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$\text{q(x)= BERT}_{q}(x)$$
==

"*Given a question q at run-time, we derive its embedding vq = EQ(q) and retrieve the top k passages with embeddings closest to vq.*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)

$$\text{E}_{Q}(q)$$
==
$$\text{q(x)= BERT}_{q}(x)$$

2. Document Retrieval: Using the encoded query, the system searches a large corpus of information to retrieve relevant documents or passages. This search, also called vector search or similarity search, finds top K document chunks within the indexed corpus by calculating similarity scores between the query vector and the document chunk vectors. In the paper the similarity between the question and the document passage is using the dot product, a Maximum Inner Product Search (MIPS) algorithm. This step is implemented within the FAISS index and prepares the documents for the generation process.

"*At run-time, DPR applies a different encoder EQ(·) that maps the input question to a d-dimensional vector, and retrieves k passages of which vectors are the closest to the question vector. We deﬁne the similarity between the question and the passage using the dot product of their vectors:*" (Karpukhin V, Oğuz B, Min S, et al (2020) Dense Passage Retrieval for Open-Domain Question Answering)

$$\text{sim}(q, p) = E_Q(q)^\top E_P(p)$$
==
"*Calculating top-k(pη(·|x)), the list of k documents z with highest prior probability pη(z|x), is a Maximum Inner Product Search (MIPS) problem, which can be approximately solved in sub-linear time.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$p_{\eta}(z|x) \propto \exp(d(z)^{\top}q(x))$$
==
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

![BERT_RAG_Retrieval_2D_t-SNE_Top3_chunk256 (1)](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/ff7a7f90-7834-4f75-839e-3385d85d7364)

Based on our query the following top-3 document chunks were retrieved:

| id | x | y | chunk |
|-----------------|-----------------|-----------------|-----------------|
| 0 (Query) | -20.651572193624503 | 2.5121936727809024 | What is Retrieval-Augmented Generation (RAG)? |
| 11 | -20,372099632067037 | 2,6042801988217015 | memory with non-parametric (i.e., retrieval-based) memories [ 20,26,48] can address some of these<br>issues because knowledge can be directly revised and expanded, and accessed knowledge can be |
| 4 | -23,154010314955553 | 3,0623941846106097 | memory have so far been only investigated for extractive downstream tasks. We explore a general-<br>purpose ﬁne-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-<br>trained parametric and non-parametric mem- |
| 70 | -20,614559198905972 | 1,5474185954633568 | documents, relying primarily on non-parametric knowledge. We also compare to “Closed-Book QA”<br>approaches [ 52], which, like RAG, generate answers, but which do not exploit retrieval, instead |

![RAG_Retrieval_2D_t-SNE_Top3_chunk256_1_out](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/c1fe9d3d-1151-4151-9c30-3b893fb1eae0)

Before we jump to the generation phase a short visualized recap about the formulas we learned so far for indexing and retrieval:
![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/b77511bb-85c4-49d5-844c-948995dd8c2a)
[Figure](https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval)

## Generation: 
In the paper for the generator BART-large , a pre-trained seq2seq transformer with 400M parameters is being used. However, for applying RAG any generator (encoder-decoder, decoder only) can be utilzed.

$$p_{\theta}(y_i|x, z, y_{1:i-1})$$

1. Integration of Context: Once the documents are encoded, they are ready to be combined with the encoded query. This expanded context is then incorporated into the prompt for generating a response.

"*To combine the input x with the retrieved content z when generating from BART, we simply concatenate them.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

"*To decode, we can plug p′θ(yi|x, y1:i−1) into a standard beam decoder.*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

$$p_{\theta}(y_i|x, z, y_{1:i-1})$$
==

The retrieved documents are injected in a prompt and given to the generator in a JSON format:

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

The authors propose two RAG model variants to decode from this set of latent documents: 

## RAG-Token Model
"*The RAG-Token model can be seen as a standard, autoregressive seq2seq generator with transition probability:*" (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)


$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i}^N \sum_{z \in \text{top-k}(p(⋅|x))} p_{\eta}(z|x) p_{\theta}(y_i | x, z, y_{1:i-1}) ]$$
==


Instead of picking one document and generating the entire answer from it, the RAG-Token model does something smarter. For each word (or token) in the answer, it looks at multiple documents to decide the best word to use next. Imagine you are writing an answer word by word. For the first word, the model considers all K documents and decides the best word based on the information from all of them.
After choosing the first word, it repeats the process for the second word, again considering the top K documents to pick the best one. For each word, the model creates a distribution (a set of possible words) from each document. It then combines these distributions to decide the final word. This process, known as marginalizing, ensures the answer is well-informed by considering all the retrieved documents. This method allows the model to draw information from different sources for each word, making the answer more comprehensive and accurate. It doesn't rely on just one document for the entire answer but uses multiple documents dynamically for each step.


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

This term represents the probability that a document 𝑧 is relevant to the input 𝑥. It tells us how likely each document is to contain the information we need.

$$p_n(z|x)$$
==
This term represents the probability of generating the token 𝑦𝑖 given the input 𝑥, the document 𝑧, and all the previous tokens 𝑦1:𝑖−1. This captures how well the document 𝑧 helps in generating the current token.

$$p_{\theta}(y_i|x, z_i, y_{1:i-1})$$
==

This process allows the RAG-Token model to effectively combine retrieval and generation on a token-by-token basis.




## RAG-Sequence Model

"*RAG employs a form of late fusion to integrate knowledge from all retrieved documents, meaning it makes individual answer predictions for document-question pairs and then aggregates the final prediction scores. Critically, using late fusion allows us to back-propagate error signals in the output to the retrieval mechanism, which can substantially improve the performance of the end-to-end system.*" (Lewis P, Riedel S, Kiela D, Piktus A [Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/))

"*We refer to this decoding procedure as “Thorough Decoding.*” (Lewis P, Perez E, Piktus A, et al (2021) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks)

Instead of picking different pieces of information for each word, the model uses the same set of K documents for generating the entire response.
These documents act as a consistent source of context throughout the generation process.The generator produces a probability distribution for the entire sequence (all the words in the answer and considering the previous words) using each of the K documents. It calculates the likelihood of different sequences by using the context provided by each of the K documents. The model combines these probabilities to determine the most likely sequence of words, effectively using the best information from all the documents.

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
This term represents the probability of generating the entire sequence ( 𝑦 ) given the input 𝑥 and the retrieved document 𝑧. It captures how well the document 𝑧 helps in generating the answer.

$$p_{\theta}(y|x, z)$$
==
Here, ( 𝑦 ) is broken down into individual tokens (words or characters). For each token ( 𝑦𝑖 ) in the sequence, we calculate its probability given the input 𝑥, the document 𝑧, and all the previous tokens 
𝑦1:𝑖−1. We multiply these probabilities together to get the overall probability for the sequence. Each of these probabilities is weigthed by the parameters of the BART model (p θ) as well as the similarity that is determined the prior probability on that document which was retrieved.

$$\prod_{i=1}^{N} p_{\theta}(y_i | x, z, y_{1:i-1})$$


Sources:

[Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)

[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)


# Advanced RAG

Referencing back to our initial example of an student however, just like the student faces the challenge of finding the right answers amidst the sea of information in the textbook, Naive RAG may struggle to discern which information is relevant.

# Modular RAG

Recent studies shows the shift towards visual RAG approaches.
![image](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/14043b0d-80a7-4d27-9a65-db688f362972)



# Applications of RAG:
- Chatbot for Customer support
- Documents Q&A
- Internal knowledge Q&A
- Agents
- Multimodal Applications
- Query analysis
- Structured Data Extraction
- 


# Resources and Further Reading:

## Transformer:
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Visual introduction to Transformers](https://www.youtube.com/watch?v=ISPId9Lhc1g)

[Visual introduction to Attention mechanism](https://www.youtube.com/watch?v=eMlx5fFNoYc)

## Large Language Models:

[A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)

[Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)

[LLM Visualization](https://bbycroft.net/llm)

[Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)

## RAG:

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)

[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)

[Retrieval Augmentation Reduces Hallucination in Conversation](https://arxiv.org/abs/2104.07567)

[A Survey on Retrieval-Augmented Text Generation](https://arxiv.org/abs/2202.01110)

[Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

[Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473)

[RAG for LLMs](https://www.promptingguide.ai/research/rag)

# Advanced RAG

| Pre-Retrieval Process | Post-Retrieval Process | Generation Process |
|-----------------|-----------------|-----------------|
|  |[RAFT: Adapting Language Model to Domain Specific RAG](http://arxiv.org/abs/2403.10131)|  |
|  |[Corrective Retrieval Augmented Generation](http://arxiv.org/abs/2401.15884) |  |




# RAG Orchestration Frameworks & Tools

| Frameworks | Tools | Evaluation | Visualization |
|-----------------|-----------------|-----------------|-----------------|
| [LangChain](https://www.langchain.com/) | [FlowiseAI](https://flowiseai.com/) | [RAGAS](https://github.com/explodinggradients/ragas) | [RAGexplorer](https://github.com/gabrielchua/RAGxplorer/) | 
| [LlamaIndex](https://www.llamaindex.ai/) | [LangFlow](https://www.langflow.org/) | [Langsmith](https://www.langchain.com/langsmith) |[RAGmap](https://github.com/JGalego/RAGmap) |
| [Haystack](https://haystack.deepset.ai/) | [Verba](https://github.com/weaviate/Verba) | [Langfuse](https://langfuse.com/) | [ChunkVisualizer](https://huggingface.co/spaces/Nymbo/chunk_visualizer) |
| [Canopy](https://github.com/pinecone-io/canopy) | [Cohere Coral](https://cohere.com/coral)  | [RAG-Arena](https://github.com/mendableai/rag-arena) | [ChunkViz](https://chunkviz.up.railway.app/) |
| [RAGFlow](https://github.com/infiniflow/ragflow?tab=readme-ov-file) | [Meltano](https://meltano.com/)  | [Auto-RAG](https://github.com/Marker-Inc-Korea/AutoRAG) | |
| [DSPy](https://github.com/stanfordnlp/dspy) | [Create-TSI](https://github.com/telekom/create-tsi)  | [RAGTune](https://github.com/misbahsy/RAGTune) | |
| [Cognita](https://github.com/truefoundry/cognita) |[Verta](https://www.verta.ai/rag)  |[Lunary](https://lunary.ai/) | 
|  | [Quivr](https://docs.quivr.app/intro) | [DeepEval](https://github.com/confident-ai/deepeval) |  |
|  |  |  |  |

## LLM Stack

[LetsBuildAI](https://letsbuild.ai/)
