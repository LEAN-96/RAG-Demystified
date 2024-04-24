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

# Limitations of Large Language Models (LLMs):
Despite their impressive abilities, LLMs have notable limitations, especially when applied in real-world situations. One major issue is that they sometimes generate information that is incorrect or entirely made up, which is called "hallucination." This problem becomes worse when combined with issues like bias, user privacy concerns, and security risks.

Another important limitation is that LLMs have a fixed amount of knowledge (Parametric memory). They only know what they learned during their training and can't adapt to new information. This makes them less effective in tasks that require the latest and most detailed knowledge, especially in specialized areas. In practical terms, this might mean they produce irrelevant or even harmful content, which can be expensive to correct.

Additionally, LLMs have technical restrictions, such as limits on the amount of text they can process at once (Token limits). This can affect their ability to handle large blocks of text and may make it harder to scale their use for bigger projects. 

# Introduction to RAG:

![RAG_Origin_Terms_Heatmap](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/83d8af41-cc3d-47da-87da-24cda82c1c44)



To address these limitations, the field of AI has introduced a general-purpose fine-tuning recipe called Retrieval-Augmented Generation (RAG). RAG was first introduced by [Lewis et. al](http://arxiv.org/abs/2005.11401) in 2020. The authors introduced RAG as a novel approach to natural language processing (NLP) tasks that require access to external knowledge sources (Non-parametric knowledge). RAG builds upon the advancements in large language models (LLMs) like GPT (Generative Pre-trained Transformer) models, integrating retrieval-based methods to enhance the generation process and to overcome the fixed amount of knowledge (Parametric knowledge).

Let's explore RAG through the analogy of a student attending an exam:
Think of a pre-trained Large Language Model (LLM) as a closed-book exam where the student relies solely on their memorized knowledge without referring to any materials. They're expected to answer questions based on what they've learned beforehand.

Now, picture RAG as an open-book exam for the same student, but with a twist: they haven't studied! In this scenario, the student can access a textbook during the exam, similar to how RAG integrates an external knowledge base with the language model. 

In essence, RAG is like an open-book exam without studying, where the student (or the model) can access additional resources but must still navigate through them to find the correct information, while pre-trained LLMs are more like closed-book exams where the model can only use what it already knows.

![RAG_Analogy](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/92b40321-be9d-44c0-9c25-96900ad67ef4)


Understanding RAG begins with grasping the main ideas of retrieval-based and generation-based approaches in NLP. RAG works similarly to a typical sequence-to-sequence (seq2seq) model, where it takes one sequence as input and produces a corresponding sequence as output. generation-based in an traditional seq2seq model  focus on creating text solely based on the input without looking at external sources. However, what sets RAG apart is that it adds an extra step. Instead of directly sending the input to the generator, RAG uses retrieval-based methods, which involve finding useful information from outside sources, like databases or existing texts, to help with generating text. RAG combines these two methods by using a retriever component (dense passage retriever) to find relevant context from external sources and a generator component (seq2seq model) to produce text based on both the input and retrieved context.

By doing this, RAG improves the quality of the text generated by the model and ensures that it's accurate and up-to-date. This approach also helps to reduce the problem of generating incorrect information, known as "hallucinations," by making the model rely on existing documentation.

Additionally, RAG enhances transparency and helps with error checking and copyright issues by clearly citing the sources of information used in the generated text. It also allows for private or specialized data to be incorporated into the text, ensuring that the output is tailored to specific needs or is more current.

One advantage is its ability to reduce the need for frequent retraining (fine-tuning) of the model. Unlike traditional approaches, where the entire model must be retrained with new documents to change what it knows, RAG simplifies this process. By updating the external database with new information, RAG eliminates the need for the model to memorize everything, saving time and computational resources. This flexibility allows us to control what the model knows simply by swapping out the documents used for knowledge retrieval.
[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)
[Retrieval Augmented Generation: Streamlining the creation of intelligent natural language processing models](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)
# Naive RAG

**High-Level RAG Architecture:**
![RAG-Architecture2](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/b944dc68-dfa0-4801-b65e-41f4d105d2ad)
The entire RAG-Process is often referred as "RAG-Pipeline".

Indexing:

Retrieval:

Generation: 
RAG uses a method called late fusion to combine information from all the documents it finds. It first predicts answers for each pair of document and question. Late fusion means that it combines all these predictions to come up with a final answer. This method is beneficial because it helps improve the overall performance of the system by allowing it to learn from its mistakes and correct them.

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

| Frameworks | Tools | Evaluation |
|-----------------|-----------------|-----------------|
| [LangChain](https://www.langchain.com/) | [FlowiseAI](https://flowiseai.com/) | [RAGAS](https://github.com/explodinggradients/ragas) |
| [LlamaIndex](https://www.llamaindex.ai/) | [LangFlow](https://www.langflow.org/) | [Langsmith](https://www.langchain.com/langsmith) |
| [Haystack](https://haystack.deepset.ai/) | [Verba](https://github.com/weaviate/Verba) | [Langfuse](https://langfuse.com/) | 
| [Canopy](https://github.com/pinecone-io/canopy) | [Cohere Coral](https://cohere.com/coral)  | [RAG-Arena](https://github.com/mendableai/rag-arena) |
| [RAGFlow](https://github.com/infiniflow/ragflow?tab=readme-ov-file) | [Meltano](https://meltano.com/)  | [Auto-RAG](https://github.com/Marker-Inc-Korea/AutoRAG) |
| [DSPy](https://github.com/stanfordnlp/dspy) |  | [RAGTune](https://github.com/misbahsy/RAGTune) |
| [Cognita](https://github.com/truefoundry/cognita) |[Create-TSI](https://github.com/telekom/create-tsi)  | [ChunkVisualizer](https://huggingface.co/spaces/Nymbo/chunk_visualizer)|
|  |[Verta](https://www.verta.ai/rag)  | [ChunkViz](https://chunkviz.up.railway.app/) |
|  |  |  |
|  |  |  |

## LLM Stack

[LetsBuildAI](https://letsbuild.ai/)
