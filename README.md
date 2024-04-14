# RAG-Demystified

Are you intrigued by the concept of **Retrieval-Augmented-Generation (RAG)** but find it intimidating due to its association with AI and machine learning? Look no further! RAG-Demystified is your gateway to understanding this groundbreaking technology without the need for prior expertise in AI.

In this repository, our mission is simple: to break down RAG into easily digestible explanations that anyone can understand. Whether you're a student exploring the realms of artificial intelligence, a professional seeking to broaden your knowledge, or simply curious about the latest advancements in technology, RAG-Demystified is designed with you in mind.

But we don't stop there. Recognizing that the AI community often assumes familiarity with concepts like **Large Language Models (LLMs)** and the transformer architecture, we go the extra mile to ensure that even the basics are accessible to everyone.

Join us on a journey of discovery as we unravel the mysteries of RAG and provide you with the tools to navigate this exciting field with confidence. By the end of your exploration with RAG-Demystified, you'll have a solid understanding of not only RAG itself but also the foundational concepts that underpin it. Welcome to RAG-Demystified, where complexity meets clarity.



# Fundamentals of Large Language Models (LLMs):


# Limitations of Large Language Models (LLMs):
Despite their impressive abilities, LLMs have notable limitations, especially when applied in real-world situations. One major issue is that they sometimes generate information that is incorrect or entirely made up, which is called "hallucination." This problem becomes worse when combined with issues like bias, user privacy concerns, and security risks.

Another important limitation is that LLMs have a fixed amount of knowledge (Parametric memory). They only know what they learned during their training and can't adapt to new information. This makes them less effective in tasks that require the latest and most detailed knowledge, especially in specialized areas. In practical terms, this might mean they produce irrelevant or even harmful content, which can be expensive to correct.

Additionally, LLMs have technical restrictions, such as limits on the amount of text they can process at once (Token limits). This can affect their ability to handle large blocks of text and may make it harder to scale their use for bigger projects. 

# Introduction to RAG:

![RAG_Origin_Terms_Heatmap](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/83d8af41-cc3d-47da-87da-24cda82c1c44)



To address these limitations, the field of AI has introduced a new approach called Retrieval-Augmented Generation (RAG). RAG was first introduced by [Lewis et. al](http://arxiv.org/abs/2005.11401) in 2020. The authors introduced RAG as a novel approach to natural language processing (NLP) tasks that require access to external knowledge sources (Non-parametric knowledge). RAG builds upon the advancements in large language models (LLMs) like GPT (Generative Pre-trained Transformer) models, integrating retrieval-based methods to enhance the generation process and to overcome the fixed amount of knowledge (Parametric knowledge).

Let's explore RAG through the analogy of a student attending an exam:
Think of a pre-trained Large Language Model (LLM) as a closed-book exam where the student relies solely on their memorized knowledge without referring to any materials. They're expected to answer questions based on what they've learned beforehand.

Now, picture RAG as an open-book exam for the same student, but with a twist: they haven't studied! In this scenario, the student can access a textbook during the exam, similar to how RAG integrates an external knowledge base with the language model. 

In essence, RAG is like an open-book exam without studying, where the student (or the model) can access additional resources but must still navigate through them to find the correct information, while pre-trained LLMs are more like closed-book exams where the model can only use what it already knows.

![RAG_Analogy](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/92b40321-be9d-44c0-9c25-96900ad67ef4)


For beginners, understanding RAG starts with grasping the core concepts of retrieval-based and generation-based approaches in NLP. Retrieval-based methods involve retrieving relevant information from external knowledge sources, such as databases or pre-existing texts, to inform the generation process. On the other hand, generation-based methods focus on generating text based solely on the input prompt without explicit access to external knowledge. 

RAG combines these two approaches by incorporating a retriever component (dense passage retriever) that retrieves relevant context from external knowledge sources and a generator component (seq2seq model) that generates text conditioned on both the input prompt and the retrieved context. By doing this, RAG improves the quality of the text generated by the model and ensures that it's accurate and up-to-date. This approach also helps to reduce the problem of generating incorrect information, known as "hallucinations," by making the model rely on existing documentation.

Additionally, RAG enhances transparency and helps with error checking and copyright issues by clearly citing the sources of information used in the generated text. It also allows for private or specialized data to be incorporated into the text, ensuring that the output is tailored to specific needs or is more current.

One of the significant advantages of RAG is that it reduces the need for frequent retraining (fine-tuning) of the model. This is because the external database is continuously updated with new information, eliminating the reliance on the model to memorize everything it needs to know. This saves time and computational resources, making RAG a more efficient and reliable approach for generating text.

# Naive RAG

**High-Level RAG Architecture:**
![RAG-Architecture2](https://github.com/LEAN-96/RAG-Demystified/assets/150592634/b944dc68-dfa0-4801-b65e-41f4d105d2ad)



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
## RAG:

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](http://arxiv.org/abs/2005.11401)

[Retrieval Augmentation Reduces Hallucination in Conversation](https://arxiv.org/abs/2104.07567)

[A Survey on Retrieval-Augmented Text Generation](https://arxiv.org/abs/2202.01110)

[Retrieval-Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2312.10997)

[Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473)

[RAG for LLMs](https://www.promptingguide.ai/research/rag)

# RAG Orchestration Frameworks & Tools

| Frameworks | Tools | Evaluation |
|-----------------|-----------------|-----------------|
| [LangChain](https://www.langchain.com/) | [FlowiseAI](https://flowiseai.com/) | [RAGAS](https://github.com/explodinggradients/ragas) |
| [LlamaIndex](https://www.llamaindex.ai/) | [LangFlow](https://www.langflow.org/) | [Langsmith](https://www.langchain.com/langsmith) |
| [Haystack](https://haystack.deepset.ai/) | [Verba](https://github.com/weaviate/Verba) | [Langfuse](https://langfuse.com/) | 
| [Canopy](https://github.com/pinecone-io/canopy) |  | [RAG-Arena](https://github.com/mendableai/rag-arena) |
| [RAGFlow](https://github.com/infiniflow/ragflow?tab=readme-ov-file) |  | [Auto-RAG](https://github.com/Marker-Inc-Korea/AutoRAG) |
|  |  | [RAGTune](https://github.com/misbahsy/RAGTune) |

## LLM Stack

[LetsBuildAI](https://letsbuild.ai/)
