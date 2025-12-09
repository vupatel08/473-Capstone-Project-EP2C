# G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering

Xiaoxin He1 Yijun Tian2 Yifei Sun1 Nitesh V. Chawla2 Thomas Laurent3 Yann LeCun4,5 Xavier Bresson1 Bryan Hooi1

{xiaoxin, yifeisun, xaviercs, bhooi}@comp.nus.edu.sg {yijun.tian, nchawla}@nd.edu, tlaurent@lmu.edu, yann@cs.nyu.edu 1National University of Singapore 2University of Notre Dame 3Loyola Marymount University 4New York University 5Meta AI

# Abstract

Given a graph with textual attributes, we enable users to ‘chat with their graph’: that is, to ask questions about the graph using a conversational interface. In response to a user’s questions, our method provides textual replies and highlights the relevant parts of the graph. While existing works integrate large language models (LLMs) and graph neural networks (GNNs) in various ways, they mostly focus on either conventional graph tasks (such as node, edge, and graph classification), or on answering simple graph queries on small or synthetic graphs. In contrast, we develop a flexible question-answering framework targeting real-world textual graphs, applicable to multiple applications including scene graph understanding, common sense reasoning, and knowledge graph reasoning. Toward this goal, we first develop a Graph Question Answering (GraphQA) benchmark with data collected from different tasks. Then, we propose our $G$ -Retriever method, introducing the first retrievalaugmented generation (RAG) approach for general textual graphs, which can be fine-tuned to enhance graph understanding via soft prompting. To resist hallucination and to allow for textual graphs that greatly exceed the LLM’s context window size, $G$ -Retriever performs RAG over a graph by formulating this task as a PrizeCollecting Steiner Tree optimization problem. Empirical evaluations show that our method outperforms baselines on textual graph tasks from multiple domains, scales well with larger graph sizes, and mitigates hallucination. Our codes and datasets are available at: https://github.com/XiaoxinHe/G-Retriever.

# 1 Introduction

Graphs and Large Language Models (LLMs). The advent of LLMs has significantly shaped the artificial intelligence landscape. As these models are applied to increasingly diverse tasks, their ability to process complex structured data will be increasingly vital. In particular, in our interconnected world, a significant portion of real-world data inherently possesses a graph structure, such as the Web, e-commerce, recommendation systems, knowledge graphs, and many others. Moreover, many of these involve graphs with textual attributes (i.e., textual graphs), making them well-suited for LLM-centric methods. This has spurred interest in combining graph-based technologies, particularly graph neural networks (GNNs), with LLMs to enhance their reasoning on graphs [44, 15, 24].

The Present Work: Enabling ‘Chat With Your Graph’. While existing works integrate LLMs and GNNs in various ways, they mostly focus on conventional graph tasks such as node, edge and graph classification [8], or answering simple questions on small or synthetic graphs [44, 31]. In contrast, we develop a flexible question-answering framework targeting complex and real-world graphs. This framework enables users to ‘chat with their graph’ via a unified conversational interface, representing a leap towards intuitive interaction with graph data, as demonstrated in Figure 1.

![](images/b15cdcff1ed6ff7624836431f8ed05a35356b94c90ff7ac56822691329bd530c.jpg)  
Figure 1: We develop a flexible question-answering framework targeting real-world textual graph applications via a unified conversational interface. Presented here are examples showcasing the model’s adeptness in handling generative and creative queries in practical graph-related tasks: common sense reasoning, scene understanding, and knowledge graph reasoning, respectively.

The Need for a Comprehensive GraphQA Benchmark. Question answering (QA) is a fundamentally important task in natural language processing, serving as a key benchmark for assessing LLMs and providing a unified interface for various capabilities. Despite extensive research in QA, a comprehensive benchmark specifically tailored for the graph modality is lacking. In contrast to existing benchmarks that focus on basic graph-based reasoning tasks such as node degree, edge existence, and shortest path [6, 44], our benchmark addresses complex and real-world graph applications including common sense reasoning, scene understanding, and knowledge graph reasoning (refer to Figure 2). This is vital for measuring progress toward a model capable of answering a wide range of questions about graphs from diverse applications.

New Architecture for GraphQA. To enable effective and efficient graph QA, even on large graphs, we propose $G$ -Retriever, a new framework combining the strengths of GNNs, LLMs, and RAG (Figure 3). Next, we will discuss the motivation, strengths, and details of our model.

Tackling Hallucination in Graph LLMs. LLMs are prone to hallucination, a phenomenon where the generated content is factually inaccurate or nonsensical [12]. We validate the presence of this issue in graph settings. In particular, we employ a baseline method that adapts MiniGPT-4 [57] to graphs, where a frozen LLM interacts with a trainable GNN that encodes graph data as a soft prompt, as in GraphToken [31]. Our findings, shown in Table 1, indicate that hallucination, an important problem in text-based LLMs, is also prevalent in Graph LLMs. This may be attributed to the baseline’s inability to recall the entire graph structure from a single graph embedding, leading to the generation of incorrect nodes or edges during the QA task. In contrast, by employing RAG for direct information retrieval from the actual graph, our $G$ -Retriever mitigates this issue, as substantiated by Table 1.

Enhancing Scalability and Efficiency in Graph LLMs. Recent research endeavors have explored translating graphs into natural language, such as by flattening nodes and edges into a text sequence, enabling their processing by LLMs for graph-based tasks [56, 6]. However, this method faces critical scalability issues. Converting a graph with thousands of nodes and edges into a text sequence results in an excessive number of tokens, surpassing the input capacity of many LLMs. An alternative of truncating the graph text sequence to fit the LLM’s input token limit leads to loss of information and response quality. $G$ -Retriever overcomes these issues with its RAG component, which allows for effective scaling to larger graphs by selectively retrieving only relevant parts of the graph.

Tailoring the RAG Approach to Graphs. Existing RAG methodologies are primarily designed for simpler data types or knowledge graphs, where information is retrieved in a manner isolated from the graph structure [7, 1, 36, 16]. Hence, we introduce a new retrieval approach for general textual graphs. Notably, we formulate subgraph retrieval as a Prize-Collecting Steiner Tree (PCST)

Table 1: Observation and mitigation of hallucination in graph LLMs.   

<table><tr><td>User</td><td>elephants elephant —in eating bushes standing by grass on the side of→river</td></tr><tr><td rowspan="6">LLM w/ Graph Prompt Tuning</td><td>[Incorrect response with hallucinated nodes and edges] The animal in the bushes is a deer.</td></tr><tr><td>Nodes:</td></tr><tr><td></td></tr><tr><td>*Deer (node 1) * Bushes (node 2)</td></tr><tr><td>Edges:</td></tr><tr><td>* Deer → Bushes (edge 1)</td></tr><tr><td rowspan="5"></td><td>[Correct response with accurate node and edge references]</td></tr><tr><td>Based on the information provided in the graph, the animal in the bushes is an elephant.</td></tr><tr><td>The nodes that support this answer are:</td></tr><tr><td>* Node 1: The elephant is mentioned in the first line of the graph.</td></tr><tr><td>The edges that support this answer are: * The edge connects the elephant to the bushes, indicating that the elephant is in the bushes.</td></tr></table>

optimization problem, which takes the neighborhood information into account during retrieval. This also allows the return of a subgraph most relevant to a query, thereby improving explainability.

The contributions of this paper are outlined as follows:

• Pioneering the integration of Graph RAG. We present the first retrieval approach for general textual graph tasks, which greatly enhances scalability and efficiency. • Enabling ‘Chat with Your Graph’. We develop a flexible question answering framework to handle complex and real-world textual graphs through a unified conversational interface. • Introducing A Novel GraphQA Benchmark. We introduce a diverse benchmark targeted at real-world graph question answering, filling a crucial research gap. • Empirical Findings. We demonstrate the efficiency and effectiveness of $G$ -Retriever in multiple domains and present the significant finding of hallucination in graph LLMs.

# 2 Related Work

Graphs and Large Language Models. A significant body of research has emerged at the intersection of graph-based techniques and LLMs [30, 24, 15, 44, 54]. This exploration spans diverse aspects, ranging from the design of general graph models [47, 25, 51, 19, 40, 31], and multi-modal architectures [23, 49] to practical applications. Noteworthy applications include fundamental graph reasoning [52, 3, 56], node classification [8, 11, 39, 5, 50, 4, 33], graph classification/regression [32, 55], and leveraging LLMs for knowledge graph-related tasks [41, 14, 29].

Retrieval-Augmented Generation (RAG). The concept of Retrieval-Augmented Generation, initially proposed by Lewis et al. [21], has gained increased attention for its ability to mitigate the issue of hallucination within LLMs and enhance trustworthiness and explainability [7]. Despite its success in language-related tasks, the application of retrieval-based approaches to general graph tasks remains largely unexplored. Most existing work focuses primarily on the knowledge graph [38, 1, 36, 16]. Our research is the first to apply a retrieval-based approach to general graph tasks, marking a novel advancement in the field and demonstrating the versatility of RAG beyond language processing.

Parameter-Efficient Fine-Tuning (PEFT). The field of LLMs has witnessed significant advancements through various parameter-efficient fine-tuning techniques. These methodologies have played a crucial role in refining LLMs, boosting their performance while minimizing the need for extensive parameter training. Notable among these techniques are prompt tuning, as introduced by Lester et al. [20], and prefix tuning, proposed by Li and Liang [22]. Furthermore, methods like LoRA [10], and the LLaMA-adapter [53], have been influential. These advancements in PEFT have laid the foundation for the development of sophisticated multimodal models. Prominent examples in this domain include MiniGPT-4 [57], LLaVA [26], and NExT-Chat [46]. There are also emerging efforts in applying PEFT to graph LLMs, such as GraphLLM [3] and GraphToken [31] for basic graph reasoing tasks and GNP [41] for multi-option QA on knowledge graphs.

# 3 Formalization

This section establishes the notation and formalizes key concepts related to textual graphs, language models for text encoding, and large language models and prompt tuning.

Textual Graphs. A textual graph is a graph where nodes and edges possess textual attributes. Formally, it can be defined as $G ^ { - } = ( V , \bar { E _ { \ l } } , \{ \bar { x } _ { n } \} _ { n \in V } , \{ x _ { e } \} _ { e \in E } )$ , where $V$ and $E$ represent the sets of nodes and edges, respectively. Additionally, $x _ { n } \in D ^ { L _ { n } }$ and $x _ { e } \in D ^ { L _ { e } }$ denote sequential text associate with a node $n \in V$ or an edge $e \in E$ , where $D$ represents the vocabulary, and $L _ { n }$ and $L _ { e }$ signify the length of the text associated with the respective node or edge.

Language Models for Text Encoding. In the context of textual graphs, language models (LMs) are essential for encoding the text attributes associated with nodes and edges, thereby learning representations that capture their semantic meaning. For a node $n$ with text attributes $x _ { n } \in D ^ { L _ { n } }$ , an LM encodes these attributes as:

$$
z _ { n } = \mathbf { L M } ( x _ { n } ) \in \mathbb { R } ^ { d } ,
$$

where $z _ { n }$ is the output of the LM, and $d$ is the dimension of the output vector.

Large Language Models and Prompt Tuning. LLMs have introduced a new paradigm for taskadaptation known as “pre-train, prompt, and predict”, replacing the traditional “pre-train, fine-tune” paradigm. In this paradigm, the LLM is first pre-trained on a large corpus of text data to learn general language representations. Then, rather than fine-tuning the model on task-specific labeled data, the model is prompted with a textual prompt that specifies the task and context. Subsequently, the model generates the output directly based on the prompt and the input.

The LLM, parameterized by weights $\theta$ , takes a sequence of tokens $X$ , and a prompt $P$ as input, and generates a sequence of tokens $Y = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { r } \}$ as output. Formally, the probability distribution of the output sequence given the concatenated input sequence and prompt, i.e., $[ P ; X ]$ , is defined as

$$
p _ { \theta } ( Y | [ P ; X ] ) = \prod _ { i = 1 } ^ { r } p _ { \theta } ( y _ { i } | y _ { < i } , [ P ; X ] ) .
$$

Here, $y _ { < i }$ represents the prefix of sequence $y$ up to position $i - 1$ , and $p ( y _ { i } | y _ { < i } , [ P ; X ] )$ represents the probability of generating token $y _ { i }$ given $y _ { < i }$ and $[ P ; X ]$ .

Soft prompt tuning eliminates the need for manual prompt design. Given a series of $p$ tokens $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { p } \}$ , after being processed by the text embedder, it forms a matrix $X _ { e } \in \mathbb { R } ^ { p \times d _ { l } }$ , where $d _ { l }$ is the dimension of the embedding space. Soft prompts can be represented as parameters $P _ { e } \in \mathbb { R } ^ { q \times d _ { l } }$ , where $q$ is the length of the prompt. The prompt is then concatenated with the embedded input, forming a single matrix $[ P _ { e } ; X _ { e } ] ^ { - } \in \bar { \mathbb { R } ^ { ( q + p ) \times \bar { d } _ { l } } }$ . This combined matrix is processed by the self-attention layers in LLM as usual. Training involves maximizing the likelihood of $Y$ through backpropagation, with gradient updates applied solely to $P _ { e }$ , while $\theta$ remains fixed.

# 4 Proposed GraphQA Benchmark

Our GraphQA represents a comprehensive and diverse benchmark for graph question-answering. It is tailored to assess the capabilities of models in answering a wide range of questions about graphs across diverse domains.

# 4.1 Data Format

Each entry in the GraphQA benchmark consists of a textual graph, a question related to the graph, and one or more corresponding answers, as illustrated in Figure 2.

Table 2: Summary of datasets used in GraphQA benchmark.   

<table><tr><td>Dataset</td><td>ExplaGraphs</td><td>SceneGraphs</td><td>WebQSP</td></tr><tr><td>#Graphs</td><td>2,766</td><td>100,000</td><td>4,737</td></tr><tr><td>Avg. #Nodes</td><td>5.17</td><td>19.13</td><td>1370.89</td></tr><tr><td>Avg. #Edges</td><td>4.25</td><td>68.44</td><td>4252.37</td></tr><tr><td>Node Attribute</td><td>Commonsense concepts</td><td>Object attributes (e.g., color, shape)</td><td>Entities in Freebase</td></tr><tr><td>Edge Attribute</td><td>Commonsense relations</td><td>Relations (e.g., actions, spatial relations)</td><td>Relations in Freebase</td></tr><tr><td>Task Evaluation Matrix</td><td>Common sense reasoning Accuracy</td><td>Scene graph question answering Accuracy</td><td>Knowledge based question answering Hit@1</td></tr></table>

![](images/90636f082a6636e2286e4f98d4c942081839788d999337edf93014016c98c8eb.jpg)  
Figure 2: Illustrative examples from the GraphQA benchmark datasets.

Textual Graphs. The textual graph is converted into a natural language format, resulting in a list of nodes and edges, akin to a CSV file format. It is important to note that while multiple methods exist for textualizing a graph, our focus is not on identifying the optimal solution. Instead, we prioritize a straightforward yet empirically effective approach for representing graphs in natural language, facilitating the benchmark’s use in diverse GraphQA scenarios.

Questions and Answers. Questions are designed to explore specific elements or relationships within the graph. Answers, residing within the attributes of nodes or edges, often require multi-hop reasoning for accurate identification.

# 4.2 Description of Datasets

The GraphQA benchmark integrates three existing datasets: ExplaGraphs, SceneGraphs, and WebQSP. Table 2 presents the summary statistics of these datasets. It is important to note that these datasets were not originally developed for this work. However, a significant contribution of our research is the standardization and processing of these diverse datasets into a uniform data format suitable for the GraphQA benchmark. These datasets, previously utilized in different contexts, are reintroduced with a new focus tailored for GraphQA. For a detailed comparison with the original datasets, see the Appendix C.

ExplaGraphs is a dataset for generative commonsense reasoning, focusing on creating explanation graphs for stance prediction in debates. It offers detailed, unambiguous commonsense-augmented graphs to evaluate arguments supporting or refuting a belief. The primary task is to assess whether arguments are supportive or contradictory, using accuracy as the metric. We have converted the triplet-form provided in Saha et al. [35] into a standard graph format.

SceneGraphs, a visual question answering dataset, includes 100,000 scene graphs. Each graph details objects, attributes, and relations within an image. This dataset challenges users with tasks requiring spatial understanding and multi-step inference. The task is to answer open-ended questions based on a textual description of a scene graph, evaluated on accuracy. We have sampled from the GQA dataset [13] and constructed standard graphs from the provided JSON files.

WebQSP is a large-scale multi-hop knowledge graph QA dataset consisting of 4,737 questions. It was proposed by Yih et al. [48] and, following Luo et al. [28], utilizes a subset of Freebase, encompassing facts within 2-hops of entities mentioned in the questions. The task involves answering questions that require multi-hop reasoning. Given the possibility of multiple answers for the same question, the hit $@ 1$ metric is used to assess the precision of the top returned answer.

![](images/d0b9e905c1e15c8160bbf1c711276de880a63a12e731b7997e3aaabf25da8f22.jpg)  
Figure 3: Overview of the proposed $G$ -Retriever: 1) Indexing: Graphs are indexed for efficient query processing; 2) Retrieval: The most semantically relevant nodes and edges are retrieved, conditioned on the query; 3) Subgraph Construction: A connected subgraph is extracted, covering as many relevant nodes and edges as possible while maintaining a manageable graph size; 4) Generation: An answer is generated using a ‘graph prompt’, a textualized graph, and the query.

# 5 G-Retriever

In this section, we introduce $G$ -Retriever, a new architecture tailored for GraphQA, which integrates the strengths of GNNs, LLMs, and RAG. To allow efficient fine-tuning while preserving the LLM’s pretrained language capabilities, we freeze the LLM and use a soft prompting approach on the output of the GNN. Our RAG-based design mitigates hallucinations through direct retrieval of the graph, while allowing our approach to scale to graphs exceeding the LLM’s context window size. To adapt RAG to graphs, we formulate subgraph retrieval as a PCST optimization problem. This approach also allows us to enhance explainability by returning the retrieved subgraph.

$G$ -Retriever comprises four main steps: indexing, retrieval, subgraph construction and generation, as depicted in Figure 3. The implementation details of each step are elaborated in the following sections.

# 5.1 Indexing

We initiate the RAG approach by generating node and graph embeddings using a pre-trained LM.   
These embeddings are then stored in a nearest neighbor data structure.

To elaborate, consider $x _ { n } \in D ^ { L _ { n } }$ as the text attributes of node $n$ . Utilizing a pre-trained LM, such as SentenceBert [34], we apply the LM to $x _ { n }$ , yielding the representation $z _ { n }$ :

$$
z _ { n } = \mathbf { L M } ( x _ { n } ) \in \mathbb { R } ^ { d } ,
$$

where $d$ denotes the dimension of the output vector. Similar preprocessing steps are applied to edges.   
Refer to Figure 3, Step 1 for an illustrative representation.

# 5.2 Retrieval

For retrieval, we employ the same encoding strategy to the query $x _ { q }$ , to ensure consistent treatment of textual information:

$$
z _ { q } = \mathbf { L M } ( x _ { q } ) \in \mathbb { R } ^ { d } .
$$

Next, to identify the most relevant nodes and edges for the current query, we use a $\mathbf { k }$ -nearest neighbors retrieval approach. This method yields a set of ‘relevant nodes/edges’ based on the similarity between the query and each node or edge. The retrieval operation is defined as:

$$
\begin{array} { r } { V _ { k } = \operatorname { a r g t o p k } _ { n \in V } \cos ( z _ { q } , z _ { n } ) } \\ { E _ { k } = \operatorname { a r g t o p k } _ { e \in E } \cos ( z _ { q } , z _ { e } ) , } \end{array}
$$

where $z _ { n }$ and $z _ { e }$ are the embeddings of node $n$ and edge $e$ , respectively. We use the cosine similarity function, $\cos ( \cdot , \cdot )$ , to measure the similarity between the query representation and the node/edge embeddings. The argtopk operation retrieves the top- $\mathbf { \nabla } \cdot \mathbf { k }$ elements based on this similarity, providing a set of nodes $V _ { k }$ and edges $E _ { k }$ considered most relevant to the query. See Step 2 of Figure 3.

# 5.3 Subgraph Construction

This step aims to construct a subgraph that encompasses as many relevant nodes and edges as possible, while keeping the graph size manageable. This approach offers two key benefits: Firstly, it helps to filter out nodes and edges that are not pertinent to the query. This is crucial because irrelevant information can overshadow the useful data, potentially diverting the focus of the subsequent LLM from the information of interest. Secondly, it enhances efficiency; by keeping the graph size manageable, it becomes feasible to translate the graph into natural language and then input it into the LLM for processing. The Prize-Collecting Steiner Tree algorithm [2] serves as our primary method for identifying such optimally sized and relevant subgraphs. See Step 3 in Figure 3.

Prize-Collecting Steiner Tree (PCST). The PCST problem aims to find a connected subgraph that maximizes the total prize values of its nodes while minimizing the total costs of its edges. Our approach assigns higher prize values to nodes and edges more relevant to the query, as measured by cosine similarity. Specifically, the top $k$ nodes/edges are assigned descending prize values from $k$ down to 1, with the rest assigned zero. The node prize assignment is as follows:

$$
\operatorname { p r i z e } ( n ) = { \left\{ \begin{array} { l l } { k - i , } & { { \mathrm { i f ~ } } n \in V _ { k } { \mathrm { ~ a n d ~ } } n { \mathrm { ~ i s ~ t h e ~ t o p ~ } } i { \mathrm { ~ n o d e } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Edge prizes are assigned similarly. The objective is to identify a subgraph, $S ^ { * } = ( V ^ { * } , E ^ { * } )$ , that optimizes the total prize of nodes and edges, minus the costs associated with the size of the subgraph:

$$
S ^ { * } = \operatorname * { a r g m a x } _ { S \subseteq G , \operatorname * { c o n n e c t e d } } \sum _ { n \in V _ { S } } \operatorname { p r i z e } ( n ) + \sum _ { e \in E _ { S } } \operatorname { p r i z e } ( e ) - \operatorname { c o s t } ( S ) ,
$$

where

$$
\mathrm { c o s t } ( \boldsymbol { S } ) = \left| E _ { S } \right| \times C _ { e } ,
$$

and $C _ { e }$ denotes a predefined cost per edge, which is adjustable to control the subgraph size.

The original PCST algorithm is designed for node prizes only. However, given the significance of edge semantics in certain scenarios, we adapt the algorithm to accommodate edge prizes as follows: Consider an edge e with a cost $C _ { e }$ and a prize $P _ { e }$ . If $C _ { e } > P _ { e }$ , it can be treated as a reduced edge cost of $C _ { e } - P _ { e }$ . However, if $P _ { e } > C _ { e }$ , negative edge costs are not allowed in the original algorithm. Our solution involves replacing edge $e$ with a ‘virtual node’ $v _ { e }$ , connected to both endpoints of $e$ . This virtual node is assigned a prize of $P _ { e } - C _ { e }$ , and the cost of the two new edges leading to the virtual node is set to zero. This modification effectively mirrors the original problem, as including edge $e$ in the original graph is analogous to including the virtual node in the modified graph. Finally, we optimize the PCST problem using a near-linear time approach [9].

# 5.4 Answer Generation

Graph Encoder. Let $S ^ { * } = ( V ^ { * } , E ^ { * } )$ represent the retrieved subgraph. We use a graph encoder to model the structure of this graph, specifically using a standard Graph Attention Network (GAT) [43]. Our approach for encoding the retrieved subgraph is defined as follows:

$$
h _ { g } = \mathrm { P O O L } ( \mathbf { G N N } _ { \phi _ { 1 } } ( S ^ { * } ) ) \in \mathbb { R } ^ { d _ { g } } ,
$$

Here, POOL denotes the mean pooling operation, and $d _ { g }$ is the dimension of the graph encoder.

Projection Layer. We incorporate a multilayer perceptron (MLP) to align the graph token with the vector space of the LLM:

$$
\hat { h } _ { g } = \mathbf { M } \mathbf { L } \mathbf { P } _ { \phi _ { 2 } } ( h _ { g } ) \in \mathbb { R } ^ { d _ { l } } ,
$$

where $d _ { l }$ is the dimension of the LLM’s hidden embedding.

Text Embedder. To leverage the text-reasoning capabilities of LLMs, we transform the retrieved subgraph $S ^ { * }$ into a textual format. This transformation involves flattening the textual attributes of the nodes and edges, as illustrated in the green box in Figure 2. We refer to this operation as textualize $( \cdot )$ . Subsequently, we combine the textualized graph with the query to generate a response. Let $x _ { q }$ denote the query; we concatenate it with the textualized graph textualize $( S ^ { * } )$ . We then map the result to an embedding $h _ { t }$ using a text embedder, which is the first layer of a pretrained and frozen LLM:

$$
h _ { t } = \mathrm { T e x t E m b e d d e r } ( [ \mathrm { t e x t u a l i z e } ( S ^ { * } ) ; x _ { q } ] ) \in \mathbb { R } ^ { L \times d _ { l } } ,
$$

where $[ ; ]$ represents the concatenation operation, and $L$ is the number of tokens.

LLM Generation with Graph Prompt Tuning. The final stage involves generating the answer $Y$ given the graph token $\hat { h } _ { g }$ , acting as a soft prompt, and the text embedder output $h _ { t }$ . These inputs are fed through the self-attention layers of a pretrained frozen LLM, with parameter $\theta$ . The generation process is represented as follows:

$$
p _ { \theta , \phi _ { 1 } , \phi _ { 2 } } ( Y | S ^ { * } , x _ { q } ) = \prod _ { i = 1 } ^ { r } p _ { \theta , \phi _ { 1 } , \phi _ { 2 } } ( y _ { i } | y _ { < i } , [ \hat { h } _ { g } ; h _ { t } ] ) ,
$$

where $[ \hat { h } _ { g } ; h _ { t } ]$ concatenates the graph token $\hat { h } _ { g }$ and the text embedder output $h _ { t }$ . While $\theta$ is frozen, the graph token $\hat { h } _ { g }$ receives gradients, enabling the optimization of the parameters of the graph encoder $\phi _ { 1 }$ and the projection layer $\phi _ { 2 }$ through standard backpropagation.

# 6 Experiments

# 6.1 Experiment Setup

In the indexing step, we use SentenceBert [34] as the LM to encode all node and edge attributes. In the generation step, we use the open-source Llama2-7b [42] as the LLM and Graph Transformer [37] as the graph encoder. Additional details are provided in Appendix B.1.

# 6.2 Main Results

In our experiments, we consider three model configurations: 1) Inference-only: Using a frozen LLM for direct question answering; 2) Frozen LLM w/ prompt tuning (PT): Keeping the parameters of the LLM frozen and adapting only the prompt; 3) Tuned LLM: Fine-tuning the LLM with LoRA [10]. We provide more details in Appendix B.2.

Table 3: Performance comparison across ExplaGraphs, SceneGraphs, and WebQSP datasets for different configurations, including Inference-only, Frozen LLM with prompt tuning (PT), and Tuned LLM settings. Mean scores and standard deviations (mean $\pm$ std) are presented. The first best result for each task is highlighted in bold and the second best result is highlighted with an underline.   

<table><tr><td>Setting</td><td>Method</td><td>ExplaGraphs</td><td>SceneGraphs</td><td>WebQSP</td></tr><tr><td rowspan="4">Inference-only</td><td>Zero-shot</td><td>0.5650</td><td>0.3974</td><td>41.06</td></tr><tr><td>Zero-CoT [18]</td><td>0.5704</td><td>0.5260</td><td>51.30</td></tr><tr><td>CoT-BAG [44]</td><td>0.5794</td><td>0.5680</td><td>39.60</td></tr><tr><td>KAPING [1]</td><td>0.6227</td><td>0.4375</td><td>52.64</td></tr><tr><td rowspan="4">Frozen LLM w/PT</td><td>Prompt tuning</td><td>0.5763 ± 0.0243</td><td>0.6341 ± 0.0024</td><td>48.34 ± 0.64</td></tr><tr><td>GraphToken [31]</td><td>0.8508 ± 0.0551</td><td>0.4903 ± 0.0105</td><td>57.05 ± 0.74</td></tr><tr><td>G-Retriever</td><td>0.8516 ± 0.0092</td><td>0.8131 ± 0.0162</td><td>70.49 ± 1.21</td></tr><tr><td>∆Prompt tuning</td><td>↑ 47.77%</td><td>↑ 28.23%</td><td>↑ 45.81%</td></tr><tr><td rowspan="3">Tuned LLM</td><td>LoRA</td><td>0.8538 ± 0.0353</td><td>0.7862 ± 0.0031</td><td>66.03 ± 0.47</td></tr><tr><td>G-Retriever w/ LoRA</td><td>0.8705 ± 0.0329</td><td>0.8683 ± 0.0072</td><td>73.79 ± 0.70</td></tr><tr><td>∆ LoRA</td><td>↑ 1.95%</td><td>↑ 11.74%</td><td>↑ 10.44%</td></tr></table>

Table 3 demonstrates the effectiveness of our method across three datasets in various configurations. In the inference-only setting, $G$ -Retriever surpasses all baselines. Notably, LLM can perform even better when no graph knowledge is provided (i.e., question only), which might be attributed to the complexity and potential noise in the knowledge. For frozen LLM with prompt tuning, $G$ -Retriever outperforms traditional prompt tuning and GraphToken [31], a graph prompt tuning-based method, with average performance increases of $4 0 . 6 \%$ and $3 0 . 8 \%$ respectively. Furthermore, when tuned with LoRA, $G$ -Retriever achieves the best performance.

Table 4: Retrieval on graphs significantly improves efficiency.   

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Before Retrieval (Avg.)</td><td colspan="3">After Retrieval (Avg.)</td></tr><tr><td># Tokens</td><td># Nodes</td><td>Min/Epoch</td><td># Tokens</td><td># Nodes</td><td>Min/Epoch</td></tr><tr><td>SceneGraphs</td><td>1,396</td><td>19</td><td>123.1</td><td>235 (↓83%)</td><td>5 (↓74%)</td><td>86.8 (↓29%)</td></tr><tr><td>WebQSP</td><td>100,627</td><td>1,371</td><td>18.7</td><td>610 (↓99%)</td><td>18 (1.99%)</td><td>6.2(↓67%)</td></tr></table>

# 6.3 Efficiency Evaluation

The efficiency of our approach is highlighted by the data in Table 4. Implementing our graph-based retrieval significantly decreases the number of tokens required to describe the graphs in text, reduces the number of nodes in graphs, and speeds up the training process. Specifically, for the SceneGraphs dataset, tokens decreased by $83 \%$ , nodes by $74 \%$ , and training time by $29 \%$ . For the WebQSP dataset, tokens decreased by $9 9 \%$ , nodes by $9 9 \%$ , and training time by $67 \%$ . These substantial reductions demonstrate the method’s efficiency and potential in managing large-scale graph data.

# 6.4 Mitigation of Hallucination

To evaluate hallucination, we instructed the models to answer graph-related questions, specifically by identifying supporting nodes or edges from the graph. We assessed the model’s faithfulness using three metrics: the fraction of valid nodes (denoted as Valid Nodes), the fraction of valid edges (denoted as Valid Edges), and the fraction of times the entire set of cited nodes and edges was valid (denoted as Fully Valid Graphs). We manually reviewed 100 responses from both our method and the baseline (i.e.,, LLM with graph prompt tuning). Table 5 shows that $G$ -Retriever significantly reduces hallucinations by $54 \%$ compared to the baseline, as our graph retrieval ensures that the data is sourced directly from the actual graph, leading to fewer hallucinations. See Appendix $\mathbf { G }$ for details.

# 6.5 Ablation Study

In this ablation study, we assess the individual impact of key components within our pipeline. As shown in Table 6, there are performance drops when any of these components are removed, with the graph encoder and textualized graph showing declines of $2 2 . 5 1 \%$ and $1 9 . 1 9 \%$ , respectively. This demonstrates their complementary effects in representing the graph in both textual and embedded formats. Additionally, the retrieval on graphs is also important to the overall performance. Further details are available in Appendix B.3. We also present additional studies on our framework: it is robust to the choice of graph encoders (see Appendix B.4) and benefits from the increased scale of LLMs (see Appendix B.5).

Table 5: Hallucination reduction on the SceneGraphs dataset, measured by fractions of valid nodes, valid edges, and fully valid graphs (where all nodes and edges are correct).   

<table><tr><td></td><td>Baseline</td><td>G-Retriever</td></tr><tr><td>Valid Nodes</td><td>31%</td><td>77%</td></tr><tr><td>Valid Edges</td><td>12%</td><td>76%</td></tr><tr><td>Fully Valid Graphs</td><td>8%</td><td>62%</td></tr></table>

Table 6: Ablation study on the WebQSP dataset showing performance drops $\operatorname { ( H i t } @ 1 )$ when each component is removed.   

<table><tr><td>Method</td><td>Hit@1</td></tr><tr><td>G-Retriever</td><td>70.49</td></tr><tr><td>w/o Graph Encoder</td><td>54.62 (↓22.51%)</td></tr><tr><td>w/o Projection Layer</td><td>69.70 (↓1.11%)</td></tr><tr><td>w/o Textualized Graph</td><td>56.96 (↓19.19%)</td></tr><tr><td>w/o Retrieval</td><td>63.84 (↓9.43%)</td></tr></table>

Additionally, we include a detailed comparison with existing retrieval methods (see Appendix D), a discussion on the complexity (see Appendix E), and demonstrations on how to use $G$ -Retriever to ‘chat with your graph’ (see Appendix H).

# 7 Conclusion

In this work, we introduce a new GraphQA benchmark for real-world graph question answering and present $G$ -Retriever, an architecture adept at complex and creative queries. Experimental results show that $G$ -Retriever surpasses baselines in textual graph tasks across multiple domains, scales effectively with larger graph sizes, and demonstrates resistance to hallucination.

Limitations and Future Work: Currently, $G$ -Retriever employs a static retrieval component. Future developments could investigate more sophisticated RAG where the retrieval is trainable.

# Acknowledgment

BH is supported by the Ministry of Education, Singapore, under the Academic Research Fund Tier 1 (FY2023) (Grant A-8001996-00-00). XB is supported by NUS Grant ID R-252-000-B97-133. The authors would like to express their gratitude to the reviewers for their feedback, which has improved the clarity and contribution of the paper.

References   
[1] Jinheon Baek, Alham Fikri Aji, and Amir Saffari. Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. In Bhavana Dalvi Mishra, Greg Durrett, Peter Jansen, Danilo Neves Ribeiro, and Jason Wei, editors, Proceedings of the 1st Workshop on Natural Language Reasoning and Structured Explanations (NLRSE), pages 78–106, Toronto, Canada, June 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.nlrse-1.7. URL https://aclanthology.org/2023.nlrse-1.7.   
[2] Daniel Bienstock, Michel X Goemans, David Simchi-Levi, and David Williamson. A note on the prize collecting traveling salesman problem. Mathematical programming, 59(1-3):413–420, 1993.   
[3] Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, and Yang Yang. Graphllm: Boosting graph reasoning ability of large language model. arXiv preprint arXiv:2310.05845, 2023.   
[4] Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei, Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, et al. Exploring the potential of large language models (llms) in learning on graphs. arXiv preprint arXiv:2307.03393, 2023.   
[5] Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, and Jiliang Tang. Label-free node classification on graphs with large language models (llms). arXiv preprint arXiv:2310.04668, 2023.   
[6] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. arXiv preprint arXiv:2310.04560, 2023.   
[7] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.   
[8] Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, and Bryan Hooi. Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning. arXiv preprint arXiv:2305.19523, 2023.   
[9] Chinmay Hegde, Piotr Indyk, and Ludwig Schmidt. A nearly-linear time framework for graphstructured sparsity. In International Conference on Machine Learning, pages 928–937. PMLR, 2015.   
[10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.   
[11] Jin Huang, Xingjian Zhang, Qiaozhu Mei, and Jiaqi Ma. Can llms effectively leverage graph structural information: when and why. arXiv preprint arXiv:2309.16595, 2023.   
[12] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions, 2023.   
[13] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.   
[14] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt: A general framework for large language model to reason over structured data. arXiv preprint arXiv:2305.09645, 2023.   
[15] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models on graphs: A comprehensive survey. arXiv preprint arXiv:2312.02783, 2023.   
[16] Minki Kang, Jin Myung Kwak, Jinheon Baek, and Sung Ju Hwang. Knowledge graphaugmented language models for knowledge-grounded dialogue generation, 2023.   
[17] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.   
[18] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.   
[19] Bin Lei, Chunhua Liao, Caiwen Ding, et al. Boosting logical reasoning in large language models through a new framework: The graph of thought. arXiv preprint arXiv:2308.08614, 2023.   
[20] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.   
[21] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.   
[22] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.   
[23] Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, and Xinchao Wang. Graphadapter: Tuning vision-language models with dual knowledge graph. arXiv preprint arXiv:2309.13625, 2023.   
[24] Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, and Jeffrey Xu Yu. A survey of graph meets large language model: Progress and future directions. arXiv preprint arXiv:2311.12399, 2023.   
[25] Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, and Muhan Zhang. One for all: Towards training one graph model for all classification tasks. arXiv preprint arXiv:2310.00149, 2023.   
[26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.   
[27] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[28] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. arXiv preprint arXiv:2310.01061, 2023.   
[29] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. arXiv preprint arXiv:2310.01061, 2023.   
[30] Shirui Pan, Yizhen Zheng, and Yixin Liu. Integrating graphs with large language models: Methods and prospects. arXiv preprint arXiv:2310.05499, 2023.   
[31] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan Halcrow. Let your graph do the talking: Encoding structured data for llms. arXiv preprint arXiv:2402.05862, 2024.   
[32] Chen Qian, Huayi Tang, Zhirui Yang, Hong Liang, and Yong Liu. Can large language models empower molecular property prediction? arXiv preprint arXiv:2307.07443, 2023.   
[33] Yijian Qin, Xin Wang, Ziwei Zhang, and Wenwu Zhu. Disentangled representation learning with large language models for text-attributed graphs. arXiv preprint arXiv:2310.18152, 2023.   
[34] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. arXiv preprint arXiv:1908.10084, 2019.   
[35] Swarnadeep Saha, Prateek Yadav, Lisa Bauer, and Mohit Bansal. Explagraphs: An explanation graph generation task for structured commonsense reasoning. arXiv preprint arXiv:2104.07644, 2021.

[36] Priyanka Sen, Sandeep Mavadia, and Amir Saffari. Knowledge graph-augmented language models for complex question answering. In Bhavana Dalvi Mishra, Greg Durrett, Peter Jansen, Danilo Neves Ribeiro, and Jason Wei, editors, Proceedings of the 1st Workshop on Natural Language Reasoning and Structured Explanations (NLRSE), pages 1–8, Toronto, Canada, June 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.nlrse-1.1. URL https://aclanthology.org/2023.nlrse-1.1.

[37] Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjin Wang, and Yu Sun. Masked label prediction: Unified message passing model for semi-supervised classification. arXiv preprint arXiv:2009.03509, 2020.

[38] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id $\underset { . } { = }$ nnVO1PvbTv.

[39] Shengyin Sun, Yuxiang Ren, Chen Ma, and Xuecang Zhang. Large language models as topological structure enhancers for text-attributed graphs. arXiv preprint arXiv:2311.14324, 2023.

[40] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. Graphgpt: Graph instruction tuning for large language models. arXiv preprint arXiv:2310.13023, 2023.

[41] Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V Chawla, and Panpan Xu. Graph neural prompting with large language models. arXiv preprint arXiv:2309.15427, 2023.

[42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[43] Petar Velickovi ˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua ´ Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.

[44] Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? arXiv preprint arXiv:2305.10037, 2023.

[45] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[46] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.

[47] Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, and Yongfeng Zhang. Natural language is all a graph needs. arXiv preprint arXiv:2308.07134, 2023.

[48] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The value of semantic parse labeling for knowledge base question answering. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 201–206, 2016.

[49] Minji Yoon, Jing Yu Koh, Bryan Hooi, and Ruslan Salakhutdinov. Multimodal graph learning for generative tasks. arXiv preprint arXiv:2310.07478, 2023.

[50] Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, and Xuecang Zhang. Empower text-attributed graphs learning with large language models (llms). arXiv preprint arXiv:2310.09872, 2023.

[51] Junchi Yu, Ran He, and Rex Ying. Thought propagation: An analogical approach to complex reasoning with large language models. arXiv preprint arXiv:2310.03965, 2023.

[52] Jiawei Zhang. Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt. arXiv preprint arXiv:2304.11116, 2023.   
[53] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.   
[54] Ziwei Zhang, Haoyang Li, Zeyang Zhang, Yijian Qin, Xin Wang, and Wenwu Zhu. Graph meets llms: Towards large graph models, 2023.   
[55] Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhi-Hong Deng, Lingpeng Kong, and Qi Liu. Gimlet: A unified graph-text model for instruction-based molecule zero-shot learning. bioRxiv, pages 2023–05, 2023.   
[56] Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, and Jian Tang. Graphtext: Graph reasoning in text space. arXiv preprint arXiv:2310.01089, 2023.   
[57] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

# A Impact Statements

As LLMs are applied to increasingly diverse tasks, their ability to process complex structured data will be increasingly vital. Our work aims to enhance LLMs’ ability to interact with graph-structured data, while resisting hallucination, thus improving model reliability. We also enhance explainability, both by returning the retrieved subgraph, and through the use of conversational interfaces for ‘chatting with a graph’, which allows for better human-AI interaction and for models to behave in a way that is more well-aligned with human expectations.

# B Experiment

# B.1 Implementation Settings

Experiments are conducted using 2 NVIDIA A100-80G GPUs. Each experiment is replicated four times, utilizing different seeds for each run to ensure robustness and reproducibility.

Graph Encoder. We use Graph Transformer [37] as the GNN backbone. Our configuration employs 4 layers, each with 4 attention heads, and a hidden dimension size of 1024.

LLM. We use the open-sourced Llama2-7b [42] as the LLM backbone. In fine-tuning the LLM with LoRA [10], the lora_r parameter (dimension for LoRA update matrices) is set to 8, and lora_alpha (scaling factor) is set to 16. The dropout rate is set to 0.05. In prompt tuning, the LLM is configured with 10 virtual tokens. The number of max text length is 512, the number of max new tokens, i.e., the maximum numbers of tokens to generate, is 32.

PCST. For retrieval over graphs via PCST, for the SceneGraphs dataset, we select the top $k$ nodes and edges, setting $k$ to 3. Here, the cost of edges, denoted as $C _ { e }$ , is set to 1. Regarding the WebQSP dataset, we set $k = 3$ for nodes and $k = 5$ for edges, with the edge cost, $C _ { e }$ , adjusted to 0.5. For the ExplaGraphs dataset, which is characterized by a small graph size averaging 5.17 nodes and 4.25 edges (as detailed in Table 2), the entire graph can fit in the LLM’s context window size. Consequently, we aim to retrieve the whole graph by setting $k$ to 0, effectively returning the original graph unaltered.

Optimization. We use the AdamW [27] optimizer. We set the initial learning rate at 1e-5, with a weight decay of 0.05. The learning rate decays with a half-cycle cosine decay after the warm-up period. The batch size is 4, and the number of epochs is 10. To prevent overfitting and ensure training efficiency, an early stopping mechanism is implemented with a patience setting of 2 epochs.

# B.2 Details of Model Configurations

In our experiments, we consider three model configurations:

1) Inference-only: Using a frozen LLM for direct question answering with textual graph and question, see Figure 4.

![](images/c243253b92256b572b05a8f03f0920fb42ec625bdd8c1e326721a9e8fe0fecea.jpg)  
Figure 4: Model configuration $I$ ) Inference-only.

• Zero-shot. In this approach, the model is given a textual graph description and a task description, and is immediately asked to produce the desired output. No additional examples or demonstrations are provided.

• Zero-CoT. Zero-shot Chain-of-thought (Zero-CoT) prompting [18] is a follow-up to CoT prompting [45], which introduces an incredibly simple zero shot prompt by appending the words "Let’s think step by step." to the end of a question.   
• CoT-BAG. Build-a-Graph Prompting (BAG) [44] is a prompting technique that adds "Let’s construct a graph with the nodes and edges first." after the textual description of the graph is explicitly given.   
• KAPING. KAPING [1] is a zero-shot knowledge-augmented prompting method for knowledge graph question answering. It first retrieves triples related to the question from the graph, then prepends them to the input question in the form of a prompt, which is then forwarded to LLMs to generate the answer.

2) Frozen LLM w/ prompt tuning (PT): Keeping the parameters of the LLM frozen and adapting only the prompt. This includes soft prompt tuning (see Figure 5a), GraphToken [31], which is a graph prompt tuning method, and our $G$ -Retriever method (see Figure5b).

![](images/06b0a907a29c68cb609c32c395c6267e02568bf0170840b12f7efcbd39bdca82.jpg)  
Figure 5: Model configuration 2) Frozen LLM w/ prompt tuning.

3) Tuned LLM: Fine-tuning the LLM with LoRA. This includes standard fine-tuning of an LLM for downstream tasks using LoRA (see Figure 6a) and G-Retriever with LoRA (see Figure 6b).

![](images/0666a9dbb857d1bf6e0513be169904007292336df23edbbfb8999a47d976ac35.jpg)  
Figure 6: Model configuration 3) Tuned LLM.

# B.3 Details of Ablation Study

This section illustrates the modifications made to the original architecture in the ablation study, as presented in Figure 7.

Without Graph Encoder (w/o GraphEncoder): In this setting, we replaced the graph encoder with trainable soft tokens, setting the number of these virtual tokens to 10.

Without Projection Layer (w/o Projection Layer): Here, we removed the projection layer following the graph encoder. We configured the output dimension of the graph encoder to be 4,096, matching the hidden dimension of Llama2-7b. This allows the output graph token (the yellow token in Figure 7b) to be concatenated directly with the LLM tokens (blue tokens).

Without Textualized Graph (w/o Textualized Graph): In this configuration, we modified the textual input to the LLM. Rather than using a combination of the question and the textualized graph, we solely used the question.

![](images/47eef358c570ccfaba3ab18101eb6a73468a55c39a87b8c69dc80731088276ba.jpg)  
Figure 7: Ablation study configurations.

# B.4 The Choice of Graph Encoder

In addition to the Graph Transformer [37], we explore other GNNs as the graph encoder, such as GCN [17] and the GAT [43]. The comparative results of these models on the WebQSP and ExplaGraphs datasets are presented in Table 7.

Table 7: Performance of different graph encoders on the WebQSP and ExplaGraphs datasets.   

<table><tr><td>Graph Encoder</td><td>WebQSP</td><td>ExplaGraphs</td></tr><tr><td>GCN [17]</td><td>70.70</td><td>0.8394</td></tr><tr><td>GAT [43]</td><td>70.27</td><td>0.8430</td></tr><tr><td>Graph Transformer [37]</td><td>70.49</td><td>0.8516</td></tr></table>

The results demonstrate that our proposed method exhibits consistent robustness across different graph encoders. Notably, all three encoders – GCN, GAT, and GraphTransformer – demonstrate competitive and closely aligned performance on the WebQSP dataset, with $\mathrm { H i t } @ 1$ scores of 70.70, 70.27, and 70.49, respectively. However, the performance differentiation becomes more pronounced on the ExplaGraphs dataset, where GraphTransformer exhibits a superior $\mathrm { H i t } @ 1$ score of 0.8516, followed by GAT and GCN with scores of 0.8430 and 0.8394, respectively. This variation in performance across the datasets highlights the importance of encoder selection based on the specific characteristics and requirements of the dataset.

# B.5 The Choice of LLM

As for the choice of LLM, we considered both Llama2-7b and Llama2-13b. Our experiments demonstrate that stronger LLMs enhance the effectiveness of our method, as shown in Table 8, indicating that it benefits from the increased scale of the LLMs.

Table 8: Performance of different LLMs on the WebQSP dataset.   

<table><tr><td>LLM</td><td>Llama2-7b</td><td>Llama2-13b</td></tr><tr><td>Hit@1</td><td>70.49</td><td>75.58</td></tr></table>

# C GraphQA Benchmark

In this section, we detail how our GraphQA benchmark differs from the original datasets, including the specific processing steps we employed. For concrete examples that illustrate the differences between the raw text in the original dataset and in our GraphQA benchmark, please refer to Table 9.

ExplaGraphs. The original dataset1 [35] represents relationships using triplets. We have standardized this format by converting the triplets into a graph representation. Specifically, each head and tail in a triplet is transformed into a node, and the relation is transformed into an edge. Since the test dataset labels are not available, we have utilized only the training and validation (val) datasets from the original collection. We further divided these into training, val, and test subsets, using a 6:2:2 ratio.

Table 9: Comparison of text formats in original datasets and our GraphQA benchmark.   

<table><tr><td colspan="3"></td></tr><tr><td>Dataset</td><td>Original dataset</td><td>GraphQA Benmark</td></tr><tr><td>ExplaGraphs</td><td>(entrapment; capable of; being abused) (being abused; created by; police) (police; capable of; harm) (harm; used for; people) (people; part of; citizens) &quot;width&quot;: 500, &quot;objects&quot;: &quot;681267&quot;: &quot;name&quot;: &quot;banana&quot;, &quot;h&quot;: 34, &quot;relations&quot;: [&quot;object&quot;:</td><td>node_id,node_attr\n 0,entrapment\n 1,being abused\n 2,police\n 3,harm\n 4,people\n 5,citizens\n src,edge_attr,dst\n 0,capable of,1\n 1,created by,2\n 2,capable of,3\n 3,used for,4\n 4,part of,5\n</td></tr><tr><td>SceneGraphs</td><td>&quot;681262&quot;, &quot;name&quot;: &quot;to the left of&quot;], &quot;w&quot;: 64, &quot;attributes&quot;: [&quot;small&quot;, &quot;yellow&quot;], &quot;y&quot;: 55, &quot;x&quot;: 248, &quot;681265&quot;: &quot;name&quot;: &quot;spots&quot;, &quot;h&quot;: 16, &quot;relations&quot;: [], &quot;w&quot;: 26, &quot;attributes&quot;: [], &quot;y&quot;: 92, &quot;x&quot;: 245, &quot;681264&quot;: &quot;name&quot;: &quot;bananas&quot;, &quot;h&quot;: 50, &quot;relations&quot;: [&quot;object&quot;: &quot;681259&quot;, &quot;name&quot;: &quot;to the left of&quot;], &quot;w&quot;: 49, &quot;attributes&quot;: [&quot;small&quot;, &quot;yellow&quot;], &quot;y&quot;: 32, &quot;x&quot;: 268, &quot;681263&quot;: &quot;name&quot;: &quot;picnic&quot;, &quot;h&quot;: 374, &quot;relations&quot;: [], &quot;w&quot;: 499, &quot;attributes&quot;: [&quot;delicious&quot;], &quot;y&quot;: 0, &quot;x&quot;: 0, &quot;681262&quot;: &quot;name&quot;: &quot;straw&quot;, &quot;h&quot;: 95, &quot;relations&quot;: [&quot;object&quot;: &quot;681268&quot;, &quot;name&quot;: &quot;to the right of&quot;, &quot;object&quot;: &quot;681267&quot;, &quot;name&quot;: &quot;to the right of&quot;, &quot;object&quot;: &quot;681253&quot;, &quot;name&quot;: &quot;to the right of&quot;], &quot;w&quot;: 15, &quot;attributes&quot;: [&quot;white&quot;, &quot;plastic&quot;], &quot;y&quot;: 55, &quot;x&quot;: 402, &quot;681261&quot;: &quot;name&quot;: &quot;meat&quot;, &quot;h&quot;: 27, &quot;relations&quot;: [&quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;on&quot;, &quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;inside&quot;], &quot;w&quot;: 24, &quot;attributes&quot;: [&quot;small&quot;, &quot;brown&quot;, &quot;delicious&quot;], &quot;y&quot;: 123, &quot;x&quot;: 68, &quot;260&quot;: &quot;name: &quot;rice&quot;, &quot;h: 57, &quot;relations&quot;: [&quot;object: &quot;681255&quot;, &quot;name&quot;: &quot;n&quot;, &quot;object&quot;: &quot;681258&quot;, &quot;name&quot;: &quot;to the left of&quot;], &quot;w&quot;: 93, &quot;attributes&quot;: [&quot;piled&quot;, &quot;white&quot;], &quot;: 162, &quot;x&quot;: 57, &quot;681269&quot;: &quot;name&quot;: &quot;onions&quot;, &quot;h&quot;: 16, &quot;relations&quot;: [, &quot;w&quot;: 4, &quot;attributes&quot;: [&quot;green&quot;], &quot;y&quot;: 147, &quot;x&quot;: 90, &quot;681268&quot;: &quot;name&quot;: &quot;tablecloth&quot;, &quot;h&quot;: 374, &quot;relations&quot;: [&quot;object&quot;: &quot;681262&quot;, &quot;name&quot;: &quot;to the left of&quot;], &quot;w&quot;: 396, &quot;attributes&quot;: [&quot;white&quot;], &quot;y&quot;: 0, &quot;x&quot;: 0, &quot;681258&quot;: &quot;name&quot;: &quot;bowl&quot;, &quot;h&quot;: 99, &quot;relations&quot;: [&quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;next to&quot;, &quot;object&quot;: &quot;681257&quot;, &quot;name&quot;: &quot;of&quot;, &quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;near&quot;, &quot;object&quot;: &quot;681256&quot;, &quot;name&quot;: &quot;to the right of&quot;, &quot;object&quot;: &quot;681260&quot;, &quot;name&quot;: &quot;to the right of&quot;, &quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;to the right of&quot;], &quot;w&quot;: 115, &quot;attributes&quot;: [&quot;full&quot;], &quot;y&quot;: 184, &quot;x&quot;: 178, &quot;681259&quot;: &quot;name&quot;: &quot;plantains&quot;, &quot;h&quot;: 70, &quot;relations&quot;: [&quot;object&quot;: &quot;681264&quot;, &quot;name&quot;: &quot;to the right of&quot;], &quot;w&quot;: 45, &quot;attributes&quot;: [&quot;red&quot;], &quot;y&quot;: 0, &quot;x&quot;: 346, &quot;681256&quot;: &quot;name&quot;: &quot;spoon&quot;, &quot;h&quot;: 65, &quot;relations&quot;: [&quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;on&quot;, &quot;object&quot;: &quot;681257&quot;, &quot;name&quot;: &quot;to the left of&quot;, &quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;in&quot;, &quot;object&quot;: &quot;681258&quot;, &quot;name&quot;: &quot;to the left of&quot;], &quot;w&quot;: 140, &quot;attributes&quot;: [&quot;large&quot;, &quot;metal&quot;, &quot;silver&quot;], &quot;y&quot;: 196, &quot;x&quot;: 0, &quot;681257&quot;: &quot;name&quot;: &quot;dish&quot;, &quot;h&quot;: 81, &quot;relations&quot;: [&quot;object&quot;: &quot;681258&quot;, &quot;name&quot;: &quot;inside&quot;, &quot;object&quot;: &quot;681256&quot;, &quot;name&quot;: &quot;to the right of&quot;, &quot;object&quot;: &quot;681258&quot;, &quot;name&quot;: &quot;in&quot;, &quot;object&quot;: &quot;681255&quot;, &quot;name&quot;: &quot;to the right of&quot;], &quot;w&quot;: 108, &quot;attributes&quot;: [&quot;cream colored&quot;], &quot;y&quot;: 199, &quot;x&quot;: 187, &quot;681254&quot;: &quot;name&quot;: &quot;meal&quot;, &quot;h&quot;: 111, &quot;relations&quot;: [], &quot;w&quot;: 130, &quot;attributes&quot;: [ &quot;y&quot;: 121, &quot;x&quot;: 58, &quot;681255: &quot;name&quot;: &quot;plate&quot;, &quot;h&quot;: 138, &quot;relations&quot;: [&quot;object&quot;: &quot;681257&quot;, &quot;name&quot;: &quot;to the left of&quot;, &quot;object&quot;: &quot;681254&quot;, &quot;name&quot;: &quot;of&quot;, &quot;object&quot;: &quot;681254&quot;, &quot;name&quot;: &quot;with&quot;, &quot;object&quot;: &quot;681258&quot;, &quot;name&quot;: &quot;near&quot;, &quot;object&quot;: &quot;681258&quot;, &quot;name&quot;: &quot;to the left of&quot;], &quot;w&quot;: 176, &quot;attributes&quot;: [&quot;white&quot;, &quot;full&quot;], &quot;y&quot;: 111, &quot;x&quot;: 30, &quot;81253&quot;: &quot;name&quot;: &quot;banana&quot;, &quot;h&quot;: 30, &quot;relations&quot;: [&quot;object&quot;: &quot;681262&quot;, &quot;name&quot;: &quot;to the left of&quot;], &quot;w&quot;: 73, &quot;attributes&quot;: [&quot;small&quot;, &quot;yellow&quot;], &quot;y&quot;: 87, &quot;x&quot;: 237, &quot;height&quot;: 375</td><td>node_id,node_attr 0,&quot;name: banana; attribute: small, yellow; (x,y,w,h): (248, 55, 64, 34)&quot; 1,&quot;name: spots; (x,y,w,h): (245, 92, 26, 16)&quot; 2,&quot;name: bananas; attribute: small, yellow; (x,y,w,h): (268, 32, 49, 50) 3,&quot;name: picnic; attribute: delicious; (x,y,w,h): (0, 0, 499, 374)&quot; 4,&quot;name: straw; attribute: white, plastic; (x,y,w,h): (402, 55, 15, 95)&quot; 5,&quot;name: meat; attribute: small, brown, delicious; (x,y,w,h): (68, 123, 24, 27)&quot; 6,&quot;name: rice; attribute: piled, white; (x,y,w,h): (57, 162, 93, 57)&quot; 7,&quot;name: onions; atribute: green; (x,y,w,h): (90, 147, 24, 16)&quot; 8,&quot;name: tablecloth; attribute: white; (x,y,w,h): (0, 0, 396, 374)&quot; 9,&quot;name: bowl; attribute: full; (x,y,w,h): (178, 184, 115, 99)&quot; 10,&quot;name: plantains; attribute: red; (x,y,w,h): (346, 0, 45, 70)&quot; 11,&quot;name: spoon; attribute: large, metal, silver; (x,y,w,h): (0, 196, 140, 65)&quot; 12,&quot;name: dish; attribute: cream colored; (x,y,w,h): (187, 199, 108, 81)&quot; 13,&quot;name: meal; (x,y,w,h): (58, 121, 130, 111)&quot; 14,&quot;name: plate; attribute: white, full; (x,y,w,h): (30, 111, 176, 138)&quot; 15,&quot;name: banana; attribute: small, yellow; (x,y,w,h): (237, 87, 73, 30)&quot; src,edge_attr,dst 0,to the left of,4\n 2,to the left of,10\n 4,to the right of,8\n 4,to the right of,0\n 4,to the right of,15\n 5,on,14\n 5,inside,14\n 6,on,14\n 6,to the left of,9\n 8,to the left of,4\n 9,next to,14\n 9,of,12\n 9,near,14\n 9,to the right of,11\n 9,to the right of,6\n 9,to the right of,14\n 10,to the right of,2\n 11,on,14\n 11,to the left of,12\n 11,in,14\n 11,to the left of,9\n 12,inside,9\n 12,to the right of,11\n 12,in,9\n 12,to the right of,14\n 14,to the left of,12\n 14,of,13\n 14,with,13\n 14,near,9\n 14,to the left of,9\n 15,to the left of,4\n node_id,node_attr\n 0,fedex cup\n 1,m.0n1v8cy\n 2,brandt snedeker\n 3,m.08q5wy\n</td></tr><tr><td>WebQSP</td><td>[&#x27;FedEx Cup&#x27;, &#x27;sports.sports_award_type.winners&#x27;, &#x27;m.On1v8cy&#x27;], [&#x27;Brandt Snedeker&#x27;, &#x27;sports.sports_award_winner.awards&#x27;, &#x27;m.On1v8cy&#x27;], [&#x27;FedEx Cup&#x27;, &#x27;common.topic.article&#x27;, &#x27;m.08q5wy&#x27;], [&#x27;FedEx Cup&#x27;, &#x27;common.topic.notable_for&#x27;, &#x27;g.12559n8g_&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;freebase.type_profile.published&#x27;, &#x27;Published&#x27;], [&#x27;FedEx Cup&#x27;, &#x27;common.topic.notable_types&#x27;, &#x27;Sports League Award Type&#x27;], [&#x27;m.On1v8cy&#x27;, &#x27;sports.sports_award.award_winner&#x27;, &#x27;Brandt Snedeker&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;type.type.expected_by&#x27;, &#x27; Award&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;common.topic.article&#x27;, &#x27;m.06zxtxj&#x27;], [&#x27;2012 PGA Tour&#x27;, &#x27;sports.sports_league_season.awards&#x27;, &#x27;m.On1v8cy&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;freebase.type_hints.included_types&#x27;, &#x27;Topic&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;type.type.domain&#x27;, &#x27;Sports&#x27;], [&#x27;m.On1v8cy&#x27;, &#x27;sports.sports_award.award&#x27; , &#x27;FedEx Cup&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;freebase.type_profile.strict_included_types&#x27;, &#x27;Topic&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;freebase.type_profile.kind&#x27;, &#x27;Classification&#x27;], [&#x27;m.0n1v8cy&#x27;, &#x27;sports.sports_award.season&#x27;, &#x27;2012 PGA Tour&#x27;], [&#x27;Sports League Award Type&#x27;, &#x27;type.type.properties&#x27;, &#x27;Winners&#x27;]]</td><td>4,g.12559n8g_\n 5,sports league award type\n 6,published\n 7,award\n 8,m.06zxtxj\n 9,2012 pga tour\n 10,topic\n 11,sports\n 12,classification\n 13,winners\n src,edge_attr,dst 0,sports.sports_award_type.winners,1 2,sports.sports_award_winner.awards,1 0,common.topic.article,3 0,common.topic.notable_for,4 5,freebase.type_profile.published,6 0,common.topic.notable_types,5 1,sports.sports_award.award_winner,2 5,type.type.expected_by,7 5,common.topic.article,8 9,sports.sports_league_season.awards,1 5,freebase.type_hints.included_types,10 5,type.type.domain,11 1,sports.sports_award.award,0 5,freebase.type_profile.strict_included_types,10 5,freebase.type_profile.kind,12 1,sports.sports_award.season,9 5,type.type.properties,13</td></tr></table>

SceneGraphs. The original GQA dataset is designed for real-world visual reasoning and compositional question answering, aiming to address key shortcomings of previous VQA datasets [13]. It comprises 108k images, each associated with a Scene Graph. In our study, we focus differently on graph question answering; hence, we did not utilize the image counterparts, leveraging only the scene graphs from the original dataset. Additionally, the original dataset describes images using JSON files. We simplified the object IDs to suit our research needs. We randomly sampled $1 0 0 \mathrm { k }$ samples from the original dataset and divided them into training, validation, and test subsets, following a 6:2:2 ratio.

WebQSP. We follow the preprocessing steps from $\mathbf { R o G } ^ { 2 }$ [28]. The original dataset uses a list of triplets format, which we have transformed into our unified graph format. Furthermore, to avoid discrimination between capital and lowercase words, we have converted all words to lowercase. We used the same dataset split as in the original dataset.

Contribution of the GraphQA Benchmark. We acknowledge that the GraphQA benchmark involves converting three existing graph datasets into a uniform format. However, we believe this standardization provides significant value to the research community in several ways:

• Task Introduction: Unlike existing graph question-answering benchmarks that focus on small or synthetic graphs, our benchmark includes real-world applications and frames them as graph question-answering tasks.   
Standardization: A key and significant effort of this benchmark is the standardization and processing of diverse datasets into a uniform format suitable for GraphQA tasks. These datasets, previously used in different contexts, are redesigned to focus specifically on GraphQA, ensuring consistent and comparable evaluations across models.   
• Accessibility: We have open-sourced the GraphQA benchmark, providing a unified format that simplifies model application across multiple datasets. This reduces the complexity of handling various data structures and preprocessing pipelines, lowering barriers for new researchers and encouraging broader participation. We have already seen several novel works using our GraphQA benchmark, and we expect rapid adoption within the LLM and GNN communities.   
• Baseline Comparisons: The benchmark offers baseline performance metrics, helping researchers identify the strengths and weaknesses of new approaches compared to established baselines.

# D Graph Retrieval-Augmented Generation (GraphRAG)

# D.1 Elaboration on PCST-Based Retrieval

Modeling motivation. We formulate subgraph retrieval as a Prize-Collecting Steiner Tree (PCST) optimization problem. This is motivated by the need to find a connected subgraph containing most relevant nodes and edges, a goal that aligns well with the objectives of PCST: maximizing node values while minimizing edge costs. Though not universally acknowledged as optimal, we have empirically validated its effectiveness.

Effectiveness compared to other retrieval baselines. To validate the effectiveness of our PCSTbased retrieval approach, we compared it against several baselines: (1) top-k triples retrieval, i.e., KAPING [1], which retrieves the top-k triples related to the query and incorporates them into the prompt for the LLM; (2) top- $\mathbf { \nabla } \cdot \mathbf { k }$ nodes plus neighbors, which retrieves the top- $\mathbf { \nabla } \cdot \mathbf { k }$ nodes and their one-hop neighbors, capturing local context; (3) shortest path retrieval, which retrieves the top- $\mathbf { \nabla } \cdot \mathbf { k }$ nodes and computes the shortest paths between them.

For all methods, we set $k = 5$ and used llama2-7b-chat as the LLM. The results, presented in Table 10, show that our PCST-based retrieval method achieves the highest accuracy $( \mathrm { H i t } @ 1 )$ of $6 6 . 1 7 \%$ on the WebQSP dataset, outperforming all baseline methods.

Table 10: Comparison of retrieval methods on the WebQSP dataset.   

<table><tr><td>Method</td><td>Hit@1</td></tr><tr><td>PCST retrieval</td><td>66.17</td></tr><tr><td>top-k triples retrieval (KAPING)</td><td>52.64</td></tr><tr><td>top-k nodes plus its neighbors</td><td>49.82</td></tr><tr><td>shortest path retrieval</td><td>55.20</td></tr></table>

# D.2 Advantages of Subgraph-Based Retrieval

Context-Relevant. Selecting nodes and edges in isolation may overlook neighborhood information. In contrast, PCST-based retrieval is guaranteed to return a connected subgraph, capturing the graph context during the retrieval process. This approach retrieves not only high-relevance nodes or edges but also “bridge” elements that connect these with contextually significant nodes or edges, which are crucial for generating a comprehensive response.

Size Management. Compared to the shortest path method, PCST retrieval provides greater control over the size of the retrieved subgraph. By adjusting the prizes and costs on nodes and edges, users can fine-tune the subgraph’s extent. In contrast, the shortest path approach lacks the ability to control the distance between the top-k nodes, which can lead to disconnected subgraphs or the inclusion of unnecessarily long paths.

# D.3 The Impact of K for Retrieval

We identify the most relevant nodes and edges and use a $\mathbf { k }$ -nearest neighbors retrieval approach (see Equation 6). Small k values may omit crucial knowledge or information relevant to the query, while large k values could introduce excessive information, distracting the model from the essential details. To evaluate the impact of the number of $\mathrm { k }$ , we have conducted additional experiments by varying the choice of $\mathbf { k }$ to 3, 5, 10, and 20.

Table 11: The impact of k on the webqsp dataset.   

<table><tr><td>k</td><td>3</td><td>5</td><td>10</td><td>20</td></tr><tr><td>Hit@1</td><td>0.6977</td><td>0.7063</td><td>0.7248</td><td>0.7039</td></tr></table>

As shown in Table 11, the $\mathrm { H i t } @ 1$ metric initially rises for small k values, peaks at a certain point, and then declines for large $\mathrm { k }$ values. Determining the optimal $\mathbf { k }$ value can be achieved through techniques like cross-validation using a validation set.

# D.4 The Choice of Similarity Function

The choice of similarity function is also important. In this work, we use cosine similarity, a widely adopted metric for measuring vector similarity in models that process vision and language. For instance, CLIP also employs cosine similarity to assess the similarity between text and image features. Although it might not be the optimal choice, we believe that cosine similarity is a general, representative, and valid choice for facilitating fast retrieval tasks.

# E Discussion on the Complexity

# E.1 The integration of GNNs, LLMs and GraphRAG

G-Retriever is framework integrate the strengths of GNNs, LLMs and GraphRAG. The $\mathrm { L L M } { + } \mathrm { X }$ framework, which involves enriching LLMs with multi-modal capabilities by integrating an LLM with an encoder from another modality, is a widely adopted approach. Notable examples include Llava, MiniGPT-4, and Flamingo, among others. They are not complex in terms of understanding or implementation. Regarding the integration of GraphRAG, it does not require training and can be implemented during the preprocessing stage or on the fly. This approach does not significantly increase time complexity or computational complexity. On the contrary, it can substantially reduce the size of the graph (e.g., eliminating $9 9 \%$ of nodes in the WebQSP dataset), which in turn speeds up the overall running time (e.g., reducing it from $1 8 . 7 \mathrm { m i n }$ /epoch to $6 . 2 \mathrm { m i n }$ /epoch on the WebQSP dataset).

# E.2 Computational Resources

Utilizing two A100 GPUs, each with 80GB of memory, we conducted tests on Llama2-7b and WebQSP datasets. Our experiments had a training batch size of 16 and an evaluation batch size of 32, yielding the following results.

These results highlight efficiency improvements via graph RAG, which significantly reduces graph size (e.g., eliminating $9 9 \%$ of nodes in the WebQSP dataset) and speeds up running time.

Table 12: Performance and Efficiency of Various Methods on the WebQSP dataset.   

<table><tr><td>Settting</td><td>Method</td><td>Hit@1</td><td>Time</td></tr><tr><td>Inference-only</td><td>Question only Textual graph and question</td><td>61.16 41.06</td><td>31 min 40 min</td></tr><tr><td>Frozen LLM w/ PT</td><td>Prompt Tuning G-Retriever</td><td>48.34</td><td>18.7 min/epoch</td></tr><tr><td>Tuned LLM</td><td>LoRA G-Retriever w/ LoRA</td><td>70.49 66.03 73.79</td><td>6.2 min/epoch 19 min/epoch 6.9 min/epoch</td></tr></table>

# F Discussion on Explainability

We believe G-Retriever enhances explainability in the following ways:

Retrieved subgraph. By returning the most relevant subgraph in response to a query, users can see which parts of the graph are considered important for the answer. This helps users understand the basis of the model’s responses. For example, if users want to understand why certain information is present or absent in the LLM’s response, they can inspect the subgraph to see whether such information is present or absent in the retrieved subgraph.

Conversational Interface. G-Retriever allows users to ask follow-up questions and receive detailed natural language explanations. For example, if a user questions the LLM’s response, they can ask, “Why do you think [xxx]? Please explain your answer.” This interactive capability enables users to explore the model’s reasoning process and gain deeper insights into how it interprets graph data.

# G Hallucination in Graph LLMs

In this section, we present quantitative results regarding hallucinations in the SceneGraphs dataset.

Baseline. For our baseline, we adapted MiniGPT-4 [57] to graph contexts. This approach involves a frozen LLM interacting with a trainable GNN that encodes graph data as a soft prompt, denoted as LLM $\cdot +$ Graph Prompt Tuning. We focus on graph prompt tuning as the baseline, instead of converting the graph into text, since the textual representation of the graph is large and consistently exceeds the input token limits of LLMs.

Experiment Design. We instructed the LLM to answer graph-related questions and to list nodes or edges in the explanation graph that support its answers. Since standard answers for these questions do not exist, allowing the LLM to respond flexibly, it becomes challenging to evaluate its responses. To address this, we manually examined 100 responses generated by our method and the LLM with graph prompt tuning, verifying whether the nodes and edges referenced in the LLM’s output actually exist in the graph.

Evaluation Metrics. We assessed the model’s faithfulness using three metrics: the fraction of valid nodes (denoted as Valid Nodes), the fraction of valid edges (denoted as Valid Edges), and the fraction of times the entire set of nodes and edges cited was valid (denoted as Fully Valid Graphs).

Results. The results, as depicted in Table 5, illustrate the comparative effectiveness of the $G$ -Retriever over the baseline LLM $\cdot +$ Graph Prompt Tuning method in reducing hallucinations. The LLM $^ +$ Graph Prompt Tuning approach demonstrated a significantly lower accuracy in referencing graph elements, with only $31 \%$ of nodes and $12 \%$ of edges being valid, and the entire set of nodes and edges being valid only $8 \%$ of the time. In contrast, $G$ -Retriever showed substantial improvements: $7 7 \%$ validity in nodes, $76 \%$ in edges, and $62 \%$ in the overall validity of referenced node-edge sets. These results underscore the significant reduction in hallucinations with $G$ -Retriever, particularly in the challenging task of accurately citing both nodes and edges in graph-based contexts.

# H Demonstrations

We demonstrate the interaction capabilities of G-Retriever with creative questions on different datasets: ExplaGraphs, SceneGraphs, and WebQSP. These examples are showcased in Tables 13, 14, and 15, respectively. Additionally, the examples are visualized in Figure 1.

Table 13: Example of creative chat on ExplaGraphs.   

<table><tr><td>Graph</td></tr></table>

<table><tr><td></td><td>src,edge_attr,dst 0,receives action,1 2,part of,0 1,used for,3 3,capable of,4 4,created by,5 4,not capable of,6 4,is not a,7</td></tr><tr><td>User Model</td><td>Write a 500-word argument essay based on the information provided in the explanation graph. The explanation graph provides a detailed analysis of the reasons why women undergo cosmetic surgery. According to the graph, the main reasons include the desire to change one&#x27;s appearance, the influence of societal beauty standards, and the addiction to searching for perfection. These</td></tr><tr><td></td><td>reasons are further supported by the various nodes and edges in the graph, which provide additional insights into the motivations and consequences of cosmetic surgery. Firstly, the node labeled &quot;human&quot; highlights the fundamental human desire to feel attractive and confident in one&#x27;s appearance. This desire is deeply ingrained in our culture, where beauty is often associated with happiness, success, and social acceptance. As a result, many women feel pressure to conform to societal beauty standards, which can lead to feelings of insecurity and low self-esteem. In response to these feelings, some women may turn to cosmetic surgery as a means of improving their appearance and boosting their self-confidence. Secondly, the node labeled &quot;women&quot; highlights the gendered nature of cosmetic surgery. While men also undergo cosmetic surgery, the majority of cosmetic surgery patients are women. This is likely due to the fact that women are more likely to be exposed to societal beauty standards and are more likely to internalize these standards. As a result, women may be more likely to feel insecure about their appearance and more likely to seek cosmetic surgery as a means of improving their looks. Thirdly, the node labeled &quot;addiction&quot; highlights the potential negative consequences of cosmetic surgery. While some women may initially feel improved confidence and self-esteem after un- dergoing cosmetic surgery, the graph suggests that this initial boost may not last. In fact, some women may become addicted to the constant pursuit of perfection, leading to a cycle of surgery and dissatisfaction. This can have serious psychological and emotional consequences, including depression, anxiety, and a loss of self-esteem. Fourthly, the node labeled &quot;searching for perfection&quot; highlights the unrealistic expectations that many women have about the results of cosmetic surgery. The graph suggests that women may have unrealistic expectations about the ability of cosmetic surgery to improve their appearance and boost their self-esteem. This can lead to disappointment and dissatisfaction with the results of surgery, as well as a continued pursuit of perfection through additional surgery. Finally, the node labeled &quot;stop&quot; highlights the importance of critical thinking and self-reflection when it comes to cosmetic surgery. The graph suggests that women should be aware of the potential negative consequences of cosmetic surgery and should carefully consider the reasons why they are undergoing surgery. They should also be aware of the unrealistic expectations that they may have about the results of surgery and should be prepared to address any negative emotions or feelings of dissatisfaction that may arise. In conclusion, the explanation graph provides a detailed analysis of the reasons why women undergo cosmetic surgery. The graph highlights the fundamental human desire to feel attractive and confident in one&#x27;s appearance, the gendered nature of cosmetic surgery, the potential negative consequences of cosmetic surgery, and the importance of critical thinking and self-reflection when it comes to surgery. By understanding these reasons and the potential consequences of cosmetic surgery, women can make informed decisions about whether or not to undergo surgery and can</td></tr></table>

Table 14: Example of creative chat on SceneGraphs. We utilize GPT-4 to generate an image from the graph, assisting readers in visualizing it   

<table><tr><td>Graph node_id,node_attr</td><td>0,name: head; attribute: black; (x,y,w,h): (386, 185, 8, 4) 1,name: swimming pool; attribute: small, water; (x,y,w,h): (465, 92, 35, 3) 2,name: shore; (x,y,w,h): (49, 85, 436, 108) 3,name: fence; attribute: wired; (x,y,w,h): (11, 158, 167, 46) 4,name: mountains; attribute: covered; (x,y,w,h): (393, 53, 107, 12) 5,name: water; attribute: blue; (x,y,w,h): (0, 50, 499, 154) 6,name: sheep; attribute: walking, grazing; (x,y,w,h): (243, 191, 20, 13) 7,name: sky; attribute: blue, clear, cloudy; (x,y,w,h): (0, 0, 501, 60) 8,name: sheep; attribute: white, fluffy; (x,y,w,h): (409, 176, 13, 10) 9,name: field; attribute: large, green, grassy; (x,y,w,h): (0, 138, 503, 112) 10,name: grass; (x,y,w,h): (0, 84, 500, 166) 11,name: sheep; attribute: white, fluffy; (x,y,w,h): (398, 178, 13, 11) 12,name: sheep; attribute: white, fluffy; (x,y,w,h): (385, 179, 17, 11) 13,name: sheep; attribute: white, fluffy; (x,y,w,h): (364, 181, 14, 10) 14,name: shore; attribute: sandy; (x,y,w,h): (190, 123, 171, 12) src, edge_attr, dst</td></tr></table>

![](images/021d71b898f99690908938b23d664a05a3d6f845c0d55558df837ccde11fe968.jpg)

<table><tr><td>User</td><td>Write an 500-word advertisement for this place in the scene graph that would make people want to visit it.</td></tr><tr><td>Model</td><td>Sure! Here&#x27;s an example of a 500-word advertisement for the place based on the scene graph: &quot;Experience the breathtaking beauty of our place, where the rolling hills meet the sparkling water of the swimming pool. Take a leisurely stroll through the lush green fields, and let the gentle breeze caress your face as you gaze out at the picturesque view. The sheep graze peacefully in the distance, their white coats glistening in the sunlight. The sky is a brilliant blue, with fuffy white clouds drifting lazily across. Come and bask in the serene ambiance of our place, where the natural beauty will leave you in awe.&quot;</td></tr></table>

Table 15: Example of creative conversation on the WebQSP dataset.   

<table><tr><td>node_id, node_attr</td></tr><tr><td>8, benjamin franklin 63, glass harmonica 98, lightning rod 99, stealing god&#x27;s thunder: benjamin franklin&#x27;s lightning rod and the invention of america 227, benjamin franklin worked with thomas jefferson 442, franklin stove 525, bifocals</td><td></td></tr><tr><td>1044, prokop divi src, edge_attr, dst Graph</td><td>706, benjamin franklin drawing electricity from the sky 706, visual_art.artwork.art_subject, 8 8, base.kwebbase.kwtopic.connections_from, 227 8, book.book_subject.works, 99</td></tr><tr><td>525, law.invention.inventor, 8 442, law.invention.inventor, 8 98, law.invention.inventor, 1044 8, law.inventor.inventions, 98 63, law.invention.inventor, 8 98, law.invention.inventor, 8 8, law.inventor.inventions, 525</td><td>8, base.argumentmaps.innovator.original_ideas, 442 442, base.argumentmaps.original_idea.innovator, 8</td></tr><tr><td>User</td><td>8, law.inventor.inventions, 442 8, law.inventor.inventions, 63 Qu l graph, highlighting interesting facts and connections in an engaging way. G al related to Benjamin Franklin&#x27;s inventions and other notable achievements: </td></tr><tr><td>Model</td><td> positive and negative charges. groundbreaking invention that improved the lives of millions of people around the world.  o  modern heating systems. Labouoe ui a veheeoel o p o properti  electricity and thecolaborativeeforts helped to advance the ld oelecrical engneig. inventors and scientists, including Thomas Edison and Nikola Tesla. Divi, helped to accelerate the pace of innovation and pave the way for future breakthroughs. L</td></tr></table>

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: As shown in the abstract and introduction

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limitation of this work in Section 7.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: The paper does not include theoretical results.

Guidelines:

• The answer NA means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper provides comprehensive details on the experimental setup for reproducibility, as outlined in Section 6 and Appendix B.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.   
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Our codes and datasets are available at: https://anonymous.4open.   
science/r/G-Retriever.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.   
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.   
• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We specify all the training and test details in Appendix B.

Guidelines:

• The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: As shown in Table 3, we report the mean scores and standard deviations.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.   
The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96 \%$ CI, if the hypothesis of Normality of errors is not verified.   
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We reported the required compute resources in the Appendix E.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the broader impacts in Appendix A.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.   
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected.

Guidelines:

• The answer NA means that the paper does not use existing assets.   
• The authors should cite the original paper that produced the code package or dataset.   
• The authors should state which version of the asset is used and, if possible, include a URL.   
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.   
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.   
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.   
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.   
• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

# 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The new dataset introduced in the paper is accompanied by detailed documentation, including data collection methods, preprocessing steps, and usage instructions, as provided in Appendix C.

Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.