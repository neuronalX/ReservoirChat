# ReservoirChat

📄 [Arxiv Paper](https://www.arxiv.org/abs/2507.05279)  

ReservoirChat is a specialized AI model designed to enhance coding assistance and provide accurate answers to complex questions in the field of Reservoir Computing. This project leverages a Large Language Model (LLM) augmented with Retrieval-Augmented Generation (RAG) and Knowledge Graphs to reduce hallucinations and improve the reliability of generated responses.

## Overview

ReservoirChat focuses on the Reservoir Computing paradigm and the `reservoirPy` library. Unlike traditional LLMs, ReservoirChat integrates retrieval mechanisms (RAG & GraphRAG), delivering precise and reliable responses based on external data sources while leveraging the capabilities of large language models.

## Key Features

- **Retrieval-Augmented Generation (RAG)**: Enhances the LLM with a database of relevant documents, including ReservoirPy documentation, research papers, and code samples.
- **Knowledge Graph**: Structures information into entities and relationships, improving contextual and semantic accuracy.
- **Interactive Documentation**: Provides an interactive assistant for users working with the ReservoirPy library, helping with debugging and suggesting relevant hyperparameters.

## Knowledge Graph, GraphRAG, and ReservoirChat

The Knowledge Graph method structures data into entities and relationships, enhancing the contextual accuracy of responses. GraphRAG handles large volumes of data, such as scientific papers, and builds an undirected graph to capture the relationships between entities. 

## Evaluation

ReservoirChat's performance was evaluated using a custom-made benchmark that measures the precision and reliability of responses in the context of Reservoir Computing. The model was compared with other mainstream models, including ChatGPT-4o, Llama3, Codestral, and NotebookLM.

## Results

ReservoirChat outperformed its base model, Codestral, in both knowledge-based and coding tasks and concurrence other SOTA models. The model demonstrated significant improvements in response accuracy and reliability, particularly in domain-specific tasks related to Reservoir Computing.

## Future Directions

Continuous addition of new scientific publications to the knowledge graph will keep ReservoirChat aligned with advances in the field. Future work will focus on optimizing response time and quality, ensuring safety, and verifying data fairness.

## Reference

> @misc{boraud2025reservoirchatinteractivedocumentationenhanced,  
>       title={ReservoirChat: Interactive Documentation Enhanced with LLM and Knowledge Graph for ReservoirPy},   
>       author={Virgile Boraud and Yannis Bendi-Ouis and Paul Bernard and Xavier Hinaut},  
>       year={2025},  
>       eprint={2507.05279},  
>       archivePrefix={arXiv},  
>       primaryClass={cs.SE},  
>       url={https://arxiv.org/abs/2507.05279},   
> }


## License

This project is licensed under the GNU General Public License v3.0 License. See the LICENSE file for details.

## Acknowledgments

This work was supported by Inria and the Mnemosyne project.

---

For more information, please visit the [ReservoirChat website](https://chat.reservoirpy.inria.fr/).
