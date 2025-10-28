# the-fantastic-four---AI-Medical-Chatbot


#  MediCare RAG: Citation-Backed Medical Q&A

**Team:** The Fantastic Four

> **Submission for:** Hack A Cure 
> **Problem Statement:** Build a medical chatbot/Q&A system using RAG that provides accurate, explainable, and citation-backed answers.

---

## 👥 1. Team & Roles

Our four-member team is structured to efficiently tackle the RAG pipeline, aiming for high performance and strict adherence to the **citation-backed** requirement.


| **Dheeraj Rangu** | **Backend & Deployment Lead** | FastAPI API development, endpoint deployment, and final URL generation. |
| **Ramisetty Amrutha** | **Data Engineering Lead** | Data ingestion, document chunking, and managing the ChromaDB vector store. |
| **Shaik Mehajabeen** | **RAG Logic & Prompt Engineer** | Designing the LangChain RAG flow, LLM selection, and prompt engineering for faithfulness. |
| **Likith Sankarnarayana** | **Evaluation & Documentation Lead** | RAGAS testing, documenting architecture, and ensuring submission requirements are met. |

---

##  2. Technical Strategy & Planned Stack

We are adopting a robust, open-source stack that prioritizes speed and integration for a 24-hour delivery.

### Core Technologies
| Component | Planned Technology | Role in the System |
| :--- | :--- | :--- |
| **RAG Framework** | **LangChain** | The primary orchestration layer. |
| **Vector Store** | **ChromaDB (Local)** | For fast, efficient similarity search on the medical dataset. |
| **Embedding Model** | **`all-MiniLM-L6-v2`** | Selected for its speed and balance of performance for semantic search. |
| **LLM / Generator** | **[To Be Finalized - Open-Source Option]** | We will select a suitable, fast LLM (e.g., a Mistral variant) that can run locally or via a free tier API, focused on instruction-following. |
| **Backend API** | **FastAPI** | To serve the required `/query` endpoint quickly. |

### Success Criteria Focus
Our two primary goals, aligned with the judging criteria, are:
1.  **RAGAS Score:** Maximizing **Faithfulness** and **Context Recall** through careful chunking and strict prompt design.
2.  **Citation Compliance:** Guaranteeing every answer includes accurate source metadata (Document Name, Page Number).

---

##  3. Pre-Hackathon Setup Checklist


| Task | Assignee | Status |


---

##  4. 24-Hour Execution Timeline (Draft)


| Phase | Time Allotment | Key Deliverables | Assigned Lead |
| :--- | :--- | :--- | :--- |
| **Ingestion & Vectorization (Steps 1 & 2)** | 7 hours | Finalized `ingestion.py`, persistent ChromaDB created. | Amrutha |
| **RAG Chain Assembly (Step 3)** | 7 hours | Working RAG logic, optimized LLM prompt for citations. | Mehajabeen |
| **API Wrapper & Deployment (Step 4)** | 6 hours | Deployed FastAPI endpoint, live testing. | Dheeraj |
| **Evaluation & Polish (Step 5)** | 4 hours | RAGAS score optimization, final documentation, demo video. | Likith |
