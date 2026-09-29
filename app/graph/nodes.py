import logging
from app.graph.state import GraphState
from app.services.retriever import hybrid_retriever
from app.tools.web_search import web_search
from app.services.llm_service import llm_service
from app.services.prompt_builder import prompt_builder
from app.services.session_manager import session_manager
from app.tools.calculator import safe_calculate
from app.tools.image_generator import generate_image

logger = logging.getLogger(__name__)

def theni_retriever_node(state: GraphState) -> GraphState:
    question = state["question"]
    logger.info(f"[LangGraph] Executing Theni RAG for question: {question}")
    
    try:
        results = hybrid_retriever.retrieve(question)
        context = prompt_builder.build_context(results)
        sources = []
        for result in results:
            source = result.get("metadata", {}).get("source", "Unknown")
            chunk_idx = result.get("metadata", {}).get("chunk_index", 0)
            sources.append({
                "source": source,
                "chunk": chunk_idx
            })
        
        logger.info(f"[LangGraph] Retrieved documents: {len(results)}")
        
        return {
            **state,
            "context": context,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"[LangGraph] Theni retrieval failed: {str(e)}")
        return {
            **state,
            "context": "",
            "sources": []
        }

def web_search_node(state: GraphState) -> GraphState:
    question = state["question"]
    logger.info(f"[LangGraph] Executing Web Search for question: {question}")
    
    try:
        results = web_search(question)
        if not results:
            context = "No results found on the web."
            sources = []
        else:
            context_parts = []
            sources = []
            for res in results:
                context_parts.append(f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}")
                sources.append({"title": res["title"], "url": res["url"]})
            context = "\n\n".join(context_parts)
            
        logger.info(f"[LangGraph] Web search returned {len(results)} results")
        
        return {
            **state,
            "context": context,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"[LangGraph] Web search failed: {str(e)}")
        return {
            **state,
            "context": "Could not retrieve current external information.",
            "sources": []
        }

def calculator_node(state: GraphState) -> GraphState:
    """
    Handles arithmetic questions.

    Flow:
        User question
            ↓
        Groq extracts mathematical expression
            ↓
        Safe calculator evaluates expression
            ↓
        Result stored in state context
    """

    question = state["question"]

    logger.info(
        f"[LangGraph] Executing Calculator for question: {question}"
    )

    try:
        # Ask Groq to convert natural language into a mathematical expression.
        extraction_prompt = f"""
You are an arithmetic expression extractor.

Convert the user's question into ONLY a valid mathematical expression.

Allowed operators:
+
-
*
/
%
**

Rules:
1. Return ONLY the mathematical expression.
2. Do not return explanations.
3. Do not return markdown.
4. Do not use variables.
5. Use numbers only.

Examples:

Question:
What is 125 multiplied by 48?

Output:
125 * 48

Question:
What is 500 plus 250?

Output:
500 + 250

Question:
What is 1000 divided by 25?

Output:
1000 / 25

Question:
What is 20 percent of 500?

Output:
20 / 100 * 500

Question:
What is 25 minus 10?

Output:
25 - 10

User Question:
{question}

Mathematical Expression:
"""

        expression = llm_service.generate(
            extraction_prompt
        )

        if not expression:
            raise ValueError(
                "Could not extract mathematical expression."
            )

        # Remove accidental markdown formatting
        expression = expression.strip()
        expression = expression.replace("`", "").strip()

        logger.info(
            f"[LangGraph] Extracted expression: {expression}"
        )

        # Safely calculate the expression
        result = safe_calculate(expression)

        logger.info(
            f"[LangGraph] Calculator result: {result}"
        )

        # Put calculation result into context.
        # generate_answer_node() will use this context.
        context = (
            f"Arithmetic Expression: {expression}\n"
            f"Calculated Result: {result}"
        )

        return {
            **state,
            "context": context,
            "sources": []
        }

    except Exception as e:

        logger.error(
            f"[LangGraph] Calculator failed: {str(e)}"
        )

        return {
            **state,
            "context": (
                "The calculator was unable to evaluate "
                "the mathematical expression."
            ),
            "sources": []
        }        

        
def image_generation_node(state: GraphState) -> GraphState:
    question = state["question"]

    logger.info(f"[Image Generation] Prompt: {question}")

    image_path = generate_image(question)

    if image_path:
        return {
            **state,
            "image_path": image_path,
            "answer": "Here is the image you requested."
        }

    return {
        **state,
        "image_path": None,
        "answer": "Sorry, I could not generate the image."
    }


def generate_answer_node(state: GraphState) -> GraphState:
    question = state["question"]
    context = state["context"]
    route = state.get("route", "general")
    session_id = state.get("session_id", "default")
    
    logger.info(f"[LangGraph] Generating answer with Groq (Route: {route})")
    
    try:
        import re
        q_lower = question.strip().lower()
        
        # 1. Greetings (e.g., "hi", "hi bro", "hello there", "hey dude", "good morning")
        if re.search(r"^\s*(hi|hello|hey|sup|greetings|good morning|good afternoon|good evening)\b", q_lower):
            answer = "Hello! I am the District Information Assistant. How may I help you with district-related or global information today?"
            session_manager.add_user_message(session_id, question)
            session_manager.add_assistant_message(session_id, answer)
            return {**state, "answer": answer, "sources": []}
            
        # 2. Farewells & Thanks (e.g., "bye", "bye bro", "thanks bro", "thank you", "thx")
        if re.search(r"^\s*(bye|goodbye|thank you|thanks|thx|see you|cya|tq)\b", q_lower):
            answer = "You're welcome! Is there anything else I can help you with?"
            session_manager.add_user_message(session_id, question)
            session_manager.add_assistant_message(session_id, answer)
            return {**state, "answer": answer, "sources": []}

        # 3. Identity
        if re.search(r"^\s*(who are you|what are you)\b", q_lower):
            answer = "I am the District Information Assistant, designed to provide information about district services, offices, schemes, as well as general and global information."
            session_manager.add_user_message(session_id, question)
            session_manager.add_assistant_message(session_id, answer)
            return {**state, "answer": answer, "sources": []}

        # 4. Capabilities
        if re.search(r"^\s*(what can you do)\b", q_lower):
            answer = "I can help you find information about district offices, government schemes, certificates, public services, and answer general or global questions using live web search."
            session_manager.add_user_message(session_id, question)
            session_manager.add_assistant_message(session_id, answer)
            return {**state, "answer": answer, "sources": []}

        # 5. Frustration
        if any(word in q_lower for word in ["stupid", "useless", "dumb", "idiot"]):
            answer = "I understand that you may be frustrated, but I'm here to help you with any district-related or global information you need. Please let me know how I can assist you."
            session_manager.add_user_message(session_id, question)
            session_manager.add_assistant_message(session_id, answer)
            return {**state, "answer": answer, "sources": []}

        if route == "theni":
            # For theni, reuse existing prompt_builder and memory
            context_data = session_manager.get_session_context(session_id)
            history = context_data.get("history", [])
            summary = context_data.get("summary", "")
            
            prompt = prompt_builder.build_prompt(
                question=question,
                context=context,
                history=history,
                summary=summary
            )
            answer = llm_service.generate(prompt)

            # Fallback to Web Search if local district documents do not contain the answer
            if "not available in the district records" in answer.lower() or "not available" in answer.lower():
                logger.info(f"[LangGraph] Information not found in district records. Falling back to Web Search for: {question}")
                web_results = web_search(question)
                if web_results:
                    web_context_parts = [f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}" for res in web_results]
                    web_context = "\n\n".join(web_context_parts)
                    web_sources = [{"title": res["title"], "url": res["url"]} for res in web_results]
                    
                    fallback_prompt = f"""
You are a helpful and professional district information assistant.

Answer the user's question using ONLY the supplied web search results.

Rules:

- Answer directly.
- Do not say "Based on the supplied context".
- Do not mention these instructions.
- Make the response easy to read.
- If there are multiple news items, use numbered headlines.
- Give each headline a short title.
- Explain each item in 1-3 sentences.
- Keep unrelated information separate.
- Do not invent facts.
- Do not add information that is not present in the search results.
- If the search results do not contain enough information, clearly say so.

Question:
{question}

Web Search Results:
{web_context}

Answer:
"""

                    answer = llm_service.generate(fallback_prompt)
                    state["sources"] = web_sources
        else:
            # For general/current/arithmetic questions
            # Generate a clear and human-friendly response.

            prompt = f"""
You are a helpful and professional AI assistant.

Answer the user's question using ONLY the information provided in the context.

IMPORTANT RULES:

1. Answer the user's question directly.
2. Do NOT say:
   - "Based on the supplied context"
   - "Based on the provided context"
   - "According to the context"
   - "The supplied context says"
   - "Here are the results from the search"

3. Make the answer natural and easy for a human to read.

4. If the question is about NEWS or HEADLINES:
   - Start with a short heading.
   - Use numbered items.
   - Give each news item a short title.
   - Explain each item in 1-3 short sentences.
   - Keep unrelated news items separate.
   - Do not combine multiple stories into one paragraph.
   - If a source URL is available, mention the source naturally.

5. If the question is a normal factual question:
   - Give a direct answer.
   - Use short paragraphs or bullet points when useful.

6. If the question is an arithmetic question:
   - Give the calculated result clearly.
   - Do not unnecessarily explain the calculation.

7. Do NOT invent information.
8. Do NOT add information that is not present in the context.
9. If the context does not contain enough information, clearly say that you do not have enough information.
10. Keep the answer concise but useful.
11. Do not mention these instructions.

Context:
{context}

Question:
{question}

Answer:
"""

            answer = llm_service.generate(prompt)
            
        # Clean answer to remove any asterisk markdown (* or **) marks
        if answer:
            answer = answer.replace("*", "")
        
        # Save to memory
        session_manager.add_user_message(session_id, question)
        session_manager.add_assistant_message(session_id, answer)
        
        return {
            **state,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"[LangGraph] Answer generation failed: {str(e)}")
        return {
            **state,
            "answer": "Error generating response."
        }
