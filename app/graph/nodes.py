from app.graph.query_normalizer import normalize_query
import logging
import re
from app.graph.state import GraphState
from app.services.retriever import hybrid_retriever
from app.tools.web_search import web_search
from app.services.gemini_service import gemini_service
from app.services.llm_service import llm_service
from app.services.prompt_builder import prompt_builder
from app.services.session_manager import session_manager
from app.tools.calculator import safe_calculate
from app.tools.image_generator import generate_image
from app.tools.news_validator import NewsValidator
from app.services.news_service import get_news
from app.services.news_processor import process_news_articles
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def query_normalizer_node(state: GraphState):

    original_query = state["question"]

    result = normalize_query(original_query)

    return {
        **state,
        "original_query": original_query,
        "normalized_query": result["normalized_query"],
        "language": result["language"]
    }


def theni_retriever_node(state: GraphState) -> GraphState:
    question = state.get("normalized_query") or state["question"]
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
    question = state.get("normalized_query") or state["question"]
    logger.info(f"[LangGraph] Executing Web Search for question: {question}")
    
    # 1. Fast bypass for conversational queries
    q_lower = question.strip().lower()
    is_conversational = (
        bool(re.search(r"^\s*(hi|hello|hey|sup|greetings|good morning|good afternoon|good evening)\b", q_lower))
        or bool(re.search(r"^\s*(bye|goodbye|thank you|thanks|thx|see you|cya|tq)\b", q_lower))
        or bool(re.search(r"^\s*(who are you|what are you|what can you do)\b", q_lower))
        or any(word in q_lower for word in ["stupid", "useless", "dumb", "idiot"])
    )
    if is_conversational:
        logger.info("[LangGraph] Conversational intent detected in web_search_node. Skipping search.")
        return {
            **state,
            "context": "",
            "sources": []
        }

    # 2. Primary: Google Gemini with native Google Search Grounding
    if gemini_service.is_available:
        try:
            logger.info("[LangGraph] Querying Gemini with native Google Search Grounding...")
            gem_res = gemini_service.search_and_answer(question)
            if gem_res.get("success") and gem_res.get("answer"):
                logger.info(
                    "[LangGraph] Gemini grounded search succeeded (%s) with %d sources.",
                    gem_res.get("model", "gemini"),
                    len(gem_res.get("sources", []))
                )
                return {
                    **state,
                    "answer": gem_res["answer"],
                    "sources": gem_res.get("sources", []),
                    "context": gem_res["answer"]
                }
            logger.warning(f"[LangGraph] Gemini search failed ({gem_res.get('error')}). Falling back to SerpApi...")
        except Exception as e:
            logger.warning(f"[LangGraph] Gemini search exception: {e}. Falling back to SerpApi...")

    # 3. Secondary fallback: SerpApi + Groq flow
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
            
        logger.info(f"[LangGraph] SerpApi web search returned {len(results)} results")
        
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

    question = state.get("normalized_query") or state["question"]

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
    question = state.get("normalized_query") or state["question"]

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

    
def parse_date_from_text(question: str) -> datetime | None:
    q = question.strip()

    # Pattern 1: DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YYYY
    m = re.search(r'\b(\d{1,2})[/\.\-](\d{1,2})[/\.\-](\d{4})\b', q)
    if m:
        p1, p2, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if p1 > 12 and p2 <= 12:
            d, m_val = p1, p2
        elif p2 > 12 and p1 <= 12:
            d, m_val = p2, p1
        else:
            d, m_val = p1, p2  # Default DD/MM/YYYY
        try:
            return datetime(y, m_val, d)
        except ValueError:
            pass

    # Pattern 2: YYYY-MM-DD or YYYY/MM/DD
    m = re.search(r'\b(\d{4})[/\.\-](\d{1,2})[/\.\-](\d{1,2})\b', q)
    if m:
        y, m_val, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, m_val, d)
        except ValueError:
            pass

    # Pattern 3: Month names (e.g. 28 October 2026, 28th October 2026, October 28, 2026)
    months = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    # "28th October 2026" or "28 October 2026"
    m = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)\s+(\d{4})\b', q, re.IGNORECASE)
    if m:
        d = int(m.group(1))
        month_str = m.group(2).lower()
        y = int(m.group(3))
        if month_str in months:
            try:
                return datetime(y, months[month_str], d)
            except ValueError:
                pass

    # "October 28, 2026" or "October 28 2026"
    m = re.search(r'\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b', q, re.IGNORECASE)
    if m:
        month_str = m.group(1).lower()
        d = int(m.group(2))
        y = int(m.group(3))
        if month_str in months:
            try:
                return datetime(y, months[month_str], d)
            except ValueError:
                pass

    return None

    
def datetime_node(state: GraphState) -> GraphState:
    question = state.get("normalized_query") or state["question"]
    logger.info(f"[LangGraph] Executing DateTime node for question: {question}")
    
    try:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        tz_name = "Asia/Kolkata"
    except Exception:
        now = datetime.now()
        tz_name = "Local Time"

    # Try deterministic regex date extraction first
    target_dt = parse_date_from_text(question)

    # If regex did not match, try LLM date extraction fallback for natural language dates
    if not target_dt:
        try:
            import json
            extraction_prompt = f"""
Reference date today: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}).
Question: {question}

Determine if the user is asking about a specific or relative date (e.g. "tomorrow", "yesterday", "28/10/2026", "Christmas 2026", etc.).
If yes, extract that date in ISO format (YYYY-MM-DD).
If the user is only asking for current date/time ("today", "now", "current date", "what time is it"), return "CURRENT".

Return ONLY JSON:
{{"type": "SPECIFIC", "date": "YYYY-MM-DD"}} OR {{"type": "CURRENT"}}
"""
            extracted_res = llm_service.generate(extraction_prompt, temperature=0.0)
            if extracted_res:
                json_match = re.search(r'\{.*\}', extracted_res, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    if data.get("type") == "SPECIFIC" and data.get("date"):
                        date_str = data.get("date").strip()
                        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception as e:
            logger.warning(f"[LangGraph] LLM date extraction fallback error: {str(e)}")

    if target_dt:
        day_of_week = target_dt.strftime("%A")
        date_formatted = target_dt.strftime("%Y-%m-%d")
        display_date = target_dt.strftime("%d %B %Y")
        
        logger.info(f"[LangGraph] Parsed target date: {date_formatted} ({day_of_week})")
        
        context_str = (
            f"Requested Target Date: {display_date} ({date_formatted})\n"
            f"Calculated Day of Week: {day_of_week}\n"
            f"Reference Today's Date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})"
        )
        
        result = {
            "requested_date": date_formatted,
            "day": day_of_week,
            "display_date": display_date
        }
    else:
        result = {
            "date": now.strftime("%Y-%m-%d"),
            "day": now.strftime("%A"),
            "time": now.strftime("%I:%M:%S %p"),
            "timezone": tz_name
        }
        
        logger.info(f"[LangGraph] DateTime result (current time): {result}")

        context_str = (
            f"Current Date: {result['date']}\n"
            f"Day of Week: {result['day']}\n"
            f"Current Time: {result['time']}\n"
            f"Timezone: {result['timezone']}"
        )

    return {
        **state,
        "tool_result": result,
        "context": context_str
    }

def current_news_node(state: GraphState) -> GraphState:
    question = state.get("normalized_query") or state["question"]
    location = state.get("location") or "General"
    time_range = state.get("time_range") or "today"
    
    logger.info(
        f"[LangGraph] Executing Current News Node | Query: '{question}' | "
        f"Location: '{location}' | TimeRange: '{time_range}'"
    )

    # 1. GNews Multi-Query Discovery (Step 5)
    gnews_candidates = get_news(location=location, max_results=15)
    
    # 2. Extract original webpage text, evaluate location relevance, source reliability & deduplicate (Steps 3-13)
    verified_articles = process_news_articles(
        articles=gnews_candidates,
        location=location,
        days=7,
        min_location_relevance=0.60
    )

    # 3. Fallback to web_search if GNews returned 0 scraped articles
    if not verified_articles:
        logger.info(f"[Current News Node] GNews returned 0 verified articles for '{location}'. Trying fallback web search...")
        fallback_raw = web_search(f"{location} local news headlines today")
        try:
            now_date = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            now_date = datetime.now().date()
            
        accepted_fallback, audit_logs = NewsValidator.filter_and_validate(
            results=fallback_raw,
            user_query=question,
            location=location,
            time_range=time_range,
            reference_date=now_date
        )
        if accepted_fallback:
            for item in accepted_fallback:
                pub_date_item = item.get("published_date") or item.get("display_date") or "Unknown"
                verified_articles.append({
                    "title": item["title"],
                    "source": item["source"],
                    "published_date": pub_date_item,
                    "event_date": pub_date_item,
                    "event_status": "RECENT",
                    "url": item["url"],
                    "content": item["content"],
                    "location_relevance": 0.80,
                    "source_reliability": 0.70,
                    "composite_score": 0.75,
                    "verification_status": "verified"
                })

    # Build context for LLM grounding using Structured Article Model
    if not verified_articles:
        context = f"NO_VERIFIED_NEWS: I couldn't verify reliable current news for '{location}'."
        sources = []
    else:
        context_parts = []
        sources = []
        for res in verified_articles[:5]:
            content_text = res.get('content') or ""
            content_excerpt = content_text[:450].strip()
            pub_date_val = res.get('published_date') or "Unknown"
            event_date_val = res.get('event_date') or pub_date_val
            event_status_val = res.get('event_status') or "Recent"

            context_parts.append(
                f"Headline: {res['title']}\n"
                f"Source: {res['source']}\n"
                f"Published Date: {pub_date_val}\n"
                f"Event Date: {event_date_val}\n"
                f"Event Status: {event_status_val}\n"
                f"Location Relevance: {location}\n"
                f"Original Article URL: {res['url']}\n"
                f"Article Excerpt: {content_excerpt}..."
            )
            sources.append({
                "title": res["title"],
                "url": res["url"],
                "source": res["source"],
                "published_date": pub_date_val,
                "event_status": event_status_val
            })
        context = "\n\n".join(context_parts)

    return {
        **state,
        "context": context,
        "sources": sources
    }

def generate_answer_node(state: GraphState) -> GraphState:
    question = state["question"]
    normalized_query = state.get("normalized_query") or question
    language = state.get("language", "english")
    context = state["context"]
    route = state.get("route", "general")
    session_id = state.get("session_id", "default")
    location = state.get("location") or "the requested location"
    time_range = state.get("time_range") or "today"
    
    logger.info(f"[LangGraph] Generating answer with Groq (Route: {route})")
    
    try:
        import re
        q_lower = question.strip().lower()
        
        # 1. Greetings
        if re.search(r"^\s*(hi|hello|hey|sup|greetings|good morning|good afternoon|good evening)\b", q_lower):
            answer = "Hello! I am the District Information Assistant. How may I help you with district-related or global information today?"
            session_manager.add_user_message(session_id, question)
            session_manager.add_assistant_message(session_id, answer)
            return {**state, "answer": answer, "sources": []}
            
        # 2. Farewells & Thanks
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

        if route == "current_news":
            if "NO_VERIFIED_NEWS" in context:
                answer = f"I couldn't verify reliable current news for {location}."
            else:
                try:
                    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
                    today_str = now_ist.strftime("%B %d, %Y")
                except Exception:
                    today_str = "Today"

                prompt = f"""
You are a grounded, professional news reporting assistant.

Formulate a concise and accurate news report using ONLY the verified news items supplied below.

CRITICAL DATE RULES:
1. PRESERVE ORIGINAL PUBLICATION DATES: Always report the exact Published Date provided in the context (e.g., "October 6, 2026"). NEVER convert yesterday's or an older publication date into "Date: Today" or "Today".
2. TODAY'S DATE REFERENCE: Today is {today_str}. Only label an article as published "Today" if its Published Date in the context is explicitly "{today_str}".
3. EVENT STATUS: State the Event Status (e.g. Current, Ongoing, Recent, Upcoming) as given in the context.
4. ACCURACY & ZERO FABRICATION: Report ONLY the verified news items supplied in the context below. If only 2 or 3 verified stories exist, report ONLY those 2 or 3 verified stories. Do NOT invent news, dates, sources, or URLs.
5. FORMAT REQUIREMENTS: Format each verified news item as follows:
   #### [Headline Title]
   [Concise factual summary in 1-3 sentences]
   - **Source:** [Publisher Name]
   - **Published:** [Exact Published Date from context]
   - **Status:** [Event Status]
   - **Relevance:** [{location}]
   - **Original Article:** [Read full article](Original Article URL)

Verified News Context:
{context}

Original User Question:
{question}

Normalized Question:
{normalized_query}

User Language:
{language}

Answer:
"""
                answer = llm_service.generate(prompt, temperature=0.0)

            # If news items did not answer the user's specific question, fall back to live Gemini / Web Search!
            cannot_answer_phrases = [
                "no information regarding", "cannot answer this question",
                "couldn't verify", "no verified news", "not mentioned in the provided",
                "does not include any news"
            ]
            if any(phrase in answer.lower() for phrase in cannot_answer_phrases):
                logger.info(f"[LangGraph] News items did not answer question '{question}'. Falling back to Gemini/Web Search...")
                if gemini_service.is_available:
                    gem_res = gemini_service.search_and_answer(question)
                    if gem_res.get("success") and gem_res.get("answer"):
                        answer = gem_res["answer"]
                        state["sources"] = gem_res.get("sources", [])

        elif route == "theni":
            # For theni, reuse existing prompt_builder and memory
            context_data = session_manager.get_session_context(session_id)
            history = context_data.get("history", [])
            summary = context_data.get("summary", "")
            
            prompt = prompt_builder.build_prompt(
                question=question,
                context=context,
                history=history,
                summary=summary,
                normalized_query=normalized_query,
                language=language
            )
            answer = llm_service.generate(prompt)

            # Fallback to Web Search if local district documents do not contain the answer
            if "not available in the district records" in answer.lower() or "not available" in answer.lower():
                logger.info(f"[LangGraph] Information not found in district records. Falling back to Web Search for: {question}")
                gemini_handled = False
                if gemini_service.is_available:
                    try:
                        gem_res = gemini_service.search_and_answer(normalized_query)
                        if gem_res.get("success") and gem_res.get("answer"):
                            answer = gem_res["answer"]
                            state["sources"] = gem_res.get("sources", [])
                            gemini_handled = True
                            logger.info("[LangGraph] Gemini handled fallback search successfully.")
                    except Exception as e:
                        logger.warning(f"[LangGraph] Gemini fallback search error: {e}")

                if not gemini_handled:
                    web_results = web_search(normalized_query)
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
- Make the response easy to read.
- Do not invent facts.
- Respond in the same language/style as the user's original question whenever possible.

Question:
{question}

Web Search Results:
{web_context}

Answer:
"""

                        answer = llm_service.generate(fallback_prompt)
                        state["sources"] = web_sources
        else:
            # If answer was already generated (e.g. by Gemini Search Grounding in web_search_node)
            if state.get("answer"):
                answer = state["answer"]
            else:
                # For general/current/arithmetic questions
                prompt = f"""
You are a helpful and professional AI assistant.

Answer the user's question using ONLY the information provided in the context.

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
