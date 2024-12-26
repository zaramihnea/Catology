# QueryClassifier.py
import google.generativeai as genai
import argparse
from typing import Any


def setup_llm() -> Any:
    """Configure and return the LLM model."""
    genai.configure(api_key="AIzaSyDEBKa-klOmaLZyXXUA-H1VnVvteEymFDk")
    return genai.GenerativeModel("gemini-1.5-flash")


def classify_query(query: str, model: Any) -> str:
    """Classify the type of cat-related query."""
    prompt = f"""
    Analyze this query: "{query}"

    If the user wants a cat breed description (e.g., "Tell me about Sphynx cats", "What are Persian cats like?"), respond with exactly: Description

    If the user wants to compare cat breeds (e.g., "Compare Siamese and Persian cats", "What's the difference between Maine Coon and Ragdoll?"), respond with exactly: Comparison

    If the user is describing a cat and wants to know its breed (e.g., "My cat is shy and affectionate", "What breed has these traits: intelligent, active?"), respond with exactly: Prediction

    If the query doesn't clearly fit any of these categories, generate a friendly, helpful response that:
    1. Acknowledges their question
    2. Explains what you can help with (breed descriptions, comparisons, or predictions)
    3. Invites them to ask a related question
    4. Uses a different phrasing each time
    5. Keeps the response concise (2-3 sentences)

    Examples of varied responses for invalid queries:
    - "While I can't help with cat food recommendations, I'd be happy to tell you about different cat breeds, compare breeds, or help identify a breed based on characteristics! What would you like to know?"
    - "That's an interesting question about cat behavior! I specialize in cat breeds - I can describe breeds, compare them, or help identify a breed based on traits. Would you like to explore any of those?"
    - "I focus on cat breeds rather than general cat care. I can tell you about specific breeds, compare different breeds, or help identify a breed based on characteristics - would any of those interest you?"

    Respond with ONLY one of the options (Description/Comparison/Prediction) or a varied help message as described above. No additional text.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(description='Classify cat-related queries')
    parser.add_argument('query', type=str, help='User query about cats')
    args = parser.parse_args()

    try:
        model = setup_llm()
        result = classify_query(args.query, model)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()