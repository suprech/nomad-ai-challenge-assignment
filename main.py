import json
import openai
import requests
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()

BASE_URL = "https://nomad-movies.nomadcoders.workers.dev"


# 함수 정의
# 인기 영화 목록을 가져온다.
def get_popular_movies():
    return requests.get(f"{BASE_URL}/movies").json()


# 영화 상세 정보를 가져온다.
def get_movie_details(id):
    return requests.get(f"{BASE_URL}/movies/{id}").json()


# 영화 출연진/제작진 정보를 가져온다.
def get_movie_credits(id):
    return requests.get(f"{BASE_URL}/movies/{id}/credits").json()


# 함수 매핑
FUNCTION_MAP = {
    "get_popular_movies": get_popular_movies,
    "get_movie_details": get_movie_details,
    "get_movie_credits": get_movie_credits,
}

# AI에게 알려줄 도구(tools) 정의
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_popular_movies",
            "description": "Get a list of currently popular movies.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": "Get detailed information about a specific movie by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The movie ID",
                    }
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_credits",
            "description": "Get the cast and crew of a specific movie by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The movie ID",
                    }
                },
                "required": ["id"],
            },
        },
    },
]

messages = []
messages.append(
    {
        "role": "system",
        "content": "You are a movie expert agent. Answer questions about movies using the provided tools. Reply in the same language the user uses.",
    }
)


def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    message = response.choices[0].message
    process_ai_response(message)


def process_ai_response(message):
    if message.tool_calls:
        # AI의 tool_calls 응답을 messages에 기록
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )
        # 각 tool_call에 대해 실제 함수 실행
        for tc in message.tool_calls:
            fn_name = tc.function.name
            arguments = tc.function.arguments
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"Function Calling {fn_name}({args})")

            function_to_run = FUNCTION_MAP.get(fn_name)
            result = function_to_run(**args)

            # 실행 결과를 role: "tool"로 추가
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        # 도구 결과를 포함해서 AI 다시 호출
        call_ai()
    else:
        # 일반 텍스트 응답
        messages.append({"role": "assistant", "content": message.content})
        print(f"AI: {message.content}")


def main():
    print("Movie Agent (종료: q)")
    while True:
        user_input = input("\nYou: ")
        if user_input in ("q", "quit"):
            break
        messages.append({"role": "user", "content": user_input})
        call_ai()


if __name__ == "__main__":
    main()
