import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=10,
    messages=[
        {"role": "user", "content": "Say hello"}
    ]
)

print(message.content[0].text)