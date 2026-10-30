
import os
import re
import requests

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

WEBSITE_URL = "https://company-policy-assistant-icxf.onrender.com"

app = App(token=os.environ["SLACK_BOT_TOKEN"])


@app.event("app_mention")
def answer_policy_question(event, say):
    question = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
    thread_ts = event.get("thread_ts") or event["ts"]

    if not question:
        say(
            text="Please mention me with a company policy question.",
            thread_ts=thread_ts
        )
        return

    say(
        text="I'm checking the company policies. Please wait...",
        thread_ts=thread_ts
    )

    try:
        response = requests.post(
            WEBSITE_URL + "/compare",
            json={"question": question},
            timeout=180
        )
        response.raise_for_status()
        results = response.json()

        policy_result = results["results"]["llm_with_vector"]

        if policy_result.get("status") != "success":
            raise ValueError("The policy search was unsuccessful.")

        answer = policy_result["answer"]
        policy = policy_result["policy"]

        say(
            text=f"*Answer:* {answer}\n*Relevant Policy:* {policy}",
            thread_ts=thread_ts
        )

    except Exception as error:
        print("Bot error:", error, flush=True)
        say(
            text="Sorry, I couldn't retrieve the policy answer. "
                 "Please try again.",
            thread_ts=thread_ts
        )


if __name__ == "__main__":
    print("Starting Company Policy Assistant Slack Bot...")
    SocketModeHandler(
        app,
        os.environ["SLACK_APP_TOKEN"]
    ).start()
